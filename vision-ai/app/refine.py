"""CLIP zero-shot reclassification of YOLO crops.

YOLO26 COCO often labels tanks as boat and drones as helicopter/airplane.
We re-score ambiguous (or all) boxes against a tactical label set.
"""
from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("refine")

# (display_name, CLIP prompt)
CANDIDATES: list[tuple[str, str]] = [
    ("tank", "a photo of a military battle tank"),
    ("armored vehicle", "a photo of an armored personnel carrier"),
    ("military truck", "a photo of a military truck"),
    ("drone", "a photo of a small UAV drone in the sky"),
    ("helicopter", "a photo of a helicopter"),
    ("fighter jet", "a photo of a fighter jet"),
    ("boat", "a photo of a boat on water"),
    ("ship", "a photo of a large ship"),
    ("person", "a photo of a person"),
    ("soldier", "a photo of a soldier in uniform"),
    ("car", "a photo of a civilian car"),
    ("truck", "a photo of a civilian truck"),
    ("airplane", "a photo of a passenger airplane"),
    ("building", "a photo of a building"),
]

# COCO labels that commonly collide with tactical objects
AMBIGUOUS = {
    "boat", "airplane", "helicopter", "truck", "bus", "train",
    "surfboard", "kite", "sports ball", "bird", "frisbee",
}

_REFINE_ENABLED = os.getenv("REFINE_CROPS", "true").lower() in ("1", "true", "yes")
_REFINE_ALL = os.getenv("REFINE_ALL", "false").lower() in ("1", "true", "yes")
_REFINE_MAX = int(os.getenv("REFINE_MAX", "8"))
_REFINE_MIN_CONF = float(os.getenv("REFINE_MIN_CONF", "0.18"))
# Only replace YOLO label when CLIP is at least this much more confident, or YOLO was ambiguous
_REPLACE_MARGIN = float(os.getenv("REFINE_REPLACE_MARGIN", "0.02"))

_lock = threading.Lock()
_model = None
_preprocess = None
_tokenizer = None
_text_features = None
_device = "cpu"


def enabled() -> bool:
    return _REFINE_ENABLED


def warmup() -> bool:
    """Load CLIP once (call at startup). Returns False if unavailable."""
    if not _REFINE_ENABLED:
        return False
    try:
        _get_model()
        return True
    except Exception as e:
        log.warning("CLIP refine unavailable: %s", e)
        return False


def _get_model():
    global _model, _preprocess, _tokenizer, _text_features, _device
    with _lock:
        if _model is not None:
            return _model, _preprocess, _tokenizer, _text_features

        import open_clip
        import torch

        _device = "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        model.to(_device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        prompts = [p for _, p in CANDIDATES]
        with torch.no_grad():
            text = tokenizer(prompts)
            text_features = model.encode_text(text.to(_device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        _model = model
        _preprocess = preprocess
        _tokenizer = tokenizer
        _text_features = text_features
        log.info("CLIP refine ready (%d labels)", len(CANDIDATES))
        return _model, _preprocess, _tokenizer, _text_features


def _crop(image_bgr: np.ndarray, bbox: list[float], pad_frac: float = 0.12) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = bw * pad_frac, bh * pad_frac
    xa = max(0, int(x1 - px))
    ya = max(0, int(y1 - py))
    xb = min(w, int(x2 + px))
    yb = min(h, int(y2 + py))
    if xb <= xa or yb <= ya:
        return None
    return image_bgr[ya:yb, xa:xb]


def _classify_crop(crop_bgr: np.ndarray) -> tuple[str, float]:
    import torch
    from PIL import Image

    model, preprocess, _, text_features = _get_model()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    image_in = preprocess(pil).unsqueeze(0).to(_device)
    with torch.no_grad():
        image_features = model.encode_image(image_in)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits[0].cpu().numpy()
    idx = int(probs.argmax())
    return CANDIDATES[idx][0], float(probs[idx])


def refine_detections(image_bgr: np.ndarray, detections: list[dict]) -> list[dict]:
    """Return new detection list with class_name possibly corrected; adds detector_class."""
    if not _REFINE_ENABLED or not detections:
        return detections

    try:
        _get_model()
    except Exception as e:
        log.warning("refine skipped: %s", e)
        return detections

    # Highest-confidence first, cap work
    ordered = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
    out: list[dict] = []
    refined_n = 0

    for det in ordered:
        d = dict(det)
        d["detector_class"] = det.get("class_name", "")
        yolo_name = (det.get("class_name") or "").lower()
        should = _REFINE_ALL or yolo_name in AMBIGUOUS

        if should and refined_n < _REFINE_MAX:
            crop = _crop(image_bgr, det.get("bbox") or [0, 0, 0, 0])
            if crop is not None and crop.size > 0:
                try:
                    label, conf = _classify_crop(crop)
                    refined_n += 1
                    d["refine_confidence"] = round(conf, 4)
                    d["refine_class"] = label
                    if conf >= _REFINE_MIN_CONF:
                        # Ambiguous COCO hits (boat/heli/…) → always take CLIP label
                        if yolo_name in AMBIGUOUS:
                            d["class_name"] = label
                        elif label != yolo_name and conf >= float(det.get("confidence", 0)) + _REPLACE_MARGIN:
                            d["class_name"] = label
                except Exception as e:
                    log.debug("crop classify failed: %s", e)

        out.append(d)

    # Preserve original order (by confidence desc as returned by detector)
    return out
