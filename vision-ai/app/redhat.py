"""PoC: CLIP zero-shot 'wearing a red hat / red fedora' on YOLO person crops.

Does not fine-tune YOLO. Uses existing open_clip stack (same as refine.py).
"""
from __future__ import annotations

import logging
import os
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("redhat")

# Positive = Red Hat *fedora* only (branded soft fedora / Shadowman-style), not caps
# Negative = bare head, baseball/trucker caps (incl. red), beanies, other hats, red clothes
POSITIVE: list[tuple[str, str]] = [
    ("rh_fedora", "a photo of a person wearing a Red Hat brand red fedora hat"),
    ("rh_fedora_side", "a photo of a bright red fedora with a wide brim on a person's head"),
    ("rh_merch", "a photo of official Red Hat fedora merch worn on someone's head"),
]
NEGATIVE: list[tuple[str, str]] = [
    ("no_hat", "a photo of a person with a bare head and no hat"),
    ("red_baseball_cap", "a photo of a person wearing a red baseball cap"),
    ("red_trucker_cap", "a photo of a person wearing a red trucker mesh cap"),
    ("black_baseball_cap", "a photo of a person wearing a black baseball cap"),
    ("beanie", "a photo of a person wearing a knit beanie"),
    ("other_fedora", "a photo of a person wearing a brown or black fedora"),
    ("cowboy_hat", "a photo of a person wearing a cowboy hat"),
    ("red_shirt", "a photo of a person in a red shirt with no hat on their head"),
    ("red_jacket", "a photo of a person in a red jacket with no hat on their head"),
    ("red_hair", "a photo of a person with red hair and no fedora"),
]

_LABELS = POSITIVE + NEGATIVE
_POS_NAMES = {n for n, _ in POSITIVE}

_ENABLED = os.getenv("REDHAT_DETECT", "true").lower() in ("1", "true", "yes")
_MIN_PERSON_CONF = float(os.getenv("REDHAT_MIN_PERSON_CONF", "0.35"))
# Slightly stricter — fedora-only prompts; caps should lose to red_baseball_cap / etc.
_MIN_CLIP_CONF = float(os.getenv("REDHAT_MIN_CLIP_CONF", "0.30"))
_MIN_POS_MARGIN = float(os.getenv("REDHAT_POS_MARGIN", "0.15"))  # pos_mass - neg_mass
_MAX_PERSONS = int(os.getenv("REDHAT_MAX_PERSONS", "6"))
_PERSON_CLASSES = {
    c.strip().lower()
    for c in os.getenv("REDHAT_PERSON_CLASSES", "person,soldier").split(",")
    if c.strip()
}

_lock = threading.Lock()
_model = None
_preprocess = None
_text_features = None
_device = "cpu"


def enabled() -> bool:
    return _ENABLED


def warmup() -> bool:
    if not _ENABLED:
        return False
    try:
        _get_model()
        return True
    except Exception as e:
        log.warning("redhat CLIP unavailable: %s", e)
        return False


def _get_model():
    global _model, _preprocess, _text_features, _device
    with _lock:
        if _model is not None:
            return _model, _preprocess, _text_features

        import open_clip
        import torch

        _device = "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        model.to(_device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        prompts = [p for _, p in _LABELS]
        with torch.no_grad():
            text = tokenizer(prompts)
            text_features = model.encode_text(text.to(_device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        _model = model
        _preprocess = preprocess
        _text_features = text_features
        log.info("redhat CLIP ready (%d prompts)", len(_LABELS))
        return _model, _preprocess, _text_features


def _hat_region_crop(image_bgr: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Crop upper person region (head/hat), with upward pad."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # Head/hat ≈ top 40% of person box; pad up 25% of that height
    top = y1
    bottom = y1 + bh * 0.42
    pad_y = bh * 0.12
    pad_x = bw * 0.08
    xa = max(0, int(x1 - pad_x))
    ya = max(0, int(top - pad_y))
    xb = min(w, int(x2 + pad_x))
    yb = min(h, int(bottom + pad_y * 0.3))
    if xb <= xa or yb <= ya:
        return None
    return image_bgr[ya:yb, xa:xb]


def _score_crop(crop_bgr: np.ndarray) -> dict:
    import torch
    from PIL import Image

    model, preprocess, text_features = _get_model()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    image_in = preprocess(pil).unsqueeze(0).to(_device)
    with torch.no_grad():
        image_features = model.encode_image(image_in)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0].cpu().numpy()

    ranked = sorted(
        [
            {"label": _LABELS[i][0], "prompt": _LABELS[i][1], "prob": float(probs[i])}
            for i in range(len(_LABELS))
        ],
        key=lambda x: x["prob"],
        reverse=True,
    )
    pos_mass = float(sum(p["prob"] for p in ranked if p["label"] in _POS_NAMES))
    neg_mass = float(1.0 - pos_mass)
    top = ranked[0]
    wearing = (
        top["label"] in _POS_NAMES
        and top["prob"] >= _MIN_CLIP_CONF
        and (pos_mass - neg_mass) >= _MIN_POS_MARGIN
    )
    return {
        "wearing_red_hat": wearing,
        "top_label": top["label"],
        "top_prob": round(top["prob"], 4),
        "pos_mass": round(pos_mass, 4),
        "neg_mass": round(neg_mass, 4),
        "scores": [
            {"label": r["label"], "prob": round(r["prob"], 4)} for r in ranked[:5]
        ],
    }


def annotate(image_bgr: np.ndarray, hits: list[dict]) -> np.ndarray:
    out = image_bgr.copy()
    for h in hits:
        x1, y1, x2, y2 = [int(v) for v in h["bbox"]]
        color = (0, 0, 220) if h.get("wearing_red_hat") else (160, 160, 160)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        conf = h.get("top_prob", 0)
        tag = "RH FEDORA" if h.get("wearing_red_hat") else h.get("top_label", "?")
        text = f"{tag} {conf:.0%}"
        cv2.putText(
            out, text, (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return out


def scan_persons(image_bgr: np.ndarray, detections: list[dict]) -> dict:
    """Score person boxes for red-hat. Returns summary + per-person hits."""
    t0 = time.perf_counter()
    if not _ENABLED:
        return {
            "enabled": False,
            "redhat_hits": 0,
            "should_capture": False,
            "persons": [],
            "ms": 0,
        }

    try:
        _get_model()
    except Exception as e:
        log.warning("redhat scan skipped: %s", e)
        return {
            "enabled": True,
            "error": str(e),
            "redhat_hits": 0,
            "should_capture": False,
            "persons": [],
            "ms": 0,
        }

    persons = [
        d for d in detections
        if (d.get("class_name") or "").lower() in _PERSON_CLASSES
        and float(d.get("confidence") or 0) >= _MIN_PERSON_CONF
    ]
    persons = sorted(persons, key=lambda d: d.get("confidence", 0), reverse=True)[:_MAX_PERSONS]

    hits: list[dict] = []
    for det in persons:
        crop = _hat_region_crop(image_bgr, det.get("bbox") or [0, 0, 0, 0])
        if crop is None or crop.size == 0:
            continue
        try:
            score = _score_crop(crop)
        except Exception as e:
            log.debug("redhat crop failed: %s", e)
            continue
        hits.append({
            "bbox": det.get("bbox"),
            "person_confidence": det.get("confidence"),
            "detector_class": det.get("class_name"),
            **score,
        })

    n_hits = sum(1 for h in hits if h.get("wearing_red_hat"))
    ms = (time.perf_counter() - t0) * 1000
    return {
        "enabled": True,
        "redhat_hits": n_hits,
        "should_capture": n_hits > 0,
        "persons": hits,
        "ms": round(ms, 1),
        "thresholds": {
            "min_person_conf": _MIN_PERSON_CONF,
            "min_clip_conf": _MIN_CLIP_CONF,
            "pos_margin": _MIN_POS_MARGIN,
        },
    }
