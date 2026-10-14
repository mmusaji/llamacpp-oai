#!/usr/bin/env python3
"""Local Red Hat CLIP PoC — no cluster required.

Uses Ultralytics YOLO (person boxes) + open_clip (red hat / fedora).
Shares prompt logic with app/redhat.py (copied lightly so this runs standalone).

Usage:
  python3 -m venv .venv && source .venv/bin/activate
  pip install ultralytics open-clip-torch opencv-python-headless pillow
  python poc_redhat_clip.py samples/red_fedora.jpg
  python poc_redhat_clip.py samples/          # all images in dir
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow importing sibling package when run from vision-ai/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.redhat import annotate, scan_persons, warmup  # noqa: E402


def yolo_persons(image_bgr: np.ndarray, weights: str, conf: float) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(image_bgr, conf=conf, verbose=False)
    dets: list[dict] = []
    if not results:
        return dets
    r0 = results[0]
    names = r0.names or {}
    if r0.boxes is None:
        return dets
    for box in r0.boxes:
        cid = int(box.cls.item())
        name = names.get(cid, str(cid))
        if name != "person":
            continue
        xyxy = box.xyxy[0].tolist()
        dets.append({
            "class_id": cid,
            "class_name": name,
            "confidence": round(float(box.conf.item()), 4),
            "bbox": [round(float(v), 1) for v in xyxy],
        })
    return dets


def run_one(path: Path, weights: str, conf: float, out_dir: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"cannot read {path}")
    dets = yolo_persons(image, weights, conf)
    scan = scan_persons(image, dets)
    result = {
        "file": str(path),
        "persons_yolo": len(dets),
        **scan,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = annotate(image, scan.get("persons") or [])
    out_path = out_dir / f"{path.stem}_redhat.jpg"
    cv2.imwrite(str(out_path), drawn)
    result["annotated"] = str(out_path)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Local red-hat CLIP PoC")
    ap.add_argument("input", help="image file or directory")
    ap.add_argument("--weights", default="yolov8n.pt", help="Ultralytics weights")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO person conf")
    ap.add_argument("--out", default="poc_out", help="annotated output dir")
    args = ap.parse_args()

    if not warmup():
        print("CLIP warmup failed — is open-clip-torch installed?", file=sys.stderr)
        return 1

    src = Path(args.input)
    paths: list[Path]
    if src.is_dir():
        paths = sorted(
            p for p in src.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
    else:
        paths = [src]

    if not paths:
        print("no images found", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    for p in paths:
        r = run_one(p, args.weights, args.conf, out_dir)
        flag = "CAPTURE" if r.get("should_capture") else "skip"
        print(
            f"[{flag}] {p.name}: persons={r['persons_yolo']} "
            f"hits={r.get('redhat_hits')} clip_ms={r.get('ms')}"
        )
        for person in r.get("persons") or []:
            print(
                f"    {person.get('top_label')} "
                f"p={person.get('top_prob')} pos={person.get('pos_mass')} "
                f"wear={person.get('wearing_red_hat')}"
            )
        print(json.dumps({k: r[k] for k in ("should_capture", "redhat_hits", "thresholds")}, indent=2))
    print(f"annotated → {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
