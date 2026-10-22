import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640  # must match your ONNX export

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def preprocess_bgr(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize to INPUT_SIZE, NCHW, [0,1] — mirrors detector.preprocess()."""
    orig_h, orig_w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)
    return blob, (orig_w, orig_h)


def postprocess(outputs: dict, orig_size: tuple[int, int]) -> list[dict]:
    """Approximate detector.postprocess(): boxes only, no masks."""
    orig_w, orig_h = orig_size
    inp_w = inp_h = INPUT_SIZE
    out = outputs["output0"]
    while out.ndim > 2 and out.shape[0] == 1:
        out = out.squeeze(0)

    dets: list[dict] = []

    # Case 1: end-to-end layout (N, 6+) = xyxy, score, class[, mask coeffs]
    if out.ndim == 2 and 6 <= out.shape[-1] <= 6 + 64:
        scores = out[:, 4]
        class_ids = out[:, 5].astype(int)
        boxes = out[:, :4]

        mask = scores > CONF_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0] * orig_w / inp_w
        y1 = boxes[:, 1] * orig_h / inp_h
        x2 = boxes[:, 2] * orig_w / inp_w
        y2 = boxes[:, 3] * orig_h / inp_h
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        for i in range(len(boxes_xyxy)):
            cid = int(class_ids[i])
            name = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else str(cid)
            dets.append({
                "class_id": cid,
                "class_name": name,
                "confidence": float(scores[i]),
                "bbox": [float(v) for v in boxes_xyxy[i]],
            })
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        return dets

    # Case 2: classic YOLO ONNX: (4+nc, N) or (N, 4+nc)
    pred = out.squeeze()
    if pred.ndim != 2:
        return []

    if pred.shape[0] < pred.shape[1]:
        boxes_xywh = pred[:4].T
        cls_scores = pred[4:].T
    else:
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]

    if cls_scores.size == 0:
        return []

    scores = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1).astype(int)
    conf_mask = scores > CONF_THRESHOLD
    boxes_xywh = boxes_xywh[conf_mask]
    scores = scores[conf_mask]
    class_ids = class_ids[conf_mask]

    if len(boxes_xywh) == 0:
        return []

    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = (cx - bw / 2) * orig_w / inp_w
    y1 = (cy - bh / 2) * orig_h / inp_h
    x2 = (cx + bw / 2) * orig_w / inp_w
    y2 = (cy + bh / 2) * orig_h / inp_h
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    idxs = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        scores.tolist(),
        CONF_THRESHOLD,
        IOU_THRESHOLD,
    )
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).reshape(-1)

    for i in idxs:
        cid = int(class_ids[i])
        name = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else str(cid)
        dets.append({
            "class_id": cid,
            "class_name": name,
            "confidence": float(scores[i]),
            "bbox": [float(v) for v in boxes_xyxy[i]],
        })
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to YOLO ONNX model")
    parser.add_argument("--image", required=True, help="Path to test image (e.g. convoy jpeg)")
    args = parser.parse_args()

    print(f"Loading ONNX model: {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print("Inputs:")
    for i in inputs:
        print(f"  {i.name}: shape={i.shape}, type={i.type}")
    print("Outputs:")
    for o in outputs:
        print(f"  {o.name}: shape={o.shape}, type={o.type}")

    input_name = inputs[0].name

    img = Image.open(args.image).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    blob, orig_size = preprocess_bgr(img_bgr)

    feeds = {input_name: blob}
    t0 = time.perf_counter()
    ort_outs = sess.run(None, feeds)
    dt = (time.perf_counter() - t0) * 1000.0

    out_dict = {"output0": ort_outs[0]}
    if len(ort_outs) > 1:
        out_dict["output1"] = ort_outs[1]

    dets = postprocess(out_dict, orig_size)
    print(f"Inference time: {dt:.1f} ms  detections: {len(dets)}")
    for d in dets[:10]:
        bbox_str = ",".join(f"{v:.1f}" for v in d["bbox"])
        print(f"  {d['class_name']:>10s}  conf={d['confidence']:.3f}  bbox={bbox_str}")


if __name__ == "__main__":
    main()

