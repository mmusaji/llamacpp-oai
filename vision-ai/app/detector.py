import time

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


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


class KServeDetector:
    def __init__(self, inference_url: str, model_name: str,
                 input_size: tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.25,
                 excluded_classes: set[str] | None = None,
                 token: str | None = None,
                 class_names: list[str] | None = None,
                 iou_threshold: float = 0.45):
        self.inference_url = inference_url.rstrip("/")
        self.model_name = model_name
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.excluded_classes = excluded_classes or set()
        self.class_names = class_names if class_names else COCO_CLASSES
        self.token = token
        self._client = grpcclient.InferenceServerClient(url=inference_url)
        self._has_seg = None

    def _label(self, cid: int) -> str:
        if 0 <= cid < len(self.class_names):
            return self.class_names[cid]
        return str(cid)

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, (orig_w, orig_h)

    def _pack_detections(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        seg_masks: list[np.ndarray] | None = None,
    ) -> tuple[list[dict], list[np.ndarray] | None]:
        detections = []
        keep_indices = []
        for i in range(len(boxes_xyxy)):
            cid = int(class_ids[i])
            name = self._label(cid)
            if self.excluded_classes and name in self.excluded_classes:
                continue
            keep_indices.append(i)
            detections.append({
                "class_id": cid,
                "class_name": name,
                "confidence": round(float(scores[i]), 4),
                "bbox": [round(float(v), 1) for v in boxes_xyxy[i]],
            })
        if seg_masks is not None:
            seg_masks = [seg_masks[i] for i in keep_indices]
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, seg_masks

    def _postprocess_e2e(
        self, output: np.ndarray, protos, orig_size: tuple[int, int]
    ) -> tuple[list[dict], list[np.ndarray] | None]:
        """Ultralytics end-to-end layout: (N, 6+) = xyxy, score, class[, mask coeffs]."""
        orig_w, orig_h = orig_size
        inp_w, inp_h = self.input_size
        has_seg = protos is not None and output.ndim >= 2 and output.shape[-1] > 6

        scores = output[:, 4]
        class_ids = output[:, 5].astype(int)
        boxes = output[:, :4]
        mask_coeffs = output[:, 6:] if has_seg else None

        conf_mask = scores > self.conf_threshold
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        class_ids = class_ids[conf_mask]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[conf_mask]

        if len(boxes) == 0:
            return [], None

        x1 = boxes[:, 0] * orig_w / inp_w
        y1 = boxes[:, 1] * orig_h / inp_h
        x2 = boxes[:, 2] * orig_w / inp_w
        y2 = boxes[:, 3] * orig_h / inp_h
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        seg_masks = None
        if has_seg and mask_coeffs is not None:
            protos = protos.squeeze(0)
            raw_masks = mask_coeffs @ protos.reshape(protos.shape[0], -1)
            raw_masks = 1.0 / (1.0 + np.exp(-raw_masks))
            raw_masks = raw_masks.reshape(-1, protos.shape[1], protos.shape[2])

            seg_masks = []
            for i in range(len(raw_masks)):
                m = cv2.resize(raw_masks[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                bx1, by1, bx2, by2 = [int(v) for v in boxes_xyxy[i]]
                cropped = np.zeros_like(m, dtype=np.uint8)
                cropped[by1:by2, bx1:bx2] = (m[by1:by2, bx1:bx2] > 0.5).astype(np.uint8)
                seg_masks.append(cropped)

        return self._pack_detections(boxes_xyxy, scores, class_ids, seg_masks)

    def _postprocess_raw(
        self, output: np.ndarray, orig_size: tuple[int, int]
    ) -> tuple[list[dict], list[np.ndarray] | None]:
        """Classic YOLO ONNX: (4+nc, N) or (N, 4+nc) — xywh + class scores, no NMS."""
        orig_w, orig_h = orig_size
        inp_w, inp_h = self.input_size

        pred = output.squeeze() if output.ndim == 3 else output
        if pred.ndim != 2:
            return [], None
        # Prefer (C, N) when one dim looks like channels (4+nc)
        if pred.shape[0] < pred.shape[1]:
            # (4+nc, N)
            boxes_xywh = pred[:4].T
            cls_scores = pred[4:].T
        else:
            # (N, 4+nc)
            boxes_xywh = pred[:, :4]
            cls_scores = pred[:, 4:]

        if cls_scores.size == 0:
            return [], None

        scores = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1).astype(int)
        conf_mask = scores > self.conf_threshold
        boxes_xywh = boxes_xywh[conf_mask]
        scores = scores[conf_mask]
        class_ids = class_ids[conf_mask]

        if len(boxes_xywh) == 0:
            return [], None

        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - bw / 2) * orig_w / inp_w
        y1 = (cy - bh / 2) * orig_h / inp_h
        x2 = (cx + bw / 2) * orig_w / inp_w
        y2 = (cy + bh / 2) * orig_h / inp_h
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS (OpenCV expects list of boxes / scores)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(indices) == 0:
            return [], None
        flat = np.array(indices).reshape(-1)
        return self._pack_detections(
            boxes_xyxy[flat], scores[flat], class_ids[flat], None
        )

    def postprocess(self, outputs: dict, orig_size: tuple[int, int]) -> tuple[list[dict], list[np.ndarray] | None]:
        output = outputs["output0"]
        while output.ndim > 2 and output.shape[0] == 1:
            output = output.squeeze(0)
        protos = outputs.get("output1")

        # E2E / NMS-baked: last dim is 6 (xyxy, score, cls) or >6 with mask coeffs
        if output.ndim == 2 and output.shape[-1] >= 6 and output.shape[-1] <= 6 + 32:
            # Heuristic: e2e rows are few (≤300); raw is usually (4+nc, 8400)
            if output.shape[0] <= 300 or output.shape[-1] == 6 or (
                output.shape[-1] > 6 and output.shape[0] <= 300
            ):
                return self._postprocess_e2e(output, protos, orig_size)

        # Raw YOLO-World / detect export: (4+nc, N) with nc typically small for military
        return self._postprocess_raw(output, orig_size)

    def detect(self, image: np.ndarray) -> tuple[list[dict], list[np.ndarray] | None, float]:
        blob, orig_size = self.preprocess(image)

        inputs = [grpcclient.InferInput("images", list(blob.shape), "FP32")]
        inputs[0].set_data_from_numpy(blob)

        outputs = [grpcclient.InferRequestedOutput("output0")]
        if self._has_seg is not False:
            outputs.append(grpcclient.InferRequestedOutput("output1"))

        t0 = time.perf_counter()
        try:
            result = self._client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
            )
        except Exception:
            if self._has_seg is None:
                outputs = [grpcclient.InferRequestedOutput("output0")]
                result = self._client.infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs,
                )
                self._has_seg = False
            else:
                raise
        inference_ms = (time.perf_counter() - t0) * 1000

        output_dict = {"output0": result.as_numpy("output0")}
        if self._has_seg is not False:
            try:
                o1 = result.as_numpy("output1")
                if o1 is not None:
                    output_dict["output1"] = o1
                    self._has_seg = True
                else:
                    self._has_seg = False
            except Exception:
                self._has_seg = False


        detections, masks = self.postprocess(output_dict, orig_size)
        return detections, masks, inference_ms

    def is_ready(self) -> bool:
        try:
            return self._client.is_model_ready(self.model_name)
        except Exception:
            return False


PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
]


def draw_detections(image: np.ndarray, detections: list[dict],
                    masks: list[np.ndarray] | None = None) -> np.ndarray:
    annotated = image.copy()
    overlay = annotated.copy()
    # Scale stroke/font with image size so boxes stay readable in Command Center
    h, w = annotated.shape[:2]
    thick = max(3, int(round(min(h, w) / 200)))
    font_scale = max(0.6, min(h, w) / 800)
    font_thick = max(2, thick - 1)

    for i, det in enumerate(detections):
        color = PALETTE[i % len(PALETTE)]

        if masks and i < len(masks):
            overlay[masks[i] > 0] = color

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        y_text = max(th + 8, y1)
        cv2.rectangle(annotated, (x1, y_text - th - 8), (x1 + tw + 4, y_text), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y_text - 6),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA,
        )

    if masks:
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

    return annotated
