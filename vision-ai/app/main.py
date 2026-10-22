import io
import json
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.detector import KServeDetector, draw_detections
from app.camera import RTSPCamera
from app.video import VideoPlayer
from app import refine
from app import redhat

MODEL_NAME = os.getenv("MODEL_NAME", "model")
RTSP_URL = os.getenv("RTSP_URL", "rtsp://172.16.199.110/stream1")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
UNDISTORT_K1 = float(os.getenv("UNDISTORT_K1", "0"))
_excluded_env = os.getenv("EXCLUDED_CLASSES", "")
EXCLUDED_CLASSES = {c.strip() for c in _excluded_env.split(",") if c.strip()} if _excluded_env else set()

# Multi-model: "yolo26=host:port,yolo-military=host:port"
# Falls back to single INFERENCE_URL if not set
INFERENCE_ENDPOINTS_ENV = os.getenv("INFERENCE_ENDPOINTS", "")
INFERENCE_URL = os.getenv("INFERENCE_URL", "yolo26-grpc.sovereign-ai.svc.cluster.local:8001")

# Class names: CLASS_NAMES_<MODEL> (e.g. CLASS_NAMES_YOLO_MILITARY) or CLASS_NAMES default.
# Comma-separated. Empty → COCO-80 in the detector.
_DEFAULT_CLASS_NAMES_ENV = os.getenv("CLASS_NAMES", "")


def _class_names_for(model: str) -> list[str] | None:
    key = "CLASS_NAMES_" + model.upper().replace("-", "_")
    raw = os.getenv(key) or _DEFAULT_CLASS_NAMES_ENV
    names = [c.strip() for c in raw.split(",") if c.strip()]
    return names or None

camera: RTSPCamera | None = None
_detectors: dict[str, KServeDetector] = {}
_default_model: str | None = None
_model_detections: dict[str, list[dict]] = {}
_model_masks: dict[str, list[np.ndarray] | None] = {}
_inference_lock = threading.Lock()
_inference_threads: list[threading.Thread] = []
_inference_running = False

_video_player: VideoPlayer | None = None
_video_detections: dict[str, list[dict]] = {}
_video_masks: dict[str, list[np.ndarray] | None] = {}
_video_lock = threading.Lock()
_video_threads: list[threading.Thread] = []
_video_running = False
_video_generation = 0


def _inference_loop(model_name: str, detector: KServeDetector):
    while _inference_running:
        if camera is None or not camera.connected:
            time.sleep(0.5)
            continue

        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        try:
            detections, masks, _ = detector.detect(frame)
            with _inference_lock:
                _model_detections[model_name] = detections
                _model_masks[model_name] = masks
        except Exception:
            pass


def _video_inference_loop(model_name: str, detector: KServeDetector):
    while _video_running:
        if _video_player is None or not _video_player.active:
            time.sleep(0.1)
            continue
        frame = _video_player.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        try:
            detections, masks, _ = detector.detect(frame)
            with _video_lock:
                _video_detections[model_name] = detections
                _video_masks[model_name] = masks
        except Exception:
            pass


def _stop_video():
    global _video_player, _video_running, _video_threads, _video_generation
    # Bump generation first so any open /video/stream generators exit promptly
    _video_generation += 1
    _video_running = False
    player = _video_player
    _video_player = None
    if player:
        player.stop()
        try:
            os.unlink(player.path)
        except OSError:
            pass
    for t in _video_threads:
        t.join(timeout=1.0)
    _video_threads = []
    _video_detections.clear()
    _video_masks.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera, _default_model, _inference_threads, _inference_running

    endpoints = {}
    if INFERENCE_ENDPOINTS_ENV:
        for entry in INFERENCE_ENDPOINTS_ENV.split(","):
            entry = entry.strip()
            if "=" in entry:
                name, url = entry.split("=", 1)
                endpoints[name.strip()] = url.strip()

    if not endpoints:
        endpoints["default"] = INFERENCE_URL

    for name, url in endpoints.items():
        # Military World: skip COCO noise filter (labels won't match)
        names = _class_names_for(name)
        excluded = set() if names else EXCLUDED_CLASSES
        _detectors[name] = KServeDetector(
            inference_url=url,
            model_name=MODEL_NAME,
            input_size=(INPUT_SIZE, INPUT_SIZE),
            conf_threshold=CONF_THRESHOLD,
            excluded_classes=excluded,
            class_names=names,
        )
        _model_detections[name] = []
        _model_masks[name] = None

    _default_model = next(iter(_detectors))

    camera = RTSPCamera(RTSP_URL, undistort_k1=UNDISTORT_K1)
    camera.start()

    _inference_running = True
    for name, detector in _detectors.items():
        t = threading.Thread(target=_inference_loop, args=(name, detector), daemon=True)
        t.start()
        _inference_threads.append(t)

    # Warm CLIP so first booth upload is not a cold download
    try:
        ok = refine.warmup()
        print(f"crop refine: {'ready' if ok else 'disabled/unavailable'}", flush=True)
    except Exception as e:
        print(f"crop refine warmup failed: {e}", flush=True)
    try:
        ok = redhat.warmup()
        print(f"redhat CLIP: {'ready' if ok else 'disabled/unavailable'}", flush=True)
    except Exception as e:
        print(f"redhat CLIP warmup failed: {e}", flush=True)

    yield

    _stop_video()
    _inference_running = False
    for t in _inference_threads:
        t.join(timeout=5)
    camera.stop()


app = FastAPI(title="Vision AI", lifespan=lifespan)


def _get_detector(model: str | None = None) -> tuple[str, KServeDetector]:
    name = model if model and model in _detectors else _default_model
    if name is None or name not in _detectors:
        raise HTTPException(status_code=404, detail="No detector configured")
    return name, _detectors[name]


def _detect_with_tiling(
    detector: KServeDetector,
    image: np.ndarray,
    tiles: int,
) -> tuple[list[dict], list[np.ndarray] | None, float]:
    """
    Optional NxN tiling for ISR-style images where objects are small.

    tiles <= 1 → current behaviour (single full-frame detect()).
    tiles >= 2 → split into tiles×tiles grid, run detect() on each tile,
    map boxes back into full-frame coordinates, and merge results.
    """
    if tiles is None or tiles <= 1:
        return detector.detect(image)

    h, w = image.shape[:2]
    tiles = max(2, min(int(tiles), 4))  # clamp to 2–4 to avoid runaway compute
    tile_w = w // tiles
    tile_h = h // tiles

    all_dets: list[dict] = []
    all_masks: list[np.ndarray] | None = []  # becomes None if no masks at all
    total_ms = 0.0

    for ty in range(tiles):
        for tx in range(tiles):
            x1 = int(tx * tile_w)
            y1 = int(ty * tile_h)
            x2 = w if tx == tiles - 1 else int((tx + 1) * tile_w)
            y2 = h if ty == tiles - 1 else int((ty + 1) * tile_h)

            tile = image[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            dets, masks, det_ms = detector.detect(tile)
            total_ms += det_ms

            for idx, d in enumerate(dets):
                bx1, by1, bx2, by2 = d.get("bbox", [0, 0, 0, 0])
                gx1 = bx1 + x1
                gy1 = by1 + y1
                gx2 = bx2 + x1
                gy2 = by2 + y1
                nd = dict(d)
                nd["bbox"] = [gx1, gy1, gx2, gy2]
                all_dets.append(nd)

                if masks is not None and idx < len(masks):
                    m = masks[idx]
                    full = np.zeros((h, w), dtype=np.uint8)
                    mh, mw = m.shape[:2]
                    fh = min(y2 - y1, mh)
                    fw = min(x2 - x1, mw)
                    full[y1 : y1 + fh, x1 : x1 + fw] = m[:fh, :fw]
                    all_masks.append(full)

    if not all_dets:
        return [], None, total_ms

    if all_masks is not None and len(all_masks) == 0:
        all_masks = None

    return all_dets, all_masks, total_ms


@app.get("/health")
def health():
    return {
        "status": "ok",
        "camera_connected": camera.connected if camera else False,
        "inference_running": _inference_running,
        "models": list(_detectors.keys()),
        "refine_crops": refine.enabled(),
        "redhat_detect": redhat.enabled(),
    }


@app.get("/models")
def list_models():
    return {"models": list(_detectors.keys()), "default": _default_model}


@app.post("/detect")
async def detect_image(file: UploadFile, model: str = None, refine_crops: bool = True, tile: int | None = None):
    name, detector = _get_detector(model)

    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detections, masks, det_ms = _detect_with_tiling(detector, image, tiles=tile)
    if refine_crops and refine.enabled():
        detections = refine.refine_detections(image, detections)

    return {
        "model": name,
        "detections": detections,
        "inference_ms": round(det_ms, 1),
        "image_size": [image.shape[1], image.shape[0]],
        "segmentation": masks is not None,
        "refined": bool(refine_crops and refine.enabled()),
    }


@app.get("/detect/camera")
def detect_camera(model: str = None):
    name, detector = _get_detector(model)
    if camera is None or not camera.connected:
        raise HTTPException(status_code=503, detail="Camera not connected")

    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    detections, masks, det_ms = detector.detect(frame)

    return {
        "model": name,
        "detections": detections,
        "inference_ms": round(det_ms, 1),
        "image_size": [frame.shape[1], frame.shape[0]],
        "segmentation": masks is not None,
    }


@app.get("/snapshot")
def snapshot(annotate: bool = True, model: str = None):
    if camera is None or not camera.connected:
        raise HTTPException(status_code=503, detail="Camera not connected")

    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    if annotate:
        name, _ = _get_detector(model)
        with _inference_lock:
            detections = list(_model_detections.get(name, []))
            masks = _model_masks.get(name)
        if detections:
            frame = draw_detections(frame, detections, masks)

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")


@app.get("/stream")
def mjpeg_stream(annotate: bool = True, model: str = None):
    if camera is None or not camera.connected:
        raise HTTPException(status_code=503, detail="Camera not connected")

    name, _ = _get_detector(model)

    def generate():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            if annotate:
                with _inference_lock:
                    detections = list(_model_detections.get(name, []))
                    masks = _model_masks.get(name)
                if detections:
                    frame = draw_detections(frame, detections, masks)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/video/upload")
async def upload_video(file: UploadFile, model: str = None):
    global _video_player, _video_running, _video_threads
    _stop_video()

    suffix = os.path.splitext(file.filename or "video.mp4")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
    finally:
        tmp.close()

    player = VideoPlayer(tmp.name)
    try:
        player.start()
    except ValueError as e:
        os.unlink(tmp.name)
        raise HTTPException(status_code=400, detail=str(e))

    _video_player = player

    _video_running = True
    for name, detector in _detectors.items():
        t = threading.Thread(target=_video_inference_loop, args=(name, detector), daemon=True)
        t.start()
        _video_threads.append(t)

    return {
        "status": "playing",
        "fps": player.fps,
        "total_frames": player.total_frames,
        "resolution": list(player.resolution),
    }


@app.get("/video/stream")
def video_stream(model: str = None):
    if _video_player is None or not _video_player.active:
        raise HTTPException(status_code=404, detail="No video playing")
    name, _ = _get_detector(model)
    gen = _video_generation

    def generate():
        while (
            gen == _video_generation
            and _video_player is not None
            and _video_player.active
        ):
            frame = _video_player.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            with _video_lock:
                detections = list(_video_detections.get(name, []))
                masks = _video_masks.get(name)
            if detections:
                frame = draw_detections(frame, detections, masks)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video/status")
def video_status():
    if _video_player is None:
        return {"active": False}
    return {
        "active": _video_player.active,
        "fps": _video_player.fps,
        "total_frames": _video_player.total_frames,
        "resolution": list(_video_player.resolution),
    }


@app.get("/detections")
def current_detections(model: str = None):
    """Latest detections for live camera or uploaded video (for Shepherd brief)."""
    name, _ = _get_detector(model)
    with _video_lock:
        video_dets = list(_video_detections.get(name, []))
    dets = video_dets if video_dets else list(_model_detections.get(name, []))
    slim = [
        {
            "class_name": d.get("class_name"),
            "confidence": d.get("confidence"),
            **({"detector_class": d["detector_class"]} if d.get("detector_class") else {}),
        }
        for d in (dets or [])[:20]
    ]
    return {"model": name, "detections": slim, "source": "video" if video_dets else "live"}


@app.delete("/video")
def delete_video():
    _stop_video()
    return {"status": "stopped"}


@app.post("/detect/redhat")
async def detect_redhat(file: UploadFile, model: str = None, annotate: bool = False):
    """PoC: YOLO person boxes → CLIP 'wearing red hat/fedora' → capture flag.

    Returns JSON. Set annotate=true for a JPEG with boxes (X-RedHat-Hits header).
    """
    name, detector = _get_detector(model)
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detections, _, det_ms = detector.detect(image)
    scan = redhat.scan_persons(image, detections)

    payload = {
        "model": name,
        "inference_ms": round(det_ms, 1),
        "image_size": [image.shape[1], image.shape[0]],
        "detections_total": len(detections),
        **scan,
    }

    if not annotate:
        return payload

    drawn = redhat.annotate(image, scan.get("persons") or [])
    _, jpeg = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(
        io.BytesIO(jpeg.tobytes()),
        media_type="image/jpeg",
        headers={
            "X-RedHat-Hits": str(scan.get("redhat_hits", 0)),
            "X-Should-Capture": "1" if scan.get("should_capture") else "0",
            "X-RedHat-Json": json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
            "Access-Control-Expose-Headers": "X-RedHat-Hits, X-Should-Capture, X-RedHat-Json",
        },
    )


@app.post("/detect/annotate")
async def detect_annotate(file: UploadFile, model: str = None, refine_crops: bool = True, tile: int | None = None):
    name, detector = _get_detector(model)
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections, masks, _ = _detect_with_tiling(detector, image, tiles=tile)
    if refine_crops and refine.enabled():
        detections = refine.refine_detections(image, detections)
    if detections:
        image = draw_detections(image, detections, masks)
    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Compact inventory for CC → Shepherd brief (avoid a second detect+CLIP pass)
    slim = [
        {
            "class_name": d.get("class_name"),
            "confidence": d.get("confidence"),
            **({"detector_class": d["detector_class"]} if d.get("detector_class") else {}),
        }
        for d in (detections or [])[:20]
    ]
    det_hdr = json.dumps(slim, separators=(",", ":"), ensure_ascii=True)
    return StreamingResponse(
        io.BytesIO(jpeg.tobytes()),
        media_type="image/jpeg",
        headers={
            "X-Detections": det_hdr,
            "X-Detection-Model": name,
            "Access-Control-Expose-Headers": "X-Detections, X-Detection-Model",
        },
    )
