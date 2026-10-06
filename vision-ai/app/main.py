import io
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

MODEL_NAME = os.getenv("MODEL_NAME", "model")
RTSP_URL = os.getenv("RTSP_URL", "rtsp://172.16.199.110/stream1")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
UNDISTORT_K1 = float(os.getenv("UNDISTORT_K1", "0"))
_excluded_env = os.getenv("EXCLUDED_CLASSES", "")
EXCLUDED_CLASSES = {c.strip() for c in _excluded_env.split(",") if c.strip()} if _excluded_env else set()

# Multi-model: "yolo26=host:port,yolo26-seg=host:port"
# Falls back to single INFERENCE_URL if not set
INFERENCE_ENDPOINTS_ENV = os.getenv("INFERENCE_ENDPOINTS", "")
INFERENCE_URL = os.getenv("INFERENCE_URL", "yolo26-grpc.sovereign-ai.svc.cluster.local:8001")

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
    global _video_player, _video_running, _video_threads
    _video_running = False
    for t in _video_threads:
        t.join(timeout=5)
    _video_threads = []
    if _video_player:
        _video_player.stop()
        try:
            os.unlink(_video_player.path)
        except OSError:
            pass
        _video_player = None
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
        _detectors[name] = KServeDetector(
            inference_url=url,
            model_name=MODEL_NAME,
            input_size=(INPUT_SIZE, INPUT_SIZE),
            conf_threshold=CONF_THRESHOLD,
            excluded_classes=EXCLUDED_CLASSES,
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "camera_connected": camera.connected if camera else False,
        "inference_running": _inference_running,
        "models": list(_detectors.keys()),
    }


@app.get("/models")
def list_models():
    return {"models": list(_detectors.keys()), "default": _default_model}


@app.post("/detect")
async def detect_image(file: UploadFile, model: str = None):
    name, detector = _get_detector(model)

    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detections, masks, det_ms = detector.detect(image)

    return {
        "model": name,
        "detections": detections,
        "inference_ms": round(det_ms, 1),
        "image_size": [image.shape[1], image.shape[0]],
        "segmentation": masks is not None,
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

    def generate():
        while _video_player and _video_player.active:
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


@app.delete("/video")
def delete_video():
    _stop_video()
    return {"status": "stopped"}


@app.post("/detect/annotate")
async def detect_annotate(file: UploadFile, model: str = None):
    name, detector = _get_detector(model)
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections, masks, _ = detector.detect(image)
    if detections:
        image = draw_detections(image, detections, masks)
    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")
