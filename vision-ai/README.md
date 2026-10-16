# Vision AI

Real-time object detection and instance segmentation pipeline for OpenShift AI. Captures an RTSP camera feed or uploaded video, runs inference through YOLO26 models served via KServe, and streams annotated video with bounding boxes and segmentation masks.

## Architecture

```
                                                  gRPC :8001   ┌─────────────────────┐
┌──────────────┐    RTSP     ┌───────────────┐ ──────────────▶ │  YOLO26n (KServe)   │
│  TP-Link     │ ──────────▶ │  visionai-app │                 │  Object Detection   │
│  VIGI S245   │             │  (FastAPI)    │                 │  CPU                │
│  RTSP Camera │             │               │  gRPC :8001     └─────────────────────┘
└──────────────┘             │  OpenCV       │ ──────────────▶ ┌─────────────────────┐
                             │  capture +    │                 │  YOLO26n-seg        │
┌──────────────┐             │  annotation   │                 │  Segmentation Masks │
│  Video       │ ──upload──▶ │               │                 │  CPU                │
│  Upload      │             └───────────────┘                 └─────────────────────┘
└──────────────┘                    │
                                    ▼
                             Annotated MJPEG stream / snapshots / JSON API
```

## Components

### ServingRuntime - ROCm/GPU (`serve.py`)

Custom KServe ServingRuntime with ONNX Runtime + ROCm for AMD GPU inference.

- **Image**: `quay.apps.snuc-openshift.demo.local/visionai/visionai:latest`
- **Runtime**: ONNX Runtime with `ROCMExecutionProvider` (ROCm 6.4)

### ServingRuntime - CPU (`serve.py`)

CPU-only KServe ServingRuntime with ONNX Runtime.

- **Image**: `quay.apps.snuc-openshift.demo.local/visionai/visionai:cpu`
- **Runtime**: ONNX Runtime with `CPUExecutionProvider`

### Vision AI App (`app/`)

FastAPI application that captures RTSP frames and/or uploaded video, runs multi-model inference via KServe gRPC, and streams annotated MJPEG.

- **Base**: Red Hat UBI9
- **Dependencies**: OpenCV, tritonclient[grpc], numpy, FastAPI

## Models

| Model | Purpose | Format |
|-------|---------|--------|
| [YOLO26n](https://github.com/ultralytics/ultralytics) | Object detection (COCO 80 classes) | ONNX |
| [YOLO26n-seg](https://github.com/ultralytics/ultralytics) | Detection + instance segmentation (COCO 80 classes) | ONNX |

Both models run on CPU via the `onnxruntime-cpu` ServingRuntime.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status of camera, inference, and configured models |
| `/models` | GET | List configured model endpoints |
| `/detect` | POST | Upload an image, returns detection JSON |
| `/detect/camera` | GET | Run detection on current RTSP frame (JSON) |
| `/detect/annotate` | POST | Upload an image, returns annotated JPEG with detections and masks |
| `/snapshot` | GET | JPEG snapshot from camera with annotations |
| `/stream` | GET | Live MJPEG stream with detection overlay and segmentation masks |
| `/video/upload` | POST | Upload a video file for playback and inference |
| `/video/stream` | GET | MJPEG stream of uploaded video with detection overlay |
| `/video/status` | GET | Status of uploaded video playback |
| `/video` | DELETE | Stop and remove uploaded video |

Query parameters: `?model=yolo26-seg` to select a specific model, `?annotate=false` to disable overlay.

## Prerequisites

- OpenShift cluster with OpenShift AI (RHOAI) installed
- S3-compatible storage (e.g., OpenShift Data Foundation) with a data connection
- RTSP camera accessible from the cluster network
- Quay registry (in-cluster or external) for container images
- OpenShift GitOps (ArgoCD) operator installed
- OpenShift Pipelines (Tekton) operator installed

## Setup

### 1. Create namespace

```bash
oc new-project sovereign-ai
```

### 2. Create the S3 data connection and service account

Create a data connection in OpenShift AI pointing to your S3 bucket containing the ONNX models. This creates a secret named `models-odf-s3` and a service account that the InferenceServices use to pull model weights.

```bash
oc create sa models-odf-s3-sa -n sovereign-ai
```

### 3. Create the RTSP credentials secret

```bash
oc create secret generic rtsp-credentials \
  --from-literal=RTSP_URL='rtsp://<user>:<password>@<camera-ip>/stream1' \
  -n sovereign-ai
```

### 4. Upload model weights to S3

Place ONNX files in your S3 bucket:

```
/models/yolo26n-seg.onnx       # YOLO26n-seg (primary yolo26 ISVC + optional yolo26-seg)
```

### 5. Deploy with ArgoCD

```bash
oc apply -f argocd/application.yaml
```

ArgoCD watches the `main` branch and auto-syncs the `vision-ai/k8s/` directory, deploying:
- ServingRuntimes (ROCm and CPU)
- InferenceServices (primary `yolo26` = YOLO26n-seg; optional `yolo26-seg`)
- Vision AI app deployment, service, and route
- gRPC bypass services for direct pod access

## Project Structure

```
serve.py                    # KServe model server (shared by both runtimes)
requirements.txt            # Python dependencies for visionai-app
Dockerfile.rocm-base        # visionai:rocm-base  - ROCm + onnxruntime-rocm base layer
Dockerfile                  # visionai:latest     - rocm-base + serve.py (GPU runtime)
Dockerfile.cpu              # visionai:cpu        - UBI9 + onnxruntime (CPU runtime)
app/
  Dockerfile                # App image (UBI9 + OpenCV + FastAPI)
  main.py                   # FastAPI server, RTSP/video streaming, inference threads
  detector.py               # YOLO26 KServe gRPC client, postprocessing, annotation rendering
  segmenter.py              # SAM2 KServe gRPC client (legacy)
  camera.py                 # Threaded RTSP capture
  video.py                  # Video file playback with thread-safe frame buffer
k8s/
  servingruntime-rocm.yaml  # KServe ServingRuntime CR (onnxruntime-migraphx)
  servingruntime-cpu.yaml   # KServe ServingRuntime CR (onnxruntime-cpu)
  inferenceservice.yaml     # Primary yolo26 InferenceService (YOLO26n-seg)
  inferenceservice-seg.yaml # Optional yolo26-seg InferenceService (same weights)
  vision-ai-yolox.yaml      # App Deployment, Service, Route
  vision-ai-yolox-grpc.yaml # gRPC services bypassing kube-rbac-proxy
  vision-ai-yolox-rbac.yaml # RBAC for app SA to access InferenceServices
  pipeline-rbac.yaml        # RBAC for pipeline SA to restart deployments
pipelines/
  pipeline.yaml             # Tekton Pipeline (clone, build, restart-rollouts)
  pipelinerun.yaml          # Manual PipelineRun template
  triggers.yaml             # EventListener, TriggerBinding, TriggerTemplate, Route
  quay-secret.yaml          # Quay push secret placeholder (do NOT commit real creds)
scripts/
  download-models.sh        # Download ONNX models for local testing
argocd/
  application.yaml          # ArgoCD Application for k8s/ manifests
```

## Configuration

The vision-ai app is configured via environment variables in `k8s/vision-ai-yolox.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENDPOINTS` | (required) | Comma-separated `name=host:port` pairs for model endpoints |
| `MODEL_NAME` | `model` | KServe model name used in gRPC inference calls |
| `RTSP_URL` | (from secret) | RTSP camera URL (via `rtsp-credentials` secret) |
| `INPUT_SIZE` | `640` | Detection model input resolution |
| `CONF_THRESHOLD` | `0.25` | Minimum confidence for detections |
| `UNDISTORT_K1` | `0` | Barrel distortion correction coefficient |
| `EXCLUDED_CLASSES` | (see deployment) | Comma-separated COCO class names to filter from detections |

### Tuning `UNDISTORT_K1`

Corrects barrel distortion from wide-angle lenses. Uses OpenCV's distortion model with precomputed remap tables.

| Value | Effect |
|-------|--------|
| `0` | Disabled (no correction) |
| `-0.1` | Light correction |
| `-0.2` | Moderate correction |
| `-0.3` | Good starting point for ~100 FoV dome cameras |
| `-0.5` | Very strong correction |

Adjust live without redeploying:

```bash
oc set env deployment/vision-ai-yolox UNDISTORT_K1="-0.35" -n sovereign-ai
```

## CI/CD

### Tekton Pipeline

The pipeline runs on push to `main` via GitHub webhook:

1. **clone** - Clones the repository
2. **build** - Builds container images and pushes to Quay
3. **restart-rollouts** - Restarts deployments to pick up new images

### ArgoCD

One ArgoCD Application watches the `main` branch:

- `vision-ai-yolox` - syncs `vision-ai/k8s/` manifests (auto-sync, prune, self-heal)
