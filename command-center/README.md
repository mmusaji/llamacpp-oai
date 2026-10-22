# Command Center

A centralized web UI for managing and chatting with multiple AI models from a single interface. It integrates with KServe InferenceServices and external OpenAI-compatible endpoints registered as AI asset endpoints. Designed for Red Hat OpenShift AI (tested on v3.5+).

## Features

- **Unified chat interface** for KServe InferenceServices and external OpenAI-compatible endpoints
- **External endpoint support** via a ConfigMap (`gen-ai-aa-custom-model-endpoints`) that defines providers, models, and API key references
- **Image upload** for vision-capable models (sent as base64 `image_url` content parts)
- **Audio upload** for transcription-capable models (transcribed via Whisper, transcript included in chat)
- **Streaming responses** with real-time stats (TTFT, tokens/sec, prompt tokens)
- **Markdown rendering** in responses (bold, italic, headers, lists, code blocks)
- **Auto-refresh** model status every 30 seconds

## Files

```
main.py               # FastAPI backend (auth, model proxy, transcription)
Dockerfile            # Container image for Command Center
requirements.txt      # Python dependencies
static/               # Frontend (HTML, CSS, JS)
k8s/deployment.yaml   # OpenShift manifests (RBAC, Deployment, Service, Route)
pipelines/            # Tekton build pipeline
```

## Mission packages and offline switching

For the SNUC NATO demo, Command Center is also the **mission switch UI** for the edge cluster.

- **Mission = image tag** on the `command-center` ImageStream in `sovereign-ai`:
  - Baseline UI: `command-center:mission-v1` (no FEDORA controls).
  - FEDORA mission: `command-center:mission-v2` (adds CAM / UPLOAD + fedora brief path).
- The backend exposes:
  - `GET /api/missions` (auth required) → lists the **current** mission tag (derived from the Deployment image) and **available** mission tags (from the ImageStream, filtered to `mission-*` tags that already have imported images).
  - `POST /api/missions/switch` with `{"tag": "<mission-tag>"}` → **patches the `command-center` Deployment** to use that tag for the `web` container image.
- The frontend adds a **`SWITCH MISSION`** button in the top bar:
  - Shows the current mission (for example `MISSION: V1` or `MISSION: V2`).
  - Opens a modal with a dropdown of only the tags returned by `GET /api/missions`.
  - On apply, calls `POST /api/missions/switch` and shows a short status while the pod restarts.

This pattern is **offline-friendly** on the SNUC:

- As long as `mission-v1`, `mission-v2`, etc. are **pre-pulled into the internal registry** and present on the ImageStream, the field operator can swap missions locally with no OpenShift console and no WAN.
- In a connected demo, ACM + Argo CD can still drive the Git-level revision flip (`main` ↔ `mission-v2`); Command Center then becomes the **on-cluster override** for switching between the already-synced mission tags when the hub is not reachable.

## Deploying

### 1. Build and push the image

```bash
cd command-center
podman build -t quay.io/<your-user>/command-center:latest .
podman push quay.io/<your-user>/command-center:latest
```

### 2. Deploy to OpenShift

Update the `image` and `namespace` in [k8s/deployment.yaml](k8s/deployment.yaml), then apply:

```bash
oc apply -f k8s/deployment.yaml
```

This creates a ServiceAccount, RBAC Role (read access to InferenceServices, ConfigMaps, and Secrets), Deployment, Service, and Route.

### 3. Add external endpoints (optional)

Create a ConfigMap to register external OpenAI-compatible providers:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gen-ai-aa-custom-model-endpoints
data:
  config.yaml: |
    providers:
      inference:
      - provider_id: my-provider
        provider_type: remote::openai
        config:
          base_url: https://api.example.com/openai/v1
          custom_gen_ai:
            api_key:
              secretRef:
                name: my-api-key-secret
                key: api_key
    registered_resources:
      models:
      - provider_id: my-provider
        model_id: model-name
        model_type: llm
        metadata:
          display_name: My Model
          capabilities:
          - text-generation
          - vision              # enables image upload
          - audio-transcription # enables audio upload
```

Create the corresponding Secret for the API key:

```bash
oc create secret generic my-api-key-secret --from-literal=api_key=<YOUR_API_KEY>
```

## CI/CD

The Tekton pipeline in [pipelines/](pipelines/) builds the image from `./command-center` and restarts the deployment on completion.
