# Command Center

A centralized web UI for managing and chatting with multiple AI models from a single interface. It integrates with KServe InferenceServices and external OpenAI-compatible endpoints registered as AI asset endpoints. Designed for Red Hat OpenShift AI (tested on v3.5+).

## Features

- **Unified chat interface** for KServe InferenceServices and external OpenAI-compatible endpoints
- **External endpoint support** via a ConfigMap (`gen-ai-aa-custom-model-endpoints`) that defines providers, models, and API key references
- **Image upload** for vision-capable models (sent as base64 `image_url` content parts)
- **In-view vision weights selector**: switch the active detector immediately without
  redeploying the application. `YOLO26n SEG` is the booth default; `YOLO11s CUSTOM`
  is an explicitly labelled evaluation path.
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
