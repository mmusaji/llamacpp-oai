import os
import io
import json
import re
import datetime
import ssl
import base64

import yaml
import jwt
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kubernetes import client, config

app = FastAPI()

PASSWORD = os.getenv("COMMAND_PASSWORD", "commander")
JWT_SECRET = os.getenv("JWT_SECRET", os.urandom(32).hex())
NAMESPACE = os.getenv("NAMESPACE", "sovereign-ai")
ENDPOINTS_CM = os.getenv("ENDPOINTS_CONFIGMAP", "gen-ai-aa-custom-model-endpoints")
VISION_AI_URL = os.getenv("VISION_AI_URL", "http://vision-ai.sovereign-ai.svc.cluster.local:8080")

try:
    config.load_incluster_config()
except Exception:
    config.load_kube_config()

core_api = client.CoreV1Api()
custom_api = client.CustomObjectsApi()


class LoginRequest(BaseModel):
    password: str


def verify_token(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        jwt.decode(auth[7:], JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/api/auth/login")
def login(req: LoginRequest):
    if req.password != PASSWORD:
        raise HTTPException(status_code=401, detail="Access denied")
    token = jwt.encode(
        {"sub": "commander", "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)},
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"token": token}


def _load_external_endpoints() -> list[dict]:
    try:
        cm = core_api.read_namespaced_config_map(ENDPOINTS_CM, NAMESPACE)
    except client.exceptions.ApiException:
        return []
    raw = cm.data.get("config.yaml", "")
    if not raw:
        return []
    cfg = yaml.safe_load(raw)
    providers = {p["provider_id"]: p for p in cfg.get("providers", {}).get("inference", [])}
    endpoints = []
    for model in cfg.get("registered_resources", {}).get("models", []):
        provider = providers.get(model.get("provider_id"))
        if not provider:
            continue
        base_url = provider.get("config", {}).get("base_url", "").rstrip("/")
        meta = model.get("metadata", {})
        display = meta.get("display_name", model.get("model_id", ""))
        secret_ref = provider.get("config", {}).get("custom_gen_ai", {}).get("api_key", {}).get("secretRef", {})
        api_key = None
        if secret_ref.get("name"):
            try:
                secret = core_api.read_namespaced_secret(secret_ref["name"], NAMESPACE)
                key_field = secret_ref.get("key", "api_key")
                encoded = secret.data.get(key_field, "")
                api_key = base64.b64decode(encoded).decode() if encoded else None
            except client.exceptions.ApiException:
                pass
        endpoints.append({
            "name": model.get("provider_id"),
            "display": display,
            "description": model.get("model_id", ""),
            "ready": True,
            "type": "external",
            "base_url": base_url,
            "model_id": model.get("model_id", ""),
            "api_key": api_key,
            "capabilities": meta.get("capabilities", []),
        })
    return endpoints


_external_cache: list[dict] = []
_external_cache_time: float = 0


def _get_external_endpoints() -> list[dict]:
    global _external_cache, _external_cache_time
    import time
    now = time.time()
    if now - _external_cache_time > 60:
        _external_cache = _load_external_endpoints()
        _external_cache_time = now
    return _external_cache


@app.get("/api/models", dependencies=[Depends(verify_token)])
def list_models():
    isvcs = custom_api.list_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=NAMESPACE,
        plural="inferenceservices",
    )
    models = []
    for isvc in isvcs.get("items", []):
        name = isvc["metadata"]["name"]
        annotations = isvc.get("metadata", {}).get("annotations", {})
        display = annotations.get("openshift.io/display-name", name)
        description = annotations.get("openshift.io/description", "")
        conditions = isvc.get("status", {}).get("conditions", [])
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
        url = isvc.get("status", {}).get("address", {}).get("url", "")
        model_type = annotations.get("opendatahub.io/model-type", "")
        models.append({"name": name, "display": display, "description": description, "ready": ready, "type": "kserve", "url": url, "capabilities": [], "model_type": model_type})
    for ep in _get_external_endpoints():
        models.append({"name": ep["name"], "display": ep["display"], "description": ep["description"], "ready": ep["ready"], "type": "external", "capabilities": ep.get("capabilities", [])})
    return {"models": models}


def _get_model_url(model_name: str) -> str:
    isvc = custom_api.get_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=NAMESPACE,
        plural="inferenceservices",
        name=model_name,
    )
    url = isvc.get("status", {}).get("address", {}).get("url", "")
    if not url:
        raise HTTPException(status_code=502, detail=f"Model {model_name} has no endpoint")
    return url


def _get_sa_token() -> str | None:
    path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


@app.post("/api/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions(request: Request):
    body = await request.json()
    model_endpoint = body.pop("model_endpoint", None)
    if not model_endpoint:
        raise HTTPException(status_code=400, detail="model_endpoint required")

    ext = next((ep for ep in _get_external_endpoints() if ep["name"] == model_endpoint), None)
    if ext:
        chat_url = ext["base_url"].rstrip("/") + "/chat/completions"
        body["model"] = ext["model_id"]
        headers = {"Content-Type": "application/json"}
        if ext.get("api_key"):
            headers["Authorization"] = f"Bearer {ext['api_key']}"
    else:
        chat_url = _get_model_url(model_endpoint) + "/v1/chat/completions"
        body.setdefault("model", "default")
        headers = {"Content-Type": "application/json"}
        sa_token = _get_sa_token()
        if sa_token:
            headers["Authorization"] = f"Bearer {sa_token}"

    stream = body.get("stream", False)

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    timeout = httpx.Timeout(connect=60.0, read=300.0, write=30.0, pool=30.0)

    if stream:
        async def generate():
            try:
                async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
                    async with c.stream("POST", chat_url, json=body, headers=headers) as resp:
                        if resp.status_code != 200:
                            error_text = (await resp.aread()).decode("utf-8", errors="replace")
                            yield f"data: {json.dumps({'error': {'message': error_text, 'code': resp.status_code}})}\n\ndata: [DONE]\n\n"
                            return
                        async for chunk in resp.aiter_text():
                            yield chunk
            except httpx.TimeoutException:
                yield f"data: {json.dumps({'error': {'message': 'upstream timeout', 'code': 504}})}\n\ndata: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': str(e), 'code': 502}})}\n\ndata: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
        resp = await c.post(chat_url, json=body, headers=headers)
        return resp.json()


@app.post("/api/audio/transcribe", dependencies=[Depends(verify_token)])
async def transcribe_audio(file: UploadFile, model_endpoint: str = Form(...)):
    ext = next((ep for ep in _get_external_endpoints() if ep["name"] == model_endpoint), None)
    if not ext:
        raise HTTPException(status_code=400, detail="Unknown endpoint")

    transcribe_url = ext["base_url"].rstrip("/") + "/audio/transcriptions"
    headers = {}
    if ext.get("api_key"):
        headers["Authorization"] = f"Bearer {ext['api_key']}"

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout = httpx.Timeout(connect=60.0, read=300.0, write=30.0, pool=30.0)

    audio_bytes = await file.read()
    async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
        resp = await c.post(
            transcribe_url,
            files={"file": (file.filename, audio_bytes, file.content_type)},
            data={"model": "whisper-large-v3"},
            headers=headers,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


def verify_token_or_query(request: Request):
    auth = request.headers.get("Authorization", "")
    query_token = request.query_params.get("token")
    tok = None
    if auth.startswith("Bearer "):
        tok = auth[7:]
    elif query_token:
        tok = query_token
    if not tok:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        jwt.decode(tok, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/api/vision/stream", dependencies=[Depends(verify_token_or_query)])
async def vision_stream(model: str = None):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)

    upstream = f"{VISION_AI_URL}/stream"
    if model:
        upstream += f"?model={model}"

    async def proxy():
        try:
            async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
                async with c.stream("GET", upstream) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except (httpx.RemoteProtocolError, httpx.ReadError, GeneratorExit):
            return

    return StreamingResponse(proxy(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/vision/health", dependencies=[Depends(verify_token)])
async def vision_health():
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    try:
        async with httpx.AsyncClient(verify=ssl_ctx, timeout=5.0) as c:
            resp = await c.get(f"{VISION_AI_URL}/health")
            return resp.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/vision/video", dependencies=[Depends(verify_token)])
async def vision_video_upload(request: Request, model: str = None):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout = httpx.Timeout(connect=60.0, read=300.0, write=300.0, pool=30.0)

    upstream = f"{VISION_AI_URL}/video/upload"
    if model:
        upstream += f"?model={model}"

    body = await request.body()
    content_type = request.headers.get("content-type", "")

    async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
        resp = await c.post(
            upstream,
            content=body,
            headers={"Content-Type": content_type},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


@app.get("/api/vision/video/stream", dependencies=[Depends(verify_token_or_query)])
async def vision_video_stream(model: str = None):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)

    upstream = f"{VISION_AI_URL}/video/stream"
    if model:
        upstream += f"?model={model}"

    async def proxy():
        try:
            async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
                async with c.stream("GET", upstream) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except (httpx.RemoteProtocolError, httpx.ReadError, GeneratorExit):
            return

    return StreamingResponse(proxy(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.delete("/api/vision/video", dependencies=[Depends(verify_token)])
async def vision_video_delete():
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    try:
        async with httpx.AsyncClient(verify=ssl_ctx, timeout=5.0) as c:
            resp = await c.delete(f"{VISION_AI_URL}/video")
            return resp.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/vision/detections", dependencies=[Depends(verify_token)])
async def vision_detections(model: str = None):
    """Latest live/video detections from vision-ai (for Shepherd brief on feeds)."""
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    upstream = f"{VISION_AI_URL}/detections"
    if model:
        upstream += f"?model={model}"
    async with httpx.AsyncClient(verify=ssl_ctx, timeout=15.0) as c:
        resp = await c.get(upstream)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


@app.post("/api/vision/detect/annotate", dependencies=[Depends(verify_token)])
async def vision_detect_annotate(request: Request, model: str = None):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout = httpx.Timeout(connect=60.0, read=300.0, write=30.0, pool=30.0)

    upstream = f"{VISION_AI_URL}/detect/annotate"
    if model:
        upstream += f"?model={model}"

    body = await request.body()
    content_type = request.headers.get("content-type", "")

    async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
        resp = await c.post(
            upstream,
            content=body,
            headers={"Content-Type": content_type},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        headers = {
            "Access-Control-Expose-Headers": "X-Detections, X-Detection-Model",
        }
        if resp.headers.get("x-detections"):
            headers["X-Detections"] = resp.headers["x-detections"]
        if resp.headers.get("x-detection-model"):
            headers["X-Detection-Model"] = resp.headers["x-detection-model"]
        return StreamingResponse(
            io.BytesIO(resp.content),
            media_type="image/jpeg",
            headers=headers,
        )


@app.post("/api/vision/detect", dependencies=[Depends(verify_token)])
async def vision_detect(request: Request, model: str = None):
    """JSON detections (with CLIP crop refine on vision-ai)."""
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout = httpx.Timeout(connect=60.0, read=300.0, write=30.0, pool=30.0)

    upstream = f"{VISION_AI_URL}/detect"
    if model:
        upstream += f"?model={model}"

    body = await request.body()
    content_type = request.headers.get("content-type", "")

    async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
        resp = await c.post(
            upstream,
            content=body,
            headers={"Content-Type": content_type},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


@app.post("/api/vision/brief", dependencies=[Depends(verify_token)])
async def vision_brief(request: Request):
    """Tactical text brief from refined detections via Shepherd (or configured model)."""
    body = await request.json()
    detections = body.get("detections") or []
    model_endpoint = body.get("model_endpoint") or os.getenv("BRIEF_MODEL", "shepherd-alpha")

    if not detections:
        return {"brief": "No detections to brief.", "model": model_endpoint}

    lines = []
    for d in detections[:12]:
        name = d.get("class_name", "?")
        conf = d.get("confidence", 0)
        was = d.get("detector_class")
        if was and was != name:
            lines.append(f"- {name} ({conf:.0%}; detector said {was})")
        else:
            lines.append(f"- {name} ({conf:.0%})")
    inventory = "\n".join(lines)

    prompt = (
        "You are a military intelligence specialist. From this sensor inventory "
        "(object detector + visual reclassification of each box), give exactly 4 lines "
        "in this plain format (no markdown, no bold, no numbering):\n"
        "Present: ...\n"
        "Threat: ...\n"
        "Risk: ...\n"
        "Action: ...\n"
        "Be concise. Do not invent objects not listed.\n\nInventory:\n" + inventory
    )

    chat_body = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a military intelligence specialist in doctrine."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 280,
        "stream": False,
    }

    ext = next((ep for ep in _get_external_endpoints() if ep["name"] == model_endpoint), None)
    if ext:
        chat_url = ext["url"].rstrip("/") + "/v1/chat/completions"
        chat_body["model"] = ext.get("model_id", "default")
        headers = {"Content-Type": "application/json"}
        if ext.get("token"):
            headers["Authorization"] = f"Bearer {ext['token']}"
    else:
        chat_url = _get_model_url(model_endpoint) + "/v1/chat/completions"
        sa = _get_sa_token()
        headers = {"Content-Type": "application/json"}
        if sa:
            headers["Authorization"] = f"Bearer {sa}"

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)

    async with httpx.AsyncClient(verify=ssl_ctx, timeout=timeout) as c:
        resp = await c.post(chat_url, json=chat_body, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        # Shepherd sometimes emits chain-of-thought markers
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
        if "</think>" in text.lower():
            text = re.split(r"</think>", text, flags=re.I)[-1].strip()
        return {"brief": text or "(empty brief)", "model": model_endpoint, "inventory": inventory}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
