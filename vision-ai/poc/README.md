# Red Hat detect PoC (CLIP, no fine-tune)

Zero-shot path: **YOLO person boxes → CLIP head crop → red fedora / red hat**.

## What’s in tree

| Path | Role |
|------|------|
| `app/redhat.py` | CLIP scorer + `should_capture` |
| `POST /detect/redhat` | Cluster API (JSON; `?annotate=true` → JPEG) |
| `CC /api/vision/detect/redhat` | Auth proxy |
| `poc/poc_redhat_clip.py` | Local smoke (Ultralytics + open_clip) |
| `poc/webcam.html` | Mac webcam → API → auto-capture gallery |

## Local smoke (laptop, no SNUC)

```bash
cd vision-ai/poc
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install ultralytics open-clip-torch opencv-python-headless pillow
mkdir -p samples poc_out   # gitignored — do not commit photos
# drop local JPGs into samples/ (never push these)
python poc_redhat_clip.py samples/
```

## Against vision-ai

```bash
# Use a local file path; do not add sample images to the repo
curl -sS -F "file=@/path/to/local.jpg" \
  "http://VISION:8080/detect/redhat" | jq .
curl -sS -F "file=@/path/to/local.jpg" \
  "http://VISION:8080/detect/redhat?annotate=true" -o out.jpg -D -
```

## Mac webcam

1. Serve or open `webcam.html` (file:// works for camera in Chrome if you allow it; prefer a tiny static server).
2. Point **API base** at vision-ai (`http://…:8080`) **or** Command Center origin with path `/api/vision/detect/redhat` + Bearer token.
3. Start camera → auto-scan → red border captures land in the gallery.

Env knobs on vision-ai: `REDHAT_DETECT`, `REDHAT_MIN_CLIP_CONF`, `REDHAT_POS_MARGIN`, `REDHAT_MIN_PERSON_CONF`.

**Target class:** Red Hat **fedora** only (baseball/trucker caps are negatives in CLIP prompts).

## Command Center (booth)

1. Select a vision unit (`yolo26`).
2. Click **FEDORA** in the vision bar → Mac webcam → auto-scan every ~2.5s.
3. On hit: banner + gallery strip + Shepherd brief.

Requires vision-ai image rebuilt with `app/redhat.py` + CC with the FEDORA button (proxy: `POST /api/vision/detect/redhat`).

## Deploy notes (SNUC)

```bash
# vision-ai (cluster must pick up redhat module)
cd vision-ai && # rebuild/push per your usual ImageStream/pipeline
oc rollout restart deploy/vision-ai-yolox -n sovereign-ai

# command-center static — bump mission tag in k8s/deployment.yaml or rebuild image
# Argo self-heals from Git; edit Git (mission-v2) rather than only the UI
```

**Not production.** Fine-tune YOLO only if you need crowd/tiny-hat reliability beyond CLIP.
