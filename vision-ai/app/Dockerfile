FROM registry.access.redhat.com/ubi9/python-312

USER 0

RUN dnf install -y --nodocs \
        mesa-libGL glib2 \
    && dnf clean all

WORKDIR /opt/app-root/src

COPY requirements.txt .
# CPU torch + CLIP weights baked for island / offline
ENV HF_HOME=/opt/app-root/src/.cache \
    TORCH_HOME=/opt/app-root/src/.cache/torch \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt \
    && python - <<'PY'
import open_clip
open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
print("CLIP weights cached OK")
PY

COPY app/ app/

RUN chown -R 1001:0 /opt/app-root/src && chmod -R g+rwX /opt/app-root/src

USER 1001

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
