# syntax=docker/dockerfile:1.7
#
# DefectLens API + Gradio demo (Phase 6b).
#
#   docker build -t defectlens:local .
#   docker run --rm --gpus=all -p 8000:8000 -v "$PWD/models:/app/models:ro" defectlens:local
#
# * Dependencies: only the `serve` and `serve-gpu` groups of uv.lock (FastAPI, Gradio, ONNX Runtime
#   with the CUDA 13 / cuDNN 9 libraries as pip wheels, OpenCV, numpy). No torch, no TensorRT, no
#   CUDA base image: the host driver is injected by the NVIDIA container runtime (--gpus).
# * Weights are not in the image. Mount them read-only at /app/models (the ONNX files that
#   configs/serve/default.yaml names). reports/frozen_thresholds.json is in the image: it is a small
#   text file whose hash the API verifies at start-up.
# * Multi-stage: the builder has uv and the wheel cache, the runtime stage only the finished venv.
# * API only, without the demo: override the command with `defectlens.serve.app:create_app`.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ---- builder: the virtualenv from uv.lock ----------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
# --only-group: just these groups' packages, not the project's own dependencies (anomalib, torch, ...).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group serve --only-group serve-gpu

# ---- runtime ---------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /app/models && chown -R app:app /app
COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEFECTLENS_SERVE_CONFIG=configs/serve/default.yaml \
    GRADIO_ANALYTICS_ENABLED=False \
    CUDA_CACHE_PATH=/tmp/cuda-cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
COPY --chown=app:app src/defectlens ./src/defectlens
COPY --chown=app:app configs/serve ./configs/serve
COPY --chown=app:app reports/frozen_thresholds.json ./reports/frozen_thresholds.json

USER app
EXPOSE 8000
# /health does not load a model; it reports the provider that works (a CUDA probe runs at start-up).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "--factory", "defectlens.serve.demo:create_app_with_demo", "--host", "0.0.0.0", "--port", "8000"]
