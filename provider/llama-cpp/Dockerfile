# llama.cpp Inference Matrix provider — Vulkan build.
#
# Layers this provider package on top of the OFFICIAL llama.cpp Vulkan
# image instead of compiling llama.cpp (the proven legacy recipe); the
# llama-server binary ships in the image at /app/llama-server.
#
# Build context = repo root, e.g.:
#   docker build -f provider/llama-cpp/Dockerfile -t matrix-provider-llama-cpp .
#
# RUNTIME (Vulkan): the container needs access to the host's Vulkan
# compute driver. Map the render/compute devices and the driver ICDs, e.g.
# for NVIDIA (Container Toolkit exposes them natively) add
# `gpus: all` (compose) or `--gpus all`; for Mesa/AMD/Intel mount
#   devices: /dev/dri
#   volumes: /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro
# plus the matching driver libraries. The provider instance reports the
# GPUs it can see via machine-metrics (nvidia-smi when present; AMD sysfs
# otherwise) — VRAM admission uses what the backend actually reports.
#
# The model artifacts live OUTSIDE the image: mount MODELS_DIR (e.g.
# /models) and CACHE_DIR (e.g. /cache) as named volumes / host paths.
FROM ghcr.io/ggml-org/llama.cpp:full-vulkan

WORKDIR /app

# Workspace root manifests
COPY pyproject.toml uv.lock /app/

# Provider sources
COPY provider/lib /app/provider/lib
COPY provider/llama-cpp /app/provider/llama-cpp

# The official image ships uv; install it defensively (a slim variant
# without uv falls back to pip) and sync ONLY this provider package +
# provider_lib — never the admin or the other provider packages.
RUN uv sync --frozen --package matrix-provider-llama-cpp \
    || (pip install --no-cache-dir uv \
        && uv sync --frozen --package matrix-provider-llama-cpp)

ENV PATH="/app/.venv/bin:$PATH" \
    PROVIDER_PORT=8081 \
    # The official image's llama-server (NOT on PATH by default).
    LLAMA_SERVER_PATH=/app/llama-server \
    CACHE_DIR=/cache \
    MODELS_DIR=/models

EXPOSE 8081

# Override the base image's ENTRYPOINT (the ggml-org images enter straight
# into llama-server) so our shell command actually runs.
ENTRYPOINT ["sh", "-c", "exec python -m provider_llama_cpp.main"]