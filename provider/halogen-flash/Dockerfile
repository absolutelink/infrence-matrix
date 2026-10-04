# halogen-flash Inference Matrix provider.
#
# Build context = repo root, e.g.:
#   docker build -f provider/halogen-flash/Dockerfile -t matrix-provider-halogen-flash .
#
# NOTE: this image ships the provider package only. The halogen-flash
# server entrypoint must be present in the image (env
# HALOGEN_FLASH_SERVER_PATH, default `halogen-flash-server`). NPU
# small-model support additionally requires the host XRT runtime,
# /dev/accel/accel0, and the pins file at NPU_PINS_FILE (the provider
# degrades gracefully — npu_models is omitted — when the probe fails).
FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Workspace root manifests
COPY pyproject.toml uv.lock /app/

# Provider sources
COPY provider/lib /app/provider/lib
COPY provider/halogen-flash /app/provider/halogen-flash

RUN pip install uv \
    && uv sync --frozen --package matrix-provider-halogen-flash

ENV PATH="/app/.venv/bin:$PATH" \
    PROVIDER_PORT=8081 \
    HALOGEN_FLASH_SERVER_PATH=halogen-flash-server

EXPOSE 8081

CMD ["sh", "-c", "exec python -m provider_halogen_flash.main"]
