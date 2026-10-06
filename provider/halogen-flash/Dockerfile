# halogen-flash Inference Matrix provider.
#
# Build context = repo root, e.g.:
#   docker build -f provider/halogen-flash/Dockerfile -t matrix-provider-halogen-flash .
#
# The Halogen image owns the engine, its entrypoint, and its Python runtime.
# The Matrix provider is installed separately with Python 3.14 and is never
# added to Halogen's environment. This mirrors the legacy image layout while
# keeping the new provider protocol and lifecycle.
FROM python:3.14-slim AS provider-build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Workspace root manifests
COPY pyproject.toml uv.lock /app/

# Provider sources
COPY provider/lib /app/provider/lib
COPY provider/halogen-flash /app/provider/halogen-flash

RUN pip install uv \
    && uv sync --frozen --package matrix-provider-halogen-flash

FROM ghcr.io/peonist-ai/halogen-flash-server:0.16.2

USER root

# Keep the server image's Python 3.12 and ROCm environment intact for
# /usr/local/bin/entrypoint.sh. Only copy the provider interpreter/runtime and
# its venv from the Python 3.14 build stage.
COPY --from=provider-build /usr/local/bin/python3.14 /usr/local/bin/python3.14
COPY --from=provider-build /usr/local/lib/python3.14 /usr/local/lib/python3.14
COPY --from=provider-build /usr/local/lib/libpython3.14.so* /usr/local/lib/
COPY --from=provider-build /app/.venv /agent/.venv

RUN mkdir -p /models /cache

ENV PROVIDER_PORT=8081 \
    HALOGEN_FLASH_SERVER_PATH=/usr/local/bin/entrypoint.sh \
    HF_HUB_DISABLE_XET=0

EXPOSE 8081

CMD ["sh", "-c", "exec /agent/.venv/bin/python -m provider_halogen_flash.main"]
