# gufo Inference Matrix provider.
#
# Build context = repo root, e.g.:
#   docker build -f provider/gufo/Dockerfile -t matrix-provider-gufo .
#
# NOTE: this image ships the provider package only. The `gufo` engine
# binary must be present in the image (env GUFO_SERVER_PATH, default
# `gufo` on PATH) — real deployments layer it on top of this image or
# extend the base (see deployment.md: "Real-engine images layer the
# backend binary onto the provider package").
FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Workspace root manifests
COPY pyproject.toml uv.lock /app/

# Provider sources
COPY provider/lib /app/provider/lib
COPY provider/gufo /app/provider/gufo

RUN pip install uv \
    && uv sync --frozen --package matrix-provider-gufo

ENV PATH="/app/.venv/bin:$PATH" \
    PROVIDER_PORT=8081 \
    GUFO_SERVER_PATH=gufo

EXPOSE 8081

CMD ["sh", "-c", "exec python -m provider_gufo.main"]
