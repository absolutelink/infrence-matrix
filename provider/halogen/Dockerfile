# halogen Inference Matrix provider.
#
# Build context = repo root, e.g.:
#   docker build -f provider/halogen/Dockerfile -t matrix-provider-halogen .
#
# NOTE: this image ships the provider package only. The halogen server
# entrypoint must be present in the image (env HALOGEN_SERVER_PATH,
# default `halogen-server`) and requires the ROCm runtime on the host —
# real deployments layer the engine + ROCm stack on top of this image
# (see deployment.md).
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
COPY provider/halogen /app/provider/halogen

RUN pip install uv \
    && uv sync --frozen --package matrix-provider-halogen

ENV PATH="/app/.venv/bin:$PATH" \
    PROVIDER_PORT=8081 \
    HALOGEN_SERVER_PATH=halogen-server

EXPOSE 8081

CMD ["sh", "-c", "exec python -m provider_halogen.main"]
