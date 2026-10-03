# Mock Inference Matrix provider (no hardware required)
FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Workspace root manifests
COPY pyproject.toml uv.lock /app/

# Provider sources
COPY provider/lib /app/provider/lib
COPY provider/mock /app/provider/mock

RUN pip install uv \
    && uv sync --frozen --package matrix-provider-mock

ENV PATH="/app/.venv/bin:$PATH" \
    PROVIDER_PORT=8081

EXPOSE 8081

CMD ["sh", "-c", "exec python -m provider_mock.main"]
