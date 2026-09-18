FROM python:3.14-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Copy application code
COPY app/ ./app/

# Build frontend (if using separate frontend build)
# COPY frontend/ ./frontend/
# RUN cd frontend && bun install && bun run build

EXPOSE 8000 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uv", "run", "python", "-m", "app.main"]
