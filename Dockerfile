# Stage 1: Build Frontend
FROM oven/bun:1 AS frontend-builder

WORKDIR /app/frontend

# Copy frontend files
COPY frontend/package.json frontend/bun.lock ./
COPY frontend/ .

# Build frontend
RUN bun install --frozen-lockfile
RUN bun run build

# Stage 2: Build Backend with Frontend Assets
FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy backend files
COPY backend/pyproject.toml backend/uv.lock ./
COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./
COPY backend/scripts ./scripts

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./app/frontend

# Install dependencies
RUN uv sync --frozen --no-dev

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Expose port
EXPOSE 8000

# Run prestart and start server
CMD ["sh", "-c", "uv run bash scripts/prestart.sh && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"]
