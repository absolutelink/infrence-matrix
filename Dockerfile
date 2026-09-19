FROM oven/bun:1 AS frontend-build

WORKDIR /app

COPY package.json bun.lock /app/
COPY frontend/package.json /app/frontend/

WORKDIR /app/frontend

RUN bun install

COPY ./frontend /app/frontend

ARG VITE_API_URL=

RUN bun run build


FROM python:3.14

ENV PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

# Compile bytecode
ENV UV_COMPILE_BYTECODE=1

# uv Cache
ENV UV_LINK_MODE=copy

WORKDIR /app/

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --package app

COPY ./backend/scripts /app/backend/scripts

COPY ./backend/pyproject.toml ./backend/alembic.ini /app/backend/
COPY ./backend/alembic /app/backend/alembic

COPY ./backend/app /app/backend/app

COPY --from=frontend-build /app/backend/app/frontend /app/backend/app/frontend

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --package app

WORKDIR /app/backend/

# Make prestart script executable
RUN chmod +x scripts/prestart.sh

# Create script to generate runtime config
RUN echo '#!/bin/sh' > /app/backend/generate-config.sh && \
    echo 'if [ -n "$API_URL" ]; then' >> /app/backend/generate-config.sh && \
    echo '  echo "window.APP_CONFIG = { API_URL: '\''$API_URL'\'' }" > /app/backend/app/frontend/config.js' >> /app/backend/generate-config.sh && \
    echo 'fi' >> /app/backend/generate-config.sh && \
    chmod +x /app/backend/generate-config.sh

# Run migrations and start server
CMD ["sh", "-c", "/app/backend/generate-config.sh && scripts/prestart.sh && fastapi run app/main.py --workers 4"]
