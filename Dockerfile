FROM oven/bun:1 AS frontend-build

WORKDIR /app

COPY package.json bun.lock /app/
COPY admin/frontend/package.json /app/admin/frontend/

WORKDIR /app/admin/frontend

RUN bun install

COPY ./admin/frontend /app/admin/frontend

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
    uv sync --frozen --no-install-workspace --package matrix-admin

COPY ./admin/backend/scripts /app/admin/backend/scripts

COPY ./admin/backend/pyproject.toml ./admin/backend/alembic.ini /app/admin/backend/
COPY ./admin/backend/alembic /app/admin/backend/alembic

COPY ./admin/backend/app /app/admin/backend/app

COPY --from=frontend-build /app/admin/backend/app/frontend /app/admin/backend/app/frontend

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --package matrix-admin

WORKDIR /app/admin/backend/

# Install entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install entrypoint scripts
COPY docker/entrypoint.d/ /etc/entrypoint.d/
RUN chmod +x /etc/entrypoint.d/*.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Run FastAPI server. Single worker: the scheduler keeps in-process state
# (WebSocket registry, request queues) backed by Redis for cross-restart
# reconciliation. Do not raise worker count without completing the Redis
# scheduler interface.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-ping-interval", "30", "--ws-ping-timeout", "60"]
