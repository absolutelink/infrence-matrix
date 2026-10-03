#!/usr/bin/env bash
# Run database migrations for the admin backend.
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export SQLALCHEMY_URL="postgresql+psycopg://${POSTGRES_USER:-inference}:${POSTGRES_PASSWORD:-}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-inference_matrix}"

uv run alembic upgrade head
