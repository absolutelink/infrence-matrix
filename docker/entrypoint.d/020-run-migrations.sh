#!/bin/sh
# 020-run-migrations.sh - Run database migrations

set -e

echo "Running database migrations..."

cd /app/admin/backend

# Build database URL from environment variables and export for alembic.
# NOTE (+psycopg): the stack uses psycopg (v3). Without the explicit
# driver segment SQLAlchemy falls back to the psycopg2 dialect and dies
# with ModuleNotFoundError: No module named 'psycopg2'.
export SQLALCHEMY_URL="postgresql+psycopg://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-inference_matrix}"

echo "Using database URL: postgresql+psycopg://${POSTGRES_USER:-postgres}:***@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-inference_matrix}"

# Run migrations with correct database URL
alembic upgrade head

echo "Database migrations complete."
