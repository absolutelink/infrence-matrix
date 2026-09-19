#!/bin/sh
# 020-run-migrations.sh - Run database migrations

set -e

echo "Running database migrations..."

cd /app/backend

# Build database URL from environment variables and export for alembic
export SQLALCHEMY_URL="postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-app}"

echo "Using database URL: postgresql://${POSTGRES_USER:-postgres}:***@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-app}"

# Run migrations with correct database URL
alembic upgrade head

echo "Database migrations complete."
