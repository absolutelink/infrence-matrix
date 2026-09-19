#! /usr/bin/env bash

set -e
set -x

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Build database URL from environment variables
DB_URL="postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-app}"

# Run migrations with correct database URL
alembic upgrade head --sqlalchemy.url "$DB_URL"

# Create initial data in DB
python app/initial_data.py
