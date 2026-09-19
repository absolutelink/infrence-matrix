#! /usr/bin/env bash

set -e
set -x

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Run migrations
alembic upgrade head

# Create initial data in DB
python app/initial_data.py
