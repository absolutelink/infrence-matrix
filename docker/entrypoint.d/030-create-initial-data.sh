#!/bin/sh
# 030-create-initial-data.sh - Create initial database data

set -e

echo "Creating initial data..."

cd /app/backend

# Create initial data in DB
python app/initial_data.py

echo "Initial data creation complete."
