#!/bin/bash
# Debug script to run on the inference-matrix server (10.100.2.100)
# This will manually test the container and migrations

echo "=== Stopping inference-matrix service ==="
sudo systemctl stop inference-matrix

echo "=== Removing old container ==="
docker rm -f inference-matrix 2>/dev/null || true

echo "=== Checking image contents ==="
echo "Running container with interactive shell to check alembic folder..."
docker run --rm --entrypoint "" ghcr.io/absolutelink/matrix-app:main ls -la /app/backend/

echo "=== Checking if alembic folder exists ==="
docker run --rm --entrypoint "" ghcr.io/absolutelink/matrix-app:main ls -la /app/backend/alembic 2>&1 || echo "ALEMBIC FOLDER NOT FOUND!"

echo "=== Testing migration manually ==="
docker run --rm --entrypoint "" ghcr.io/absolutelink/matrix-app:main bash -c "cd /app/backend && alembic upgrade head" 2>&1

echo "=== Starting service again ==="
sudo systemctl start inference-matrix

echo "=== Checking service status ==="
sudo systemctl status inference-matrix --no-pager

echo "=== Viewing recent logs ==="
sudo journalctl -u inference-matrix.service -n 50 --no-pager
