# Base Inference Matrix Agent image (no llama.cpp included)
ARG BASE_IMAGE=python:3.14-slim

FROM $BASE_IMAGE

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install basic dependencies (no llama.cpp)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
COPY uv.lock .
RUN pip install uv && uv sync

# Copy application code
COPY app/ ./app/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Use uvicorn to run the FastAPI app directly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
