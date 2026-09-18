FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install uv && uv sync

# Copy application code
COPY app/ ./app/

# Create directories for models and cache
RUN mkdir -p /models /cache

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["uv", "run", "python", "-m", "app.main"]
