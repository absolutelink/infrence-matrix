# Base Inference Matrix Agent image (no llama.cpp included)
FROM python:3.14-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install basic dependencies (no llama.cpp)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install uv && uv sync --frozen

# Copy application code
COPY app/ ./app/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "app.main"]
