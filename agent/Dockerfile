FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install llama.cpp with CUDA support
RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build -DLLAMA_CUDA=ON \
    && cmake --build build --config Release \
    && cp build/bin/llama-server /usr/local/bin/ \
    && rm -rf /opt/llama.cpp

# Install Python dependencies
COPY pyproject.toml .
RUN pip install uv && uv sync --frozen

# Copy application code
COPY app/ ./app/

# Create directories for models and cache
RUN mkdir -p /models /cache

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["uv", "run", "python", "-m", "app.main"]
