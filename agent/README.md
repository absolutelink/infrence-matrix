# Inference Matrix Agent

Agent service for distributed inference in the Inference Matrix architecture.

## Features

- Manages llama.cpp server processes
- Handles model downloads from HuggingFace
- Reports GPU information and usage
- WebSocket event streaming to Frontend Service
- HTTP proxy for llama.cpp API

## Setup

### Requirements

- Python 3.14+
- llama.cpp with GPU support (CUDA, Metal, or Vulkan)

### Installation

```bash
uv sync
```

### Configuration

Set the following environment variables:

- `AGENT_ID` - Unique agent identifier (required)
- `AGENT_NAME` - Human-readable agent name (default: "inference-agent")
- `FRONTEND_URL` - Frontend Service URL (required)
- `FRONTEND_API_KEY` - Optional API key for authentication
- `LLAMA_SERVER_PATH` - Path to llama-server binary (default: "/usr/local/bin/llama-server")
- `MODELS_PATH` - Directory for model storage (default: "/models")
- `CACHE_PATH` - Directory for prompt cache (default: "/cache")
- `GPU_BACKEND` - GPU backend: auto, cuda, metal, vulkan (default: "auto")

### Running

```bash
uv run python -m app.main
```

Or with uvicorn:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## API Endpoints

- `GET /health` - Health check
- `GET /servers` - List running servers
- `POST /servers/start` - Start a new server
- `POST /servers/{id}/stop` - Stop a server
- `GET /models` - List downloaded models
- `POST /models/download` - Download a model
- `DELETE /models/{filename}` - Delete a model
- `GET /gpu` - Get GPU information
- `GET /gpu/usage` - Get GPU usage
- `WS /ws/status` - WebSocket event stream

## Docker

Build the image:

```bash
docker build -t inference-matrix-agent .
```

Run with GPU support:

```bash
docker run --gpus all \
  -e AGENT_ID=agent-1 \
  -e FRONTEND_URL=http://frontend:8000 \
  -v ./models:/models \
  -v ./cache:/cache \
  inference-matrix-agent
```
