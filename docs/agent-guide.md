# Agent Guide

This guide covers the Inference Matrix Agent Service - the distributed inference component that manages llama.cpp servers on GPU-equipped machines.

## Overview

The Agent Service is responsible for:

- Managing llama.cpp server subprocesses
- Handling model downloads from HuggingFace and ModelScope
- Reporting GPU information and usage metrics
- Streaming real-time events to the Frontend Service via WebSocket
- Proxying HTTP requests to llama.cpp servers

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Frontend       │  HTTP   │    Agent        │
│  Service        │◄───────►│    Service      │
│  (Port 8000)    │  WS     │  (Port 8080)    │
└─────────────────┘         └────────┬────────┘
                                     │
                              ┌──────▼───────┐
                              │  llama.cpp   │
                              │   Servers    │
                              └──────────────┘
```

## Quick Start

### Prerequisites

- Python 3.14+
- llama.cpp with GPU support
- NVIDIA GPU with CUDA (or AMD ROCm / Apple Metal)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup:**
```bash
cd agent
uv sync
```

2. **Configure environment:**
```bash
export AGENT_ID=agent-1
export AGENT_NAME=My-GPU-Agent
export FRONTEND_URL=http://localhost:8000
export DEFAULT_GPU_LAYERS=35
```

3. **Start the agent:**
```bash
uv run python -m app.main
```

Or with uvicorn:
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Docker Deployment

```bash
docker build -t inference-matrix-agent ./agent

docker run --gpus all \
  -e AGENT_ID=agent-1 \
  -e FRONTEND_URL=http://host.docker.internal:8000 \
  -v ./models:/models \
  -v ./cache:/cache \
  -p 8080:8080 \
  inference-matrix-agent
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AGENT_ID` | Unique agent identifier | Yes | - |
| `AGENT_NAME` | Human-readable name | No | `inference-agent` |
| `FRONTEND_URL` | Frontend Service URL | Yes | - |
| `FRONTEND_API_KEY` | API key for authentication | No | `None` |
| `LLAMA_SERVER_PATH` | Path to llama-server binary | No | `/usr/local/bin/llama-server` |
| `DEFAULT_GPU_LAYERS` | Default GPU layers for models | No | `35` |
| `DEFAULT_CONTEXT_SIZE` | Default context size | No | `4096` |
| `DEFAULT_BATCH_SIZE` | Default batch size | No | `512` |
| `SERVER_INACTIVITY_TIMEOUT` | Auto-shutdown timeout (seconds) | No | `300` |
| `GPU_BACKEND` | GPU backend: auto, cuda, metal, vulkan | No | `auto` |
| `MODELS_PATH` | Directory for model storage | No | `/models` |
| `CACHE_PATH` | Directory for prompt cache | No | `/cache` |
| `WS_HEARTBEAT_INTERVAL` | WebSocket heartbeat interval (seconds) | No | `30` |
| `WS_RECONNECT_INTERVAL` | Reconnect interval on failure (seconds) | No | `5` |

## API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{"status": "healthy"}
```

### Server Management

#### List Servers
```http
GET /api/servers
```

**Response:**
```json
{
  "servers": [
    {
      "server_id": "uuid",
      "model_path": "/models/llama.gguf",
      "port": 8081,
      "status": "running",
      "uptime_seconds": 123.45
    }
  ]
}
```

#### Start Server
```http
POST /api/servers/start
Content-Type: application/json

{
  "model_id": "model-uuid",
  "model_path": "/models/llama.gguf",
  "config": {
    "gpu_layers": 35,
    "context_size": 4096,
    "batch_size": 512,
    "cache_prompt": true,
    "flash_attn": true
  }
}
```

**Response:**
```json
{
  "server_id": "uuid",
  "status": "running",
  "proxy_url": "http://localhost:8081"
}
```

#### Stop Server
```http
POST /api/servers/{server_id}/stop
```

**Query Parameters:**
- `force` (boolean) - Force kill the process (default: false)

**Response:**
```json
{"status": "stopped"}
```

### Model Management

#### List Models
```http
GET /api/models
```

**Response:**
```json
{
  "models": [
    {
      "filename": "llama-3.2-3b-instruct.Q4_K_M.gguf",
      "path": "/models/llama-3.2-3b-instruct.Q4_K_M.gguf",
      "size_bytes": 2147483648
    }
  ]
}
```

#### Download Model
```http
POST /api/models/download
Content-Type: application/json

{
  "repo_id": "TheBloke/Llama-3.2-3B-Instruct-GGUF",
  "filename": "llama-3.2-3b-instruct.Q4_K_M.gguf",
  "source": "huggingface"
}
```

**Response:**
```json
{
  "status": "completed",
  "path": "/models/llama-3.2-3b-instruct.Q4_K_M.gguf"
}
```

**Status values:**
- `completed` - Download finished successfully
- `already_exists` - File already present
- `downloading` - Download in progress (async)

#### Delete Model
```http
DELETE /api/models/{filename}
```

**Response:**
```json
{"status": "deleted"}
```

### GPU Monitoring

#### Get GPU Info
```http
GET /api/gpu
```

**Response:**
```json
{
  "name": "NVIDIA GeForce RTX 4090",
  "vram_total": 25165824000,
  "vram_used": 8589934592,
  "backend": "cuda"
}
```

#### Get GPU Usage
```http
GET /api/gpu/usage
```

**Response:**
```json
{
  "gpu_percent": 45.2,
  "memory_percent": 34.1,
  "temperature": 65.0
}
```

### WebSocket Events

Connect to the WebSocket endpoint for real-time events:

```http
WS /api/ws/status
Headers:
  X-Agent-ID: agent-1
```

**Event Types:**

- `heartbeat` - Connection keepalive
- `server.started` - Server process started
- `server.stopped` - Server process stopped
- `gpu.usage` - GPU usage update
- `download.progress` - Model download progress

**Example Event:**
```json
{
  "event": "server.started",
  "data": {
    "server_id": "uuid",
    "model_id": "model-uuid",
    "port": 8081
  }
}
```

### Proxy Endpoint

Proxy requests directly to llama.cpp servers:

```http
POST /api/proxy/{server_id}/v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

For streaming requests, set `stream: true` and the response will be Server-Sent Events (SSE).

## Running llama.cpp Servers

### Server Lifecycle

1. **Start:** Frontend or API request triggers server start
2. **Run:** Server handles inference requests
3. **Monitor:** Agent tracks health and usage
4. **Shutdown:** Auto-shutdown after inactivity timeout

### Configuration Options

When starting a server, you can configure:

- `gpu_layers` - Number of layers to offload to GPU (default: 35)
- `context_size` - Context window size (default: 4096)
- `batch_size` - Batch size for processing (default: 512)
- `cache_prompt` - Enable prompt caching (default: true)
- `flash_attn` - Enable flash attention (default: true)

### Prompt Caching

The Agent automatically manages prompt cache files:

- Cache location: `/cache/{server_id}.cache`
- Caches are reused across server restarts
- Cache is model-specific

## Multi-Agent Deployment

### Registering Multiple Agents

Each Agent must have a unique `AGENT_ID`:

```bash
# Agent 1
export AGENT_ID=agent-1
export AGENT_NAME=GPU-Server-1

# Agent 2
export AGENT_ID=agent-2
export AGENT_NAME=GPU-Server-2
```

### Load Distribution

The Frontend Service distributes requests across Agents based on:

1. Model availability
2. Agent status (online/offline)
3. Current load (future feature)

### Network Requirements

- Agents must be able to reach Frontend Service (HTTP + WebSocket)
- Frontend must be able to reach Agents (HTTP)
- Recommended: Private network between services

## Troubleshooting

### Agent Won't Start

**Check environment variables:**
```bash
echo $AGENT_ID
echo $FRONTEND_URL
```

**Verify llama-server binary:**
```bash
/usr/local/bin/llama-server --version
```

### Server Won't Start

**Check logs:**
```bash
docker logs <agent-container>
```

**Common issues:**
- Model file not found
- GPU memory insufficient
- Port already in use
- Invalid configuration

**Verify model exists:**
```bash
ls -la /models/
```

### WebSocket Disconnects

**Check network connectivity:**
```bash
curl http://frontend:8000/health
```

**Verify FRONTEND_URL is correct:**
```bash
echo $FRONTEND_URL
```

**Check Frontend logs for reconnection attempts**

### GPU Not Detected

**Verify NVIDIA drivers:**
```bash
nvidia-smi
```

**Check Docker GPU configuration:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Try manual llama-server start:**
```bash
/usr/local/bin/llama-server --model /models/test.gguf --n-gpu-layers 35
```

### High Memory Usage

**Reduce GPU layers:**
```bash
DEFAULT_GPU_LAYERS=20  # Reduce from 35
```

**Reduce context size:**
```bash
DEFAULT_CONTEXT_SIZE=2048  # Reduce from 4096
```

**Check VRAM usage:**
```http
GET /api/gpu/usage
```

## Performance Tuning

### Optimal GPU Layers

Find the sweet spot for your GPU:

1. Start with `DEFAULT_GPU_LAYERS=35`
2. Monitor VRAM usage
3. Increase if VRAM available
4. Decrease if out of memory

### Batch Size Tuning

- Larger batch = better throughput, more memory
- Smaller batch = lower latency, less memory
- Default 512 is good for most cases

### Context Size

- Larger context = more tokens, more memory
- Consider 2048 for chat, 4096+ for long documents
- Memory scales linearly with context size

## Monitoring

### Health Checks

```bash
curl http://localhost:8080/health
```

### Metrics

Agent exposes metrics via the Frontend Service:

- Agent status (online/offline)
- GPU utilization
- Server count
- WebSocket connection status

### Logs

```bash
# Docker
docker logs -f <agent-container>

# Direct
journalctl -u inference-matrix-agent -f
```

## Security

### API Key Authentication

Enable authentication with Frontend:

```bash
export FRONTEND_API_KEY=your_secret_key
```

### Network Isolation

- Run Agents on private network
- Use firewall rules to restrict access
- Enable TLS for Frontend-Agent communication

### Resource Limits

Prevent resource exhaustion:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
    reservations:
      memory: 8G
```

## Development

### Running Tests

```bash
cd agent
uv run pytest tests/ -v
```

### Code Structure

```
agent/
├── app/
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   └── config.py        # Configuration
│   ├── api/
│   │   └── routes/          # API endpoints
│   ├── services/
│   │   ├── llama_server.py  # Server management
│   │   ├── proxy.py         # HTTP proxy
│   │   └── frontend_client.py # Frontend connection
│   └── utils/
└── tests/
```

### Adding New Features

1. Add endpoint in `app/api/routes/`
2. Implement service logic in `app/services/`
3. Write tests in `tests/`
4. Update this documentation

## Upgrading

```bash
# Pull latest changes
git pull

# Rebuild image
docker compose build agent

# Restart
docker compose up -d agent
```

**Note:** Model files and cache are preserved across upgrades.
