# Inference Matrix Agent Service

The Agent Service manages llama.cpp servers on inference hardware. It communicates with the Frontend Service to receive commands and report status.

## Features

- **llama.cpp Server Management** - Start/stop inference servers
- **GPU Monitoring** - Real-time VRAM and utilization tracking
- **Model Management** - Download, delete, and validate GGUF models
- **HTTP Proxy** - Proxies llama.cpp API requests from Frontend
- **WebSocket Events** - Real-time status updates to Frontend
- **Auto-Registration** - Registers with Frontend on startup

## Quick Start

### 1. Configure Environment

Create `.env` file:

```bash
AGENT_ID=agent-1
AGENT_NAME=GPU-Agent-1
FRONTEND_URL=http://frontend:8000
LLAMA_SERVER_PATH=/usr/local/bin/llama-server
DEFAULT_GPU_LAYERS=35
MODELS_PATH=/models
CACHE_PATH=/cache
```

### 2. Install Dependencies

```bash
cd agent
uv sync
```

### 3. Run Agent

```bash
uv run python -m app.main
```

### 4. Verify Registration

Check Frontend logs or WebUI to confirm Agent is registered.

## API Endpoints

- `POST /api/servers/start` - Start llama.cpp server
- `POST /api/servers/{id}/stop` - Stop server
- `GET /api/servers` - List running servers
- `GET /api/gpu/info` - GPU information
- `GET /api/models` - List model files
- `POST /api/models/download` - Download model
- `DELETE /api/models/{id}` - Delete model
- `GET /api/health` - Health check
- `WS /api/ws/status` - WebSocket event stream

## Directory Structure

```
agent/
├── app/
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   ├── config.py        # Settings
│   │   └── logging.py       # Logging config
│   ├── api/
│   │   ├── routes/
│   │   │   ├── servers.py   # Server management
│   │   │   ├── models.py    # Model files
│   │   │   ├── gpu.py       # GPU monitoring
│   │   │   └── websocket.py # WebSocket events
│   │   └── deps.py          # Dependencies
│   ├── services/
│   │   ├── llama_server.py  # llama.cpp management
│   │   ├── model_manager.py # Model downloads
│   │   ├── gpu_monitor.py   # GPU monitoring
│   │   ├── proxy.py         # HTTP proxy
│   │   └── frontend_client.py # Frontend connection
│   └── utils/
│       └── llama_cpp.py     # llama.cpp helpers
└── tests/
```

## Docker Deployment

```bash
docker compose up agent
```

See `compose.yml` for full configuration.
