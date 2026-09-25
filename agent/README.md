# Inference Matrix Agent

This is the agent service for Inference Matrix, responsible for managing local inference servers and communicating with the main frontend service.

## Features

- Registers with the main Inference Matrix frontend
- Manages llama.cpp or Halogen server instances
- Handles WebSocket communication for command execution
- Downloads and manages GGUF model files
- Reports GPU usage and system information

## Environment Variables

- `AGENT_ID` - Unique identifier for this agent
- `AGENT_NAME` - Human-readable name for the agent (default: "inference-agent")
- `AGENT_PLATFORM` - Inference platform reported to the backend (default: "llamacpp")
- `AGENT_TYPE` - Platform variant reported to the backend (default: "generic")
- `FRONTEND_URL` - URL of the main Inference Matrix frontend
- `LLAMA_SERVER_PATH` - Path to llama-server binary (default: "/usr/local/bin/llama-server")
- `MODELS_PATH` - Path to download and store models (default: "/models")
- `CACHE_PATH` - Path for prompt cache files (default: "/cache")
- `HALOGEN_ENTRYPOINT` - Halogen container entrypoint (default: "/usr/local/bin/entrypoint.sh")
- `HALOGEN_MAX_INSTANCES` - Maximum concurrent Halogen processes (default: 2)

## API Endpoints

### Server Management
- `POST /servers/start` - Start a platform-specific server
- `POST /servers/stop` - Stop a server
- `POST /servers/delete` - Stop a server and remove its cache directory
- `GET /servers/list` - List all running servers
- `GET /servers/status/{server_id}` - Get status of a specific server

When `AGENT_PLATFORM=halogen`, each server runs its own Halogen process with
private API and engine ports. The agent proxies Chat Completions, legacy
Completions, Responses, health, model, cache, and metrics requests while
keeping the Halogen engine port private. The fixed
`peonist-ai/halogen-qwen3.8-27b` repository is downloaded into
`/models/peonist-ai/halogen-qwen3.8-27b`; model selection is not exposed in
the server creation UI.

### Model Management
- `POST /models/download` - Download a model from HuggingFace
- `GET /models/list` - List all downloaded models

### GPU Monitoring
- `GET /gpu/info` - Get GPU information
- `GET /gpu/usage` - Get GPU usage statistics

## WebSocket Events

The agent connects to the frontend over WebSocket to receive commands:
- `start_server` - Start a new server instance
- `stop_server` - Stop a running server instance
- `update_model` - Update a model file

## Usage

```bash
# Run the agent
uv run python -m app.main

# With environment variables
AGENT_ID=agent-123 FRONTEND_URL=http://localhost:8000 uv run python -m app.main
```
