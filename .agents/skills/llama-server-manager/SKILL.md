---
name: llama-server-manager
description: Manage llama.cpp llama-server processes - start, stop, configure, and monitor server instances
---

# llama-server Manager Skill

Use this skill when you need to manage llama.cpp llama-server subprocesses for model inference.

## Commands

### Start a Server

Start a llama-server instance for a specific model:

```bash
# Start server with default config
uv run python -m app.services.llama_server start --model-id <model-uuid>

# Start with custom config
uv run python -m app.services.llama_server start \
  --model-id <model-uuid> \
  --gpu-layers 35 \
  --context-size 4096 \
  --batch-size 512
```

**Parameters:**
- `--model-id`: UUID of the model to load
- `--gpu-layers`: Number of layers to offload to GPU (default: 35)
- `--context-size`: Context window size (default: 4096)
- `--batch-size`: Batch size for processing (default: 512)
- `--port`: Specific port (auto-assigned if not specified)

### Stop a Server

Stop a running server instance:

```bash
# Graceful shutdown
uv run python -m app.services.llama_server stop --server-id <server-uuid>

# Force kill
uv run python -m app.services.llama_server stop --server-id <server-uuid> --force
```

### Check Server Status

```bash
# Status of specific server
uv run python -m app.services.llama_server status --server-id <server-uuid>

# List all servers
uv run python -m app.services.llama_server list

# Health check
uv run python -m app.services.llama_server health --server-id <server-uuid>
```

### Restart a Server

```bash
# Restart with same config
uv run python -m app.services.llama_server restart --server-id <server-uuid>

# Restart with new config
uv run python -m app.services.llama_server restart \
  --server-id <server-uuid> \
  --gpu-layers 50
```

### Get Server Logs

```bash
# Recent logs
uv run python -m app.services.llama_server logs --server-id <server-uuid> --lines 50

# Follow logs
uv run python -m app.services.llama_server logs --server-id <server-uuid> --follow
```

### Auto-shutdown Configuration

```bash
# Set inactivity timeout (seconds)
uv run python -m app.services.llama_server config timeout --server-id <server-uuid> --seconds 300

# Disable auto-shutdown
uv run python -m app.services.llama_server config timeout --server-id <server-uuid> --disabled
```

## Python API

Use the `LlamaServerManager` class in Python code:

```python
from app.services.llama_server import LlamaServerManager

manager = LlamaServerManager()

# Start a server
server = await manager.start_server(
    model_id="uuid-here",
    gpu_layers=35,
    context_size=4096
)

# Check status
status = await manager.get_server_status(server.id)

# Stop server
await manager.stop_server(server.id)

# Auto-cleanup old servers
await manager.cleanup_inactive_servers()
```

## When to Use

- **Starting servers**: When a model is requested for inference and no server is running
- **Stopping servers**: When user requests stop, or auto-shutdown on inactivity
- **Monitoring**: Health checks, resource monitoring
- **Configuration**: Adjusting GPU layers, context size per model
- **Troubleshooting**: Checking logs, restarting failed servers

## Best Practices

1. **Always check if server exists** before starting a new one
2. **Use graceful shutdown** unless force is required
3. **Monitor VRAM usage** before starting additional servers
4. **Log all server lifecycle events** for debugging
5. **Auto-restart on failure** with exponential backoff (max 3 retries)
6. **Clean up stale processes** on application startup

## Error Handling

Common errors and solutions:

**"Port already in use"**: Try a different port or kill existing process
**"Out of memory"**: Reduce GPU layers or context size
**"Model file not found"**: Verify model exists and path is correct
**"GPU not detected"**: Check drivers and CUDA/ROCm installation

## Configuration Files

Server configurations are stored in PostgreSQL `server_instances` table.

Default config location: `/etc/inference-matrix/llama-server.yaml`

```yaml
defaults:
  gpu_layers: 35
  context_size: 4096
  batch_size: 512
  inactivity_timeout: 300
  max_instances: 5

gpu:
  backend: auto  # auto, cuda, metal, vulkan
  main_gpu: 0
  tensor_split: null  # For multi-GPU

logging:
  level: info
  format: json
```
