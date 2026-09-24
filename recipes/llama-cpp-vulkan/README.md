# Vulkan llama.cpp Recipe

This recipe builds an Inference Matrix Agent image with llama.cpp compiled with Vulkan GPU acceleration support.

## Overview

The Vulkan recipe provides GPU acceleration for AMD and Intel GPUs through Vulkan API support, while also being compatible with NVIDIA GPUs via Vulkan drivers.

## Building

```bash
docker build -t inference-matrix-agent:llama-cpp-vulkan -f recipes/llama-cpp-vulkan/Dockerfile .
```

## Requirements

To build this recipe, you need:
- Docker with buildx support
- Access to the base agent image (`inference-matrix-agent:base`)
- Vulkan development libraries (automatically installed during build)

## Usage

```bash
docker run --gpus all \
  -e AGENT_ID=agent-1 \
  -e AGENT_PORT=8080 \
  -e FRONTEND_URL=http://host.docker.internal:8000 \
  -v ./models:/models \
  -v ./cache:/cache \
  -p 8080:8080 \
  inference-matrix-agent:llama-cpp-vulkan
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_ID` | Unique agent identifier | *(required)* |
| `AGENT_NAME` | Human-readable name | `inference-agent` |
| `FRONTEND_URL` | Frontend Service URL | *(required)* |
| `AGENT_PORT` | Port for the agent HTTP server | `8080` |
| `DEFAULT_GPU_LAYERS` | Default GPU layers | `35` |
| `GPU_BACKEND` | Set to `vulkan` | `vulkan` |

Set `AGENT_PORT` and publish the same host/container port when using a
non-default port, for example `-e AGENT_PORT=8090 -p 8090:8090`.

## GPU Support

- **AMD**: RDNA2 or newer (RX 6000 series+)
- **Intel**: Arc or newer
- **NVIDIA**: GTX 1000 series or newer (Vulkan 1.2+)

## Development

This recipe is built automatically by the GitHub Actions workflow `build-vulkan-recipe.yml` when changes are made to this directory.
