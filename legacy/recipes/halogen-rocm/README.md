# Halogen ROCm Recipe

This recipe publishes the Inference Matrix agent as `agent-halogen-rocm` and
runs one Halogen process per Matrix server instance.

## Requirements

- AMD Strix Halo / `gfx1151`
- Linux amd64
- `/dev/kfd` and `/dev/dri`
- `video` and `render` device groups
- `seccomp=unconfined`
- `ipc=host`
- Recommended 128 GB unified memory

## Build

```bash
docker build \
  --build-arg AGENT_IMAGE=ghcr.io/your-org/agent:main \
  -t agent-halogen-rocm \
  -f recipes/halogen-rocm/Dockerfile .
```

## Run

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e AGENT_ID=agent-halogen-rocm \
  -e FRONTEND_URL=http://host.docker.internal:8000 \
  -v ./models:/models \
  -v ./cache:/cache \
  -p 8080:8080 \
  agent-halogen-rocm
```

The Matrix agent allocates private Halogen API and engine ports for each
server. Only the Matrix agent port is published. The unauthenticated Halogen
engine port is never exposed to the host.

The agent downloads the complete `peonist-ai/halogen-qwen3.8-27b` repository
on first use into `/models/peonist-ai/halogen-qwen3.8-27b` and starts Halogen
with the checkpoint and tokenizer paths from that repository. The agent
requires outbound Hugging Face access for the initial download; subsequent
starts reuse the files under `/models`.
