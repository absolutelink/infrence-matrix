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

The selected checkpoint and tokenizer are configured through the server
settings and must be available under `/models`. The agent creates an isolated
`/cache/<server_uuid>` directory for every server and removes it when that
server is deleted.
