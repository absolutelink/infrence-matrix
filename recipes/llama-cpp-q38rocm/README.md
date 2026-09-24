# q38rocm llama.cpp Recipe

This recipe layers the Inference Matrix agent on
`ghcr.io/julianmb/q38rocm:latest`, providing the ROCmFP4 engine for AMD Strix
Halo systems.

The q38rocm `agent` launcher profile is a reference for useful llama.cpp
settings. It is not applied automatically by this recipe. Configure those
settings explicitly in the server settings UI when needed.

## Requirements

- AMD Strix Halo / gfx1151 hardware
- Linux amd64
- Access to `/dev/kfd` and `/dev/dri`
- Host access for the `video` and `render` device groups
- A running Inference Matrix backend

## Build

```bash
docker build \
  --build-arg AGENT_IMAGE=ghcr.io/your-org/agent:main \
  -t inference-matrix-agent:q38rocm \
  -f recipes/llama-cpp-q38rocm/Dockerfile .
```

## Run

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -e AGENT_ID=agent-q38rocm \
  -e AGENT_PORT=8080 \
  -e FRONTEND_URL=http://host.docker.internal:8000 \
  -v ./models:/models \
  -v ./cache:/cache \
  -p 8080:8080 \
  inference-matrix-agent:q38rocm
```

Set `AGENT_PORT` and publish the same host/container port when using a
non-default port, for example `-e AGENT_PORT=8090 -p 8090:8090`.

Models can be downloaded through the UI. The agent stores them under
`/models`; no model needs to be baked into the image.
