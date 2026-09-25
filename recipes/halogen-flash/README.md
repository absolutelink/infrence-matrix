# Halogen Flash Recipe

This recipe publishes the Matrix agent with
`ghcr.io/peonist-ai/halogen-flash-server:0.13.8` and runs one Flash server per
Matrix instance.

It requires AMD Strix Halo (`gfx1151`), `/dev/kfd`, `/dev/dri`, `ipc=host`,
unconfined seccomp, and a large unified-memory host. The Matrix agent allocates
one private API port and one private engine port for each process. Only the API
port is used by the agent proxy; the unauthenticated engine port is never
published.

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host \
  -e AGENT_ID=agent-halogen-flash \
  -e FRONTEND_URL=http://host.docker.internal:8000 \
  -v ./models:/models \
  -p 8080:8080 \
  ghcr.io/your-org/agent-halogen-flash:main
```

Weights and the tokenizer are downloaded from
`peonist-ai/halogen-qwen3.8-flash-next` into `/models` on first use.
