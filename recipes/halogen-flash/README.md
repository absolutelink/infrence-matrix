# Halogen Flash Recipe

This recipe publishes the Matrix agent with
`ghcr.io/peonist-ai/halogen-flash-server:0.16.2` and runs one Flash server per
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
  --ulimit memlock=-1:-1 \
  -e AGENT_ID=agent-halogen-flash \
  -e FRONTEND_URL=http://host.docker.internal:8000 \
  -v ./models:/models \
  -p 8080:8080 \
  ghcr.io/your-org/agent-halogen-flash:main
```

Weights and the tokenizer are downloaded from
`peonist-ai/halogen-qwen3.8-flash-next` into `/models` on first use.

## NPU small models (optional)

Since 0.16.0 the engine can serve small models on the Ryzen AI NPU beside the
Flash model, on the same port. The Matrix integration exposes five of them as
per-instance virtual aliases. Each is enabled per server instance in the UI
(NPU section of the Flash settings), which sets `HALOGEN_NPU_MODELS` on the
process; the agent pre-downloads the model files into `/models/npu/<id>/`
during prepare so the download progress shows in the UI.

| client-facing name | upstream model | endpoint |
|---|---|---|
| `<alias>-embed` | `qwen3-embedding-0.6b` | `POST /v1/embeddings` |
| `<alias>-rerank` | `qwen3-reranker-0.6b` | `POST /v1/rerank` |
| `<alias>-decide` | `decider-0.8b` | `POST /v1/decisions` |
| `<alias>-guard` | `qwen3guard-gen-0.6b` | `POST /v1/moderations` |
| `<alias>-nano` | `qwen3.5-2b` | `POST /v1/chat/completions` |

`<alias>` is the instance's public alias. The names are scoped per instance, so
two Flash servers never collide.

### Host requirements

The NPU is not just a flag: the host must be prepared, and the engine refuses to
start without it. The agent probes these at registration and the UI only offers
the NPU options on a capable host.

- The `amdxdna` kernel driver and the NPU firmware.
- The host's XRT with its NPU plugin, mounted read-only into the container.
- **The GPU's fabric clock held at its top speed.** GPU work and NPU work at the
  same time on Strix Halo can hang the machine while that clock changes speed.

Add to the run line:

```bash
  --device /dev/accel/accel0 \
  -v /opt/xilinx/xrt:/opt/xilinx/xrt:ro \
  -v /sys:/host/sys \
```

`-v /sys:/host/sys` lets a root-run container hold the fabric clock itself while
it runs. Rootless, install the upstream host unit once instead:

```bash
sudo install -m 755 deploy/host/halogen-fabric-clock /usr/local/sbin/
sudo install -m 644 deploy/host/halogen-fabric-clock.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now halogen-fabric-clock.service
```

`--ulimit memlock=-1:-1` is required for the NPU's locked memory; on Ubuntu the
host's hard limit must also be raised in `/etc/security/limits.conf`.

If the host's XRT lives in the distribution library directory rather than
`/opt/xilinx/xrt`, mount each of the three libraries twice (at
`/opt/xilinx/xrt/lib/` and at its own path) — see the engine's
`docs/NPU.md` for the exact loop.

### Constraints

- NPU small models run only in the engine's default single-container mode, not
  the split `engine`/`api` topology.
- The NPU runs one pass at a time (queue `HALOGEN_NPU_QUEUE`, default 64).
- Inputs are capped at 4,096 tokens for embed/rerank/moderate; decisions take 2
  to 10 options.
- NPU requests consume the Flash instance's inference-lease slots even though
  the NPU work is separate from the GPU work.
- The Flash model runs somewhat slower while the NPU works (shared memory bus
  and power budget).

### Changing the enabled set

`npu_models` is read by the engine at process start. Saving a change to the
enabled set on a running instance stops it, downloads any newly enabled model
files, and restarts with the new `HALOGEN_NPU_MODELS`.
