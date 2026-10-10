# Inference Matrix — Deployment

Deployment of the overhauled stack. Architecture:
[ARCHITECTURE.md](ARCHITECTURE.md). Local dev: [development.md](development.md).

## Topology

| Host | Runs | Access |
| --- | --- | --- |
| Application host (`core@10.100.2.100`) | `matrix-app` container (admin: FastAPI + built SPA), postgres, redis, Traefik | `docker` |
| Provider host(s) (`core@10.100.2.111`) | provider **agent** containers (one per machine + provider type + `AGENT_ID`; each hosts 1..N backends), each publishing a single admin-facing `PROVIDER_PORT` | root-scoped `podman` |
| Public | `https://matrix.thelink.family` → `/` Swagger, `/admin` WebUI, `/v1` inference, `/provider/ws` provider dial-in | Traefik TLS |

## Images

- **Admin**: built from the root `Dockerfile` — multi-stage (bun builds the
  frontend → python 3.14 + uv installs `matrix-admin` only,
  `--no-install-workspace`). Image: `ghcr.io/<owner>/matrix-app`.
- **llama-cpp**: two CI-built variants layering the provider package on
  official llama.cpp bases (no compile step; the binary is
  `/app/llama-server`):
  - **Vulkan** (`provider/llama-cpp/Dockerfile`, base
    `ghcr.io/ggml-org/llama.cpp:full-vulkan`) →
    `ghcr.io/<owner>/provider-llama-cpp`. Needs host Vulkan/GPU
    passthrough (`gpus: all` for NVIDIA, or `/dev/dri` + Vulkan ICDs for
    Mesa/AMD/Intel — see the commented compose example).
  - **CUDA 12** (`provider/llama-cpp/Dockerfile.cuda12`, base
    `ghcr.io/ggml-org/llama.cpp:full-cuda12`) →
    `ghcr.io/<owner>/provider-llama-cpp-cuda12`. **Bakes the NVIDIA
    userspace driver** (libcuda/NVML/PTX-JIT + `nvidia-smi`, from RPM Fusion)
    into the image, so it needs **no NVIDIA Container Toolkit** — expose the
    GPU device nodes (`--device /dev/nvidia0 --device /dev/nvidiactl
    --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools`). The baked
    driver version (`NVIDIA_VERSION`, default `580.178.04`) **must match the
    host kernel driver**; bump the build-arg per host. `nvidia-smi` in-image
    feeds the `vram`/`gpu_usage` machine metrics.
- **Other providers** (`gufo`, `halogen`, `halogen-flash`): built from
  `provider/<type>/Dockerfile` out-of-band — they gate on NPU/ROCm
  hardware and their backend binaries are not in CI.
- **talkies** (`provider/talkies/Dockerfile`, Phase 24 speech: TTS + ASR):
  built out-of-band on `ARG BASE_IMAGE=psyb0t/talkies:latest-cuda` (the image
  only layers the provider venv on top — the heavy talkies ML layers are never
  rebuilt). **Requires an NVIDIA GPU** (CUDA base). See the talkies deploy note
  below.

CI (`build-and-push.yml`) builds `matrix-app` and `provider-llama-cpp` on
push to `main`/`develop` and version tags. Tags follow the ref pushed
(`:main`/`:develop`, `pr-<n>`, semver on `v*.*.*` + `major.minor`, and a
short sha) — there is deliberately **no `:latest`** tag.

## Deploy order (IMPORTANT — version hard-fail)

Provider registrations are **rejected with 409** unless the provider
`version` exactly matches the admin `VERSION` (baked from the git commit id
at build time). Therefore:

1. **Deploy the admin first.**
2. Then recreate **every** provider instance with the matching image.

A mismatched provider refuses to start with a clear error rather than
running inconsistent code. This is an accepted constraint of the
solo-operator, fresh-deploy model.

## Application host (admin)

```bash
ssh core@10.100.2.100
cd <matrix-deploy-dir>
docker compose -f compose.yml -f compose.deploy.yml pull
DOMAIN=matrix.thelink.family \
POSTGRES_PASSWORD=<...> \
docker compose -f compose.yml -f compose.deploy.yml up -d
```

- `compose.deploy.yml` adds the Traefik v3 proxy with Let's Encrypt TLS
  and routes `Host(DOMAIN)` to the admin container.
- Container startup runs `scripts/prestart.sh` (alembic migrations) via
  the entrypoint before serving.
- The admin runs a **single uvicorn worker** (the scheduler is in-process;
  see ARCHITECTURE.md §6). Do not scale workers without the Redis-queue
  swap.
- Healthcheck: `GET /admin/api/health`.

Verify:
```bash
curl -fsS https://matrix.thelink.family/admin/api/health
curl -fsS https://matrix.thelink.family/openapi.json >/dev/null && echo "schema ok"
docker logs matrix-app --tail 50
```

## Provider host(s)

Provider **agent** containers run in **root's Podman space** on
`core@10.100.2.111` (quadlet/systemd or `podman run`). Each agent is bound to
one machine + one provider type + a stable `AGENT_ID`, hosts 1..N backends, and
publishes a **single** admin-facing `/v1` port. Each agent needs:

```
MACHINE_UID=<machine uid pre-created in the admin UI>
MACHINE_SECRET=<the machine's registration_secret, shown in the admin UI>
AGENT_ID=<stable id; discriminates multiple agents sharing a machine+type>
ADMIN_BASE_URL=https://matrix.thelink.family
PROVIDER_PORT=8081            # the agent's single admin-facing /v1 port; two
                              # agents on one machine must publish distinct
                              # values (deployment responsibility — the admin no
                              # longer polices or allocates per-backend ports)
CACHE_DIR=/cache              # prompt cache + provider_config.json
MODELS_DIR=/models            # model artifacts
METRICS_CATEGORIES="gpu_usage vram os_ram cpu storage"   # no 'inference'
LLAMA_SERVER_PATH=...         # for llama-cpp; env-only, never in backend_config
```

Example (podman, Vulkan):
```bash
ssh core@10.100.2.111
sudo podman pull ghcr.io/<owner>/provider-llama-cpp:<version>
sudo podman run -d --name provider-llama-cpp \
  --device /dev/dri --device /dev/kfd \
  -e MACHINE_UID=matrix-1 \
  -e MACHINE_SECRET=<machine registration_secret> \
  -e AGENT_ID=matrix-1-llama \
  -e ADMIN_BASE_URL=https://matrix.thelink.family \
  -e PROVIDER_PORT=8081 \
  -e CACHE_DIR=/cache -e MODELS_DIR=/models \
  -v provider_cache:/cache -v provider_models:/models \
  ghcr.io/<owner>/provider-llama-cpp:<version>
```

Example (podman, CUDA 12 — driver baked in, no Container Toolkit; the image
`NVIDIA_VERSION` must match the host kernel driver):
```bash
ssh core@10.100.2.111
sudo podman pull ghcr.io/<owner>/provider-llama-cpp-cuda12:<version>
sudo podman run -d --name provider-llama-cpp-cuda12 \
  --device /dev/nvidia0 --device /dev/nvidiactl \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e MACHINE_UID=matrix-1 \
  -e MACHINE_SECRET=<machine registration_secret> \
  -e AGENT_ID=matrix-1-llama \
  -e ADMIN_BASE_URL=https://matrix.thelink.family \
  -e PROVIDER_PORT=8081 \
  -e CACHE_DIR=/cache -e MODELS_DIR=/models \
  -v provider_cache:/cache -v provider_models:/models \
  ghcr.io/<owner>/provider-llama-cpp-cuda12:<version>
```

- The provider registers at startup, then **dials out** to
  `wss://matrix.thelink.family/provider/ws` — no inbound port to the
  provider host is required for control (only the agent's single
  `PROVIDER_PORT` must be reachable from the admin; litellm dials it for
  every backend of that agent and the agent routes each request to the right
  backend by `model`). Engine (backend) ports are internal to the container and
  OS-assigned by default — the admin never dials them.
- Providers read env at startup only; to change config, recreate the
  container (`podman rm -f` + `run`, or `docker compose up -d
  --force-recreate provider-<type>` where Compose is used).
- The admin assigns machine-level metrics ownership and drives
  `backend.start`/`backend.stop` over the WS as requests arrive.

### talkies (Phase 24 speech: TTS + ASR)

The talkies agent runs a CUDA speech engine; one talkies process boots **per
placed definition** (single-slug engine — the registry is read at import), so
each definition pins its model into VRAM for its whole lifetime.

- **Image**: `docker build -f provider/talkies/Dockerfile -t
  matrix-provider-talkies .` (context = repo root). The base is
  `psyb0t/talkies:latest-cuda` (`ARG BASE_IMAGE`); override only to relocate the
  engine venv (`TALKIES_PYTHON`) or registry file (`TALKIES_MODELS_FILE`).
- **GPU**: expose the NVIDIA device nodes (`/dev/nvidia0`, `/dev/nvidiactl`,
  `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`) or use the Container Toolkit
  (`gpus: all`). The engine defaults to `TALKIES_DEVICE=cuda`.
- **Volumes**: `/models` (prefetched snapshots + staged files + custom voices)
  and `/cache` (`provider_config.json`) **must be writable by uid 1000** (the
  `talkies` user). The Dockerfile pre-creates them owned by uid 1000, but
  **pre-existing volumes from an earlier deploy keep their old ownership** — run
  a one-time `chown -R 1000:1000` on the host volume (or remove it) once.
- **First boot**: the driver prefetches the definition's snapshot into
  `MODELS_DIR/models/<slug>` before spawn (network egress happens only here;
  the server then runs `HF_HUB_OFFLINE=1`), then loads the checkpoint — minutes
  on a cold box. `backend.start` heartbeats `initializing` throughout; budget
  `TALKIES_BOOT_TIMEOUT` (default 600s).
- **Seed slugs** (bundled talkies registry): `qwen3-tts-1.7b-custom`,
  `qwen3-tts-1.7b`, `qwen3-tts-1.7b-design` (TTS) and `whisper-large-v3-turbo`
  (ASR). Ballpark VRAM **per pinned process** (fp16, TTL=0): ~4–6 GiB for the
  1.7B-class Qwen3-TTS slugs, ~2–3 GiB for whisper-large-v3-turbo — size
  `vram_required_bytes` per definition accordingly (there is no
  `x-max-running-backends` cap; VRAM admission governs how many share a box).
- **Version hard-fail**: as with every provider, the talkies `VERSION` must
  exactly match the admin `VERSION` or registration 409s — deploy the admin
  first, then recreate every talkies agent.
- **Smoke status**: CI builds the provider image via the
  `build-provider-talkies` job in `build-and-push.yml` (validates the
  layered build on every push to `main`/`develop`). The `docker run` smoke
  on the GPU host is still pending (no container runtime in the dev
  sandbox). The `provider/talkies/Dockerfile.stubtest` header has the
  exact build+run sequence (stub base → provider image → expect the
  agent, not stock talkies, attempting registration).

## Database

- PostgreSQL 16. Schema applied by the squashed initial migration at
  prestart. Fresh install only — the overhaul does not migrate the old
  schema (backwards compatibility is intentionally not supported).
- Before any upgrade that touches the schema, back up the DB (see the
  backup-restore skill).

## Redis

- Redis 7. Holds WS secrets/epochs/presence, scheduler mirrors, VRAM
  ledger, metrics-ownership leases. All keys are TTL-bounded so a Redis
  flush self-heals from Postgres + fresh provider registrations (config
  itself is never lost — it's in Postgres). Redis must be reachable by the
  admin.

## Security model — trusted LAN

`/admin/api/*` and `/v1/*` are **unauthenticated**. Do not expose them
beyond a trusted network / reverse proxy without adding auth first. The
only secrets are the shared per-machine `registration_secret` (plaintext in
provider env as `MACHINE_SECRET`) and the per-agent WS secret (Redis). Rotating
the machine secret is manual (rotate it in the admin UI, redeploy the machine's
providers). See ARCHITECTURE.md
§12.

## Operational notes

- **Version drift after a partial deploy:** every mismatched provider 409s
  on registration until updated. Check `docker logs` / `podman logs` for
  the version-mismatch error and pull the matching tag.
- **Provider host reachability:** the admin must resolve `machine.host` and
  open `PROVIDER_PORT`. Use a DNS name or IP that works from the admin
  container's network.
- **VRAM over-provisioning:** admission is VRAM-budgeted per machine
  (`Machine.total_vram_bytes` vs `vram_required_bytes`). Idle-backend
  eviction to free VRAM is not yet implemented (Phase 6 TODO) — requests
  wait rather than evict. Size `total_vram_bytes` honestly.
- **Cold starts:** a request for a stopped backend pays the boot time
  inside `scheduler.acquire`; SSE keepalive during boot is a Phase 6 TODO —
  ensure the Traefik/proxy read timeout exceeds your worst-case boot.
