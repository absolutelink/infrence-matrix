# Inference Matrix — Deployment

Deployment of the overhauled stack. Architecture:
[ARCHITECTURE.md](ARCHITECTURE.md). Local dev: [development.md](development.md).

## Topology

| Host | Runs | Access |
| --- | --- | --- |
| Application host (`core@10.100.2.100`) | `matrix-app` container (admin: FastAPI + built SPA), postgres, redis, Traefik | `docker` |
| Provider host(s) (`core@10.100.2.111`) | provider instance containers (one per backend), each on its own `PROVIDER_PORT` | root-scoped `podman` |
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
    `ghcr.io/ggml-org/llama.cpp:server-cuda12`) →
    `ghcr.io/<owner>/provider-llama-cpp-cuda12`. Needs the NVIDIA
    Container Toolkit; run with `gpus: all`. Machine metrics report via
    in-image `nvidia-smi`.
- **Other providers** (`gufo`, `halogen`, `halogen-flash`): built from
  `provider/<type>/Dockerfile` out-of-band — they gate on NPU/ROCm
  hardware and their backend binaries are not in CI.

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

Provider instances run in **root's Podman space** on `core@10.100.2.111`
(quadlet/systemd or `podman run`). Each instance needs:

```
MACHINE_UID=<machine uid pre-created in the admin UI>
PROVIDER_REGISTRATION_TOKEN=<token of the provider definition>
ADMIN_BASE_URL=https://matrix.thelink.family
PROVIDER_PORT=8081            # unique per machine per instance
CACHE_DIR=/cache              # prompt cache + provider_config.json
MODELS_DIR=/models            # model artifacts
METRICS_CATEGORIES="gpu_usage vram os_ram cpu storage"   # no 'inference'
LLAMA_SERVER_PATH=...         # for llama-cpp; env-only, never in backend_config
```

Example (podman):
```bash
ssh core@10.100.2.111
sudo podman pull ghcr.io/<owner>/provider-llama-cpp:<version>
sudo podman run -d --name provider-llama-cpp \
  --device /dev/dri --device /dev/kfd \
  -e MACHINE_UID=matrix-1 \
  -e PROVIDER_REGISTRATION_TOKEN=<token> \
  -e ADMIN_BASE_URL=https://matrix.thelink.family \
  -e PROVIDER_PORT=8081 \
  -e CACHE_DIR=/cache -e MODELS_DIR=/models \
  -v provider_cache:/cache -v provider_models:/models \
  ghcr.io/<owner>/provider-llama-cpp:<version>
```

- The provider registers at startup, then **dials out** to
  `wss://matrix.thelink.family/provider/ws` — no inbound port to the
  provider host is required for control (only the `PROVIDER_PORT` must be
  reachable from the admin for litellm's HTTP calls).
- Providers read env at startup only; to change config, recreate the
  container (`podman rm -f` + `run`, or `docker compose up -d
  --force-recreate provider-<type>` where Compose is used).
- The admin assigns machine-level metrics ownership and drives
  `backend.start`/`backend.stop` over the WS as requests arrive.

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
only secrets are the per-definition `registration_token` (plaintext in
provider env) and the per-instance WS secret (Redis). Token rotation is
manual (edit the definition, redeploy the provider). See ARCHITECTURE.md
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
