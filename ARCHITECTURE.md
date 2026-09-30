# Inference Matrix — Architecture

Inference Matrix is an OpenAI-compatible inference broker that schedules GGUF model
inference across one or more hardware-local agents. The system is a monorepo with
four major boundaries: `backend/`, `frontend/`, `agent/`, and `recipes/`.

The high-level data path is:

```
Client (OpenAI SDK / WebUI)
        │  HTTPS /v1/*                 (OpenAI-compatible API)
        ▼
┌─────────────────────────────┐   WebSocket /api/ws/*    ┌──────────────────────────┐
│  Backend (FastAPI broker)   │ ◄──────────────────────► │  Agent (FastAPI, per host)│
│  - stateless request layer  │   POST /api/v1/agents/   │  - owns llama-server procs │
│  - scheduler + leases (PG)  │       register           │  - proxies llama.cpp HTTP  │
│  - metadata (PostgreSQL)    │                          │  - GPU / model file mgmt   │
└─────────────┬───────────────┘                          └────────────┬───────────────┘
              │                                                       │ subprocess spawn
              ▼                                                       ▼
        PostgreSQL 16                                   llama.cpp / Halogen engines + GPU
```

The backend never talks to llama.cpp directly. It never holds model files. It is a
**stateless broker**: all durable state lives in PostgreSQL; all hardware state
lives in the agent's process table.

---

## 1. Repository Layout

| Path | Role |
| --- | --- |
| `backend/` | FastAPI broker: OpenAI-compatible API, scheduler, agent manager, migrations |
| `frontend/` | React + TypeScript + TanStack Router WebUI (served by the backend image) |
| `agent/` | Hardware-local agent: llama-server / Halogen lifecycle + inference proxy |
| `recipes/` | Docker recipes layering the agent onto different GPU backends (Vulkan, ROCm, Halogen) |
| `docker/` | Container entrypoint + `entrypoint.d` migration/config scripts |
| `scripts/` | Client generation (`generate-client.sh`), test, release helpers |
| `docs/` | Supporting docs (architecture prose, deployment guides) |

Tooling: Python 3.14 + `uv` for backend/agent; Bun + Vite for the frontend.
Backend and agent are separate `uv` projects with their own `pyproject.toml` and
`uv.lock`.

---

## 2. Backend (the broker)

Python 3.14, FastAPI, SQLModel, PostgreSQL (async via `psycopg` + async engine).
Entry: `backend/app/main.py`.

### 2.1 Application assembly (`app/main.py`)

The app mounts three groups of routers:

1. **Management API** under `settings.API_V1_STR` (`/api/v1`) — `api_router` from
   `app/api/main.py`: agents, server instances, models, benchmarks, queue
   (`POST /queue/clear`, `GET /queue/{request_id}`), metrics, stats, huggingface
   search, utils. The queue lookup exposes status and terminal reason, not prompts.
2. **Agent/UI WebSocket** under `/api` (not `/api/v1`):
   - `/api/ws/agents/{agent_id}` — inbound **command** channel: the agent dials
     this socket and receives commands (`start_server`, `stop_server`,
     `update_model`) from the backend. (Events flow the other way — see §3.2;
     the buffer replayed here at connect is the backend's own per-agent buffer,
     not agent event replay.)
   - `/api/ws/events/{agent_id}` — UI subscription to a single agent's event stream
   - `/api/ws/queue-status` — aggregate queue/slot snapshot pushed every 2s
3. **OpenAI-compatible API** under `/v1` (kept off `/api/v1` so SDK base URLs work):
   `models`, `chat/completions`, `completions`, `embeddings`, `responses`
   (+ `responses` WebSocket), `files`, `batches`, `audio/*`.

Lifespan hooks (`lifespan` in `main.py`) start background loops:

- `agent_manager.start_cleanup_loop()` — prune offline agents.
- `inference_scheduler.reconcile_persisted_leases()` — invalidate leases left over
  from a previous process, then `start_reconciliation()`.
- `start_queue_worker()` — benchmark queue worker.
- `token_stats_prune_loop()` — prune token usage samples.
- `metrics_snapshot_loop()` — maintain Prometheus-style metrics.

An HTTP middleware (`track_inference_activity`) counts `/v1/*` requests until the
SSE body is fully drained, not merely until the handler returns. This activity
counter is separate from inference lease ownership: a streaming lease is released
at upstream completion, even if the downstream client drains more slowly (§4.4).

The frontend is served from `app.frontend("/", directory=FRONTEND_DIR)`
(`backend/app/frontend`, populated at image build time).

### 2.2 Configuration (`app/core/config.py`)

`Settings` (pydantic-settings) reads `../.env`. Key groups:

- Database: `POSTGRES_*`, `DATABASE_URL` (async `postgresql+psycopg`),
  `SYNC_DATABASE_URL` (`psycopg2`), pool sizing, and a **15s idle-transaction
  timeout** to stop abandoned row locks from convoying inference admission.
- Storage paths: `MODELS_PATH`, `FILES_PATH`, `CACHE_PATH` (forced absolute).
- CORS / frontend hosts, Sentry DSN (prod only).

### 2.3 Data model (`app/models.py`, Alembic in `backend/alembic/versions/`)

Core tables (all UUID PKs unless noted):

- **Agent** — registered inference host. `name` (unique), `platform`
  (`llamacpp`/`halogen`/...), `type`, `inference_slot_protocol` (v3 supports
  acknowledged, fenced reservations; v0–v2 use legacy admission), `host`,
  `port`, `status`, `gpu_info` (JSON), `last_seen`,
  `websocket_connected`.
- **Model** — GGUF registry. `name` (unique), `path`, `size_bytes` (BigInteger),
  `architecture`, `model_type` ∈ `llm|mtp|mmproj|dflash`, `quantization`,
  capability flags (`supports_embeddings`, `supports_vision`), source provenance.
- **ServerInstance** — a llama-server/Halogen process mirrored from the agent.
  Links `model_id` (+ optional `mmproj_model_id`, `dflash_model_id`), a unique
  public **`alias`** used as the OpenAI `model` id, `engine` + `engine_options`,
  typed settings (`gpu_layers`, `context_size`, `flash_attn`, `mtp_draft_max`,
  `server_options` JSON), `slot_generation`, `effective_capacity`,
  `vram_required_bytes`, inactivity timeout, usage counters, `agent_id`,
  `proxy_url`.
- **InferenceLease** — the scheduling primitive (§4). `request_id` (unique),
  `model_id`, `server_instance_id`, `preferred_server_id`, `required_agent_id`,
  `status` (`queued|reserving|active|released|cancelled|expired|failed`),
  `terminal_reason`, `lease_expires_at`, `slot_generation`, `reservation_id`
  (per-attempt fencing token), `reservation_expires_at`, and partial indexes for
  FIFO queue order and reservation recovery.
- **ResponseRecord** — OpenResponses API turns with `previous_response_id` chaining
  (tree conversations), full input/output item JSON, token counts, store/background
  flags.
- **PromptCache** — cache metadata only; the cache *file* lives on the agent
  (`cache_path`, `llama_cache_id`, TTL, hits).
- **DownloadJob** — model download progress (HF/ModelScope), retries, pause token.
- **TokenUsageSample** — per-request token + llama.cpp stage durations/rates for
  live stats (prompt/predicted tokens/sec surfaced directly from the engine).
- **BenchmarkDefinition** / **BenchmarkRun** — persisted `llama-bench` configs and
  queued/completed runs.
- **File** / **AudioJob** / **BatchJob** — uploaded files (sha256, purpose) and the
  audio/batch jobs that consume them.

> Note: users and API keys were removed (`b7f2d3e9c1a4`); the broker is single-user
> / trusted-network. Conversations were replaced by `ResponseRecord`
> (`b1f8c2a47d90`).

### 2.4 Services (`app/services/`)

| Service | Responsibility |
| --- | --- |
| `agent_manager.py` | Agent registration, persistent WS connection supervisor, event fan-out, offline cleanup, command dispatch, lease invalidation on server death |
| `inference_scheduler.py` | FIFO queueing, `InferenceLease` admission/renewal/release/reconciliation, VRAM room-making, target preparation |
| `inference_stream.py` | SSE keepalives while opening the agent stream and waiting for the first LLM frame; shared dispatch deadline |
| `inference_target.py` | Resolve an OpenAI `model` string (alias / name / UUID) → `InferenceTarget` with optional server/agent pinning |
| `server_startup.py` | Build start payloads, `dispatch_start`/`initialize_server`, `ensure_server_ready`, readiness waits |
| `server_lifecycle.py` | Start/stop/status mirroring of server instances |
| `server_options.py` | Typed llama-server option validation/serialization |
| `cache.py` | Prompt cache metadata operations |
| `models.py` | Model registry operations |
| `benchmark.py` | Benchmark queue worker + run orchestration |
| `token_stats.py` | Aggregate token usage samples, prune loop |
| `gpu.py` | GPU info aggregation from agents |
| `reasoning_metadata.py` | Reasoning-effort → upstream payload mapping |
| `request_activity.py` | Active-stream counter used by the HTTP middleware |
| `scheduler_locks.py` | PostgreSQL advisory locks (per-agent + benchmark) for serialized admission |
| `llama_server.py` | Legacy/local server helper (kept for compatibility) |

`agent_manager` is the single place agent lifecycle events are handled:
`server.started` / `server.stopped` / `server.error` update mirrored
`ServerInstance` rows and **invalidate active `InferenceLease`s** when the server
can no longer serve them, so the scheduler stops handing out dead slots.

---

## 3. Agent (hardware-local)

Python 3.14, FastAPI. Entry: `agent/app/main.py` (`create_app`). Routers:
`servers`, `benchmarks`, `models`, `gpu`, `websocket`, `proxy`.

### 3.1 Lifecycle

On startup (`@app.on_event("startup")`):
1. `frontend_client.start_background_tasks()` — register with the backend and run
   the WS connection loop.
2. `start_gpu_monitoring()` — periodic `gpu.usage` events.
3. `server_manager.start_log_forwarding()` — tail llama-server logs to the backend.
4. `server_manager.start_health_monitoring()` — subprocess health loop.

Required env at import: `AGENT_ID`, `FRONTEND_URL`, `MODELS_PATH` (see AGENTS.md).
`AGENT_PLATFORM`/`AGENT_TYPE` default to `llamacpp`/`generic`.

### 3.2 Registration & connection (`services/frontend_client.py`)

- `POST {FRONTEND_URL}/api/v1/agents/register` with `agent_id`, `name`, `host`,
  `port`, `gpu_info`, `inference_slot_protocol` (v3 for fenced reservations),
  and a report of the **running server set**
  (`running_server_ids` / `healthy_server_ids` / `server_statuses` with
  `slot_generation` + `effective_capacity`).
- The backend dedupes by **name** (the agent-declared `agent_id` is not persisted;
  backend UUIDs are authoritative) and reconciles the reported running set against
  its `ServerInstance` rows — anything not reported as running is marked stopped
  and its leases invalidated (agent re-registration is the reconciliation point).
- After registration, the **backend dials the agent** at
  `ws://{agent.host}:{agent.port}/ws/status` with `X-Agent-ID`
  (`agent_manager._run_agent_websocket`). The agent's `websocket.py` `/ws/status`
  endpoint subscribes to the local `event_bus` and forwards lifecycle events out
  to the backend, alongside periodic `heartbeat` frames
  (`WS_HEARTBEAT_INTERVAL`). Buffered events are **not** replayed on (re)connect —
  fresh state arrives via the next registration's `running_server_ids` instead, to
  avoid stale `server.started`/`server.stopped` flapping.
- **Two distinct sockets per agent**, dialed in opposite directions:
  - **Events: backend → agent** (`agent_manager` opens `ws://agent:8080/ws/status`;
    the agent's `event_bus` publishes `server.*`, `gpu.usage`, `download.progress`
    onto it).
  - **Commands: agent → backend** (`frontend_client.connect_websocket` opens
    `ws://backend/api/ws/agents/{id}` and handles inbound
    `start_server`/`stop_server`/`update_model`).
  In addition, the backend can issue synchronous HTTP commands to the agent via
  `agent_manager.send_to_agent` (used for reconciliation, `/servers/stop`
  fencing, and `/gpu` telemetry). A `_connection_supervisor` keeps the event
  connection alive and fences unreachable agents.

### 3.3 Server process management

`server_manager` is the in-memory source of truth for live llama-server processes
(backend rows are only a mirror). Engines:

- **`llama_server.py`** — `llama-server` subprocess: build argv from
  `ServerConfig` (gpu_layers, context, batch, flash-attn, mmproj, draft models),
  spawn with `LD_LIBRARY_PATH` set to the binary dir, wait for `/health`, monitor
  exit codes, expose `get_effective_capacity` and the `slot_generations` dict
  (per-server generation counters).
- **`halogen_server.py`** / **`halogen_flash_server.py`** — Halogen engines: one
  isolated process per server instance, separate API + engine ports, ring-buffer
  logs, capacity reporting.

`servers.py` `/start`:
- Validates engine vs platform; rejects stale `slot_generation` (409).
- **Idempotent**: a start for an already-running id with equal generation is a
  no-op success (registration re-dispatches starts every ~60s; a second spawn
  would be wrong). A higher generation stops the old process first.
- Auto-downloads models before spawn (Halogen repo, main GGUF, mmproj projector —
  with explicit `mmproj_source` required to avoid mis-resolving the main GGUF as a
  projector — and draft models).
- The **agent owns port allocation** (`_allocate_port`) unless a port is pinned.

`/prepare` reserves/validates without spawning; `/stop`, `/list`,
`/status/{id}`, `/metadata/{id}`, `/logs/{id}`, `/delete` round out lifecycle.

### 3.4 Inference proxy (`services/proxy.py` + `routes/proxy.py`)

`ServerProxy` forwards OpenAI-shaped requests to the correct local llama-server
port and streams SSE back. Notable behaviors:

- **Slot generation fencing** (`_check_generation`) — requests carrying an
  `X-Inference-Slot-Generation` are rejected if the process generation changed,
  preventing a stale lease from touching a recycled process.
- **Capacity enforcement** — optional; coordinates with `inference_operations.py`
  for agent-owned admission. Protocol v3's
  `POST /proxy/{server_id}/reservations/{request_id}` accepts an available slot
  immediately or reports busy; it **does not queue at the agent**. Its in-memory
  operation registry tracks the `request_id`, server generation, one-use
  `X-Inference-Reservation-ID`, and a 15s expiry for unconsumed reservations.
  Dispatch and cancel carry the attempt ID, so a late request cannot consume or
  cancel a newer attempt. `/proxy/{server_id}/operations/...` supports status,
  cancellation, and backend reconciliation. Legacy/unreserved requests can still
  use the agent-side wait.
- **Connect retries** with backoff (`CONNECT_RETRY_DELAYS`) before connecting to
  the engine; protocol-v3 inference is not replayed after dispatch when a read
  times out, because no response bytes does not prove the engine did no work.
- An upstream HTTP error in a stream becomes an SSE `data:` error frame instead
  of a raw JSON body masquerading as a successful empty completion.
- **`proxy_stream_background`** — drains the upstream LLM independently of
  downstream SSE backpressure: the producer owns the slot and closes it at
  upstream **EOF**, even if the downstream client stops reading. This is the fix
  for "a completed llama request holding a slot because a client connection
  remains open" (see AGENTS.md high-risk note).
- Routes expose `/proxy/{server_id}/...`, reservation admission, and operation
  endpoints for tracking/cancelling individual inferences.

### 3.5 Model & GPU services

- `model_manager.py` — list/download/validate GGUF files (HF + ModelScope);
  emits `download.progress` events.
- `gpu_monitor.py` — sample VRAM/utilization; emits `gpu.usage`.
- `log_buffers.py` — `CursorLogRing` for bounded, cursor-addressable server logs.

---

## 4. Scheduling & Inference Leases

`InferenceScheduler` (`inference_scheduler.py`) is the core of multi-request
correctness. It is PostgreSQL-backed so multiple backend workers (the image runs
`uvicorn --workers 4`) share one FIFO.

### 4.1 Constants

- `LEASE_TIMEOUT_SECONDS = 30*60` — max queue wait.
- `ACTIVE_LEASE_TTL_SECONDS = 90`, `LEASE_RENEWAL_INTERVAL_SECONDS = 30` —
  active leases must be renewed or they expire (crash safety).
- `SCHEDULER_POLL_SECONDS = 1.0` — queue poll cadence.
- `RESERVATION_RECOVERY_SECONDS = 12` — DB reservation attempt deadline, shorter
  than the agent's 15s unconsumed-slot expiry. The waiting request or the
  reconciler requeues an abandoned attempt.
- `FIRST_FRAME_TIMEOUT_SECONDS = 150`, `KEEPALIVE_SECONDS = 10`
  (`inference_stream.py`) — one dispatch deadline across agent headers and
  first LLM data, with downstream SSE comment heartbeats during both waits.
- `UPSTREAM_COMPLETION_COOLDOWN_SECONDS = 0.5`,
  `SERVER_READY_COOLDOWN_SECONDS = 3.0` — settling gaps between slot reuse.

### 4.2 `acquire()` flow

`acquire(model_id, request_id, preferred_server_id?, required_agent_id?, timeout,
is_cancelled?)` returns an `InferenceLeaseHandle`:

1. `_queue(...)` inserts a `queued` lease (FIFO by `queued_at`, `id`).
2. Loop until deadline:
   - If client disconnected → cancel lease, raise `InferenceRequestCancelled`.
   - `_admission_open()` — advisory-lock-gated admission check.
   - `_candidates(...)` → running, healthy servers first. With a protocol-v3
     agent, `_claim()` briefly locks the server and compatible FIFO head to check
     capacity and mark the request `reserving` with a fresh attempt ID. It then
     contacts the agent **outside any DB row-lock transaction**. On acceptance,
     a second short, fenced transaction checks request/attempt/server generation
     and transitions to `active`. Busy/error returns to `queued`; disconnected
     or stale attempts are cancelled or recovered. `reserving` requests retain
     their place at the compatible FIFO head and count toward pending capacity.
   - **Mixed-version admission:** protocol-v0 agents use a DB-backed active-lease
     capacity guard under a short server-row lock. Protocol-v1/v2 agents claim
     an active backend lease before waiting for capacity locally at the agent.
     Only v3 reserved inference has the **single-queue** guarantee. Deploy the
     backend before v3 agents; all versions can coexist during rollout.
   - If only `starting`/`stopped` candidates: take the per-agent advisory lock and
     `_prepare_target()` (dispatch start / wait ready), then claim.
   - Otherwise sleep `SCHEDULER_POLL_SECONDS`.
3. On claim, `monitor_disconnect(is_cancelled)` is attached so a dropped client
   mid-stream cancels its operation promptly. A disconnect while `reserving`
   terminalizes that lease; agent cancellation and expiry free any unconsumed slot.
4. Terminal mapping:
   - disconnected queued request → `cancelled`
   - scheduler timeout → `expired`
   - active lease on a stopped/failed/unreachable/expired server → `failed`
   - upstream error/EOF without successful completion → `failed`
   - successful upstream completion → `released` (`terminal_reason=completed`)
   Terminal leases are excluded from queued/active admission queries.

### 4.3 `InferenceLeaseHandle`

Holds `server`, `lease_id`, `slot_generation`, optional v3 `reservation_id`, a
`lost` event, and background renewal + disconnect-monitor tasks. Key methods:

- `guard(awaitable, cancelled?)` — runs one upstream operation, cancelling it if
  the lease is lost or the client disconnects; raises `InferenceLeaseLost` or
  `InferenceRequestCancelled`.
- `mark_upstream_started()` — lets the renewal loop renew a stream owned outside
  `guard` (used by SSE streaming).
- `dispatch_headers()` — carries request ID, generation, and, for v3, the
  reservation attempt ID through the agent proxy.
- `release(outcome)` / `cancel()` — outcome-aware terminal transitions; cancellation
  is fenced by the reservation token. Legacy capacity retains a settling cooldown.

### 4.4 Streaming lease lifecycle

In `v1_chat_completions.py` and `v1_completions.py`, the route acquires a lease,
then opens the agent stream while sending SSE `: keep-alive` comments to the
client. The same 150s deadline bounds header acquisition and the first `data:`
frame; later idle reads retain transport timeouts. The backend requires an
upstream completion marker rather than treating premature EOF as success. It
records an error event and failed lease for upstream failures; normal completion
releases the lease before downstream usage/`[DONE]` frames finish draining.
The agent's `proxy_stream_background` drains the LLM independently of client
backpressure and releases its slot at upstream EOF. Neither layer holds a slot
solely because an idle downstream client has not finished consuming SSE.

### 4.5 Reconciliation

`reconcile_stale_leases()` (every 30s) and `reconcile_persisted_leases()`
(at startup) recover expired `reserving` attempts, expire queued requests, and
query agent operations before terminalizing stale active work on healthy
servers. A stopped/unhealthy server or changed `slot_generation` fences active
leases. `_make_room()` evicts/stops idle servers (VRAM-aware via
`vram_required_bytes` vs live VRAM) to admit a higher-priority target.

---

## 5. OpenAI-Compatible API Surface

Mounted at `/v1` (`backend/app/api/routes/v1/`):

| Endpoint | Notes |
| --- | --- |
| `GET /v1/models` | Lists server aliases + models |
| `POST /v1/chat/completions` | SSE streaming via agent proxy, lease-guarded, reasoning-effort aware |
| `POST /v1/completions` | Legacy completions |
| `POST /v1/embeddings` | Embeddings (capability-checked) |
| `POST /v1/responses` (+ WS, `/responses/compact`) | OpenResponses API: tree conversations via `previous_response_id`, event-streamed, persisted as `ResponseRecord` |
| `POST /v1/files` | Upload; sha256 + purpose |
| `POST /v1/batches` | Batch jobs over JSONL files |
| `POST /v1/audio/*` | Transcription / translation / speech (system Whisper) |

The WebUI may send `X-Inference-Request-ID: chatcmpl-{uuid}` on chat requests
and poll `GET /api/v1/queue/{request_id}` for `queued`, `reserving`, `active`,
and terminal status. Other OpenAI clients need not set this header. The WebUI
also surfaces streamed errors rather than silently treating them as empty output.

`resolve_inference_target()` maps the `model` field: a **ServerInstance alias**
pins `preferred_server_id`; a model name/UUID allows any compatible replica; an
optional `agent_id` pins `required_agent_id`. The Responses router additionally
maps chain history into the llama payload (`_build_chain_history`,
`_llama_payload`).

---

## 6. Frontend

React + TypeScript + TanStack Router + shadcn/ui, built with Bun/Vite, served by
the backend image (`app.frontend`).

- `src/client/` — **generated** OpenAPI SDK (`sdk.gen.ts`, `types.gen.ts`) from
  `openapi.json` via `openapi-ts`. Regenerated by
  `bash scripts/generate-client.sh`; never hand-edited.
- `src/routes/_layout/*` — pages: dashboard (`index`), `agents`, `models`,
  `server-instances`, `chat`, `completions`, `embeddings`, `responses`,
  `benchmarks`, `audio`.
- `src/components/` — feature components (Agents, Models, ServerInstances,
  Benchmarks, Queue, Stats, Common, Sidebar, ui).
- `src/routeTree.gen.ts` — generated by TanStack Router; delete + restart Vite on
  unexpected 404s.
- Live UI uses the backend WebSockets: `/api/ws/events/{agent_id}` (per-agent
  event stream) and `/api/ws/queue-status` (aggregate queue/slot snapshot).
  Chat additionally polls its own `/api/v1/queue/{request_id}` while waiting;
  OpenAI SSE keepalive comments do not change the client-visible chunk schema.

---

## 7. Deployment & Build

### 7.1 Images

- **Matrix app** (root `Dockerfile`): multi-stage — Bun builds the frontend →
  copied into `backend/app/frontend`; `python:3.14` + `uv sync --package app`;
  entrypoint `docker/entrypoint.sh` + `docker/entrypoint.d/` (010 generate-config,
  020 run-migrations). Runs `uvicorn app.main:app --host 0.0.0.0 --port 8000
  --workers 4 --ws-ping-interval 30 --ws-ping-timeout 60`. Serves WebUI + API +
  broker on 8000.
- **Agent** (`agent/Dockerfile`): `python:3.14-slim` base, no llama.cpp bundled;
  `AGENT_PLATFORM`/`AGENT_TYPE` env; uvicorn on `AGENT_PORT` (8080).
- **Recipes** (`recipes/`): layer the agent onto GPU backends —
  `llama-cpp-vulkan` (ghcr llama.cpp full-vulkan), `llama-cpp-q38rocm`
  (ROCmFP4 / Strix Halo), `halogen-rocm`, `halogen-flash`. Built images run in
  root's Podman space on the agent host.

### 7.2 Topology (production, per AGENTS.md)

- Matrix app container `matrix-app` on `core@10.100.2.100`, served at
  `https://matrix.thelink.family` via Traefik + Let's Encrypt.
- Inference agent types on `core@10.100.2.111` (Podman, root space).
- PostgreSQL 16 for metadata.
- Confirm `GET /api/v1/agents` reports `inference_slot_protocol: 3` for agents
  serving inference before attributing live queue behavior to the v3 handoff.
  A registered value of `0`, `1`, or `2` selects legacy admission even when the
  broker and database migration are current.

### 7.3 Compose caveat

Compose files are inconsistent (see AGENTS.md): `compose.yml` defines `postgres` +
`frontend` (+ `agent`), while `compose.override.yml`/helper scripts refer to `db`
and `backend`. Inspect the selected files before any service-specific command.

### 7.4 CI (`.github/workflows/build-and-push.yml`)

Backend job: install uv → ruff → `scripts/prestart.sh` (migrations) → `pytest`.
Agent job: ruff → pytest. Image jobs: build & push matrix-app, agent, and recipe
images; they trigger on pushes to `main`/`develop`/version tags and on PRs to
`main` (pushes are gated so PR runs build-only). Most other workflow files are
`.disabled`.

---

## 8. Cross-Cutting Concerns

### 8.1 Security

No auth between backend and agent — trusted internal network only. Users/API keys
removed. The agent must never be exposed externally; all external traffic goes
through the backend (TLS via Traefik). Secrets kept out of logs.

### 8.2 Failure handling

- **Agent unreachable**: backend WS supervisor fences the agent, marks its
  `ServerInstance`s unavailable, invalidates active leases; queued requests fail
  or wait; on reconnect, re-registration reconciles the running set.
- **llama-server crash**: agent health monitor detects exit, emits
  `server.stopped`/`server.error`; backend mirrors + invalidates leases.
- **Backend restart**: startup and periodic reconciliation recover pending
  reservations and check stale active leases against agent operations; agents
  keep running processes and re-register to re-sync.
- **Client disconnect mid-stream**: the `is_cancelled` monitor fences the lease
  and cancels the agent operation; the HTTP middleware separately tracks request
  activity until its stream closes. Unconsumed v3 reservations also expire
  independently on the agent.

### 8.3 Observability

- `/api/v1/metrics` (Prometheus-style) + snapshot loop.
- `/api/v1/stats` + `TokenUsageSample`-backed live token rates.
- WS event streams (per-agent + queue-status) feed the WebUI; the agent
  `/ws/status` endpoint emits periodic `heartbeat` frames
  (`WS_HEARTBEAT_INTERVAL`) for liveness.
- Per-request `/api/v1/queue/{request_id}` statuses and correlated reservation,
  stream-open, first-frame, and stream-close logs identify where admission or
  dispatch stalled without exposing prompts in the queue-status response.
- Structured JSON logs; llama-server/Halogen logs forwarded via ring buffers.

### 8.4 Extensibility

- New inference engines: add a manager in `agent/app/services/` mirroring
  `llama_server.py`'s interface (start/stop/health/capacity/slot_generation) and
  register its platform/engine in `servers.py` validation.
- New OpenAI endpoints: add a `v1_*` router; reuse `resolve_inference_target` +
  `inference_scheduler.acquire` + `lease.guard` for correct slot semantics.
- Multi-agent load balancing / failover: already supported by candidate selection
  and `required_agent_id` pinning; VRAM-aware room-making generalizes to pools.

---

## 9. Key Invariants (do not break)

1. The agent owns live llama-server processes in memory; backend `ServerInstance`
   rows are a mirror, reconciled at agent (re-)registration.
2. Backend-generated UUIDs are authoritative server/agent identity; agent-declared
   `agent_id` is not persisted.
3. Inference admission is FIFO and lease-backed in PostgreSQL; v3 `reserving`
   claims must be fenced by both reservation ID and server generation. Terminal
   leases (`released`/`cancelled`/`expired`/`failed`) must never appear in
   queued/active admission queries.
4. A slot is held from agent reservation through upstream completion/cancellation,
   never solely by an idle downstream SSE connection. The v3 agent may not put
   an already-admitted request into a second capacity queue.
5. `slot_generation` fences recycled processes from stale leases.
6. Backend route/schema changes require `bash scripts/generate-client.sh`; never
   hand-edit `frontend/src/client/*` or `routeTree.gen.ts`.
