# Inference Matrix — Architecture

Inference Matrix is an OpenAI-compatible inference broker that schedules GGUF
model inference across one or more hardware machines. The admin container is
the brain (database, scheduler, client API); **provider agents** are
hardware-local containers that each own one inference **backend type** and may
run **one or more backend processes** (instances) of that type on a single
machine.

**Status: this document describes the litellm-based architecture on the
`litellm-architecture-overhaul` branch.** It supersedes the legacy
broker/agent design; the pre-overhaul code was removed in the final
cleanup and lives in git history only.

> **Phase 16 (machine-scoped agents) is largely IMPLEMENTED** (slice 3: the
> atomic cutover — `ProviderInstance` re-keyed onto `ProviderAgent`,
> `registration_token`/shells retired, registration/WS/scheduler/config-push/
> metrics/logs re-addressed at the agent level with per-backend
> `instance_id`). Remaining Phase 16 slices: 4 (`max_running_backends`
> hot-swap), 5 (`agent.assignments.update` push), 6 (real multi-backend-per-
> process engine hosting), 7 (React Agents/placement UI), 8 (full protocol-doc
> rewrite). See `IMPLEMENTATION_STATUS.md` Phase 16 for the exact state. The
> architecture below is the law; where code still lags a slice, the code is
> what must move.

---

## 1. System Overview

```
                      ┌───────────────────────────────────────────────┐
                      │               ADMIN CONTAINER                 │
                      │  FastAPI (1 uvicorn worker) + Redis + Postgres│
                      │                                             │
  Client             │  /                 Swagger UI               │
  (OpenAI SDK /      │  /openapi.json     OpenAPI schema           │
   WebUI)            │  /admin/*          React admin UI           │
    │  HTTPS /v1/*   │  /admin/api/*      Admin management API     │
    └───────────────▶│  /v1/*             Public inference API     │
                      │      │  (driven by litellm SDK)             │
                      │      ▼                                      │
                      │  InferenceScheduler  ── FIFO + VRAM admit   │
                      │      │                                      │
                      │      │  WS: commands ↓ / events ↑           │
                      └──────┼──────────────────────────────────────┘
                             │  /provider/ws  (one socket per AGENT)
         ┌───────────────────┼───────────────────────┐
         ▼                   ▼                       ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │  MACHINE A       │   │  MACHINE B       │   │  MACHINE N       │
    │  agent(t1) :P1   │   │  agent(t1) :P1   │   │  agent(tX) :P1   │
    │   ┌────┐ ┌────┐  │   │   ┌────┐         │   │   ┌────┐         │
    │   │bknd│ │bknd│  │   │   │bknd│         │   │   │bknd│         │
    │   │:i0 │ │:i1 │  │   │   │:i0 │         │   │   │:i0 │         │
    │   └────┘ └────┘  │   │   └────┘         │   │   └────┘         │
    │  (same type)     │   │  (+agent(t2):P2) │   │                  │
    └────────┬─────────┘   └──────────────────┘   └──────────────────┘
             │  litellm → http://<machine>:<agent PROVIDER_PORT>/v1
             │  (the agent routes each request to a backend by model)
             └──────────────────────────────────────────────────────
```

A **machine** may host several **agents** (one per `provider_type`, and even
several of the same type — discriminated by `AGENT_ID`). Each agent runs
1..N **backends** (one per assigned `ProviderDefinition`) but publishes
exactly **one** admin-facing HTTP port — its container env `PROVIDER_PORT`
(recorded on the agent as `base_port`). The agent serves a single
OpenAI/OpenResponses-compliant `/v1` surface on that port and dispatches each
request to the correct backend by the request's `model` field (which equals the
`ProviderDefinition.alias`); each backend's engine process binds a **private
internal port** inside the container that the admin never learns or dials. The
single agent WebSocket multiplexes commands/events for all of its backends,
addressed by the per-backend `ProviderInstance` id.

### Components

| Component | Role |
| --- | --- |
| **Admin** (`admin/backend`) | Owns Postgres + Redis, the scheduler, conversation state, the client-facing `/v1` API, the admin UI/API, and the provider-agent WebSocket registry. Runs **litellm** to drive inference. Knows **nothing** provider-specific. |
| **Provider agent** (`provider/<type>`) | Hardware-local container tied to **one machine + one provider type** (and a stable `AGENT_ID`). Owns 1..N backend subprocesses of that type, their lifecycles, metrics, logs, and downloads. Serves a **single** fully OpenAI/OpenResponses-spec-compliant `/v1` HTTP surface on its env `PROVIDER_PORT` and routes each request to the right backend by the `model` field (= the definition `alias`); each backend's engine binds a private internal port the admin never sees. Dials the admin over **one** WebSocket. |
| **Backend** (`ProviderInstance`) | One inference process (llama-server, halogen, gufo, …) for one `ProviderDefinition` on one agent. A private detail of the agent; the admin reaches it only over the agent's WS (control) or the agent's single `/v1` listener (data, routed by model) — **never** a per-backend port. |
| **Machine** | A physical/virtual host with a unique `uid`, pre-registered in the admin UI, holding the shared **machine registration secret** agents authenticate with, and VRAM capacity tracked for scheduler admission. |
| **Redis** | Scheduler queues/mirrors, VRAM ledger, WS presence/secrets/epochs (agent-level), metrics-ownership leases. |
| **Postgres** | Source of truth for configuration (machines, agents, provider definitions, backend instances) and stored responses. |

### Key invariants

1. A **provider agent** is bound to exactly one machine and one provider
   **type**, and runs **one or more** backends (instances) of that type. A
   `ProviderInstance` is exactly one backend for one `ProviderDefinition` on
   one agent.
2. A type may declare **`max_running_backends`** (e.g. `1` for halogen-flash).
   The admin scheduler enforces it **per agent**: at most that many backends
   of the type are `running` on a given agent at once; to boot another the
   scheduler hot-swaps (evicts the running one, LRU-first).
3. The provider agent is in full control of each backend's lifecycle and
   inference-slot admission. Slot capacity is enforced **at the provider**,
   tied to the inbound connection lifecycle — a dead admin/litellm
   connection cannot leak a slot.
4. The admin never proxies raw provider-quirk traffic. Everything between
   admin and agent is either (a) the WS control protocol (one socket per
   agent, frames addressed by backend `ProviderInstance` id) or (b)
   standard OpenAI-compatible HTTP driven by litellm against the **agent's
   single env-port `/v1` listener**, which multiplexes to the correct backend
   internally by the request's `model`. The admin carries zero
   provider-specific routing logic and never dials a per-backend port.
5. The client-facing `/v1/responses` SSE stream must pass the
   openresponses.org conformance suite. That client boundary is the
   fidelity contract; internal layers are free as long as it holds.
6. The admin owns the client-facing `resp_<uuid>` id and the conversation
   chain. litellm-side session state is never relied upon.

---

## 2. Repository Layout

```
admin/
  backend/                FastAPI admin + public inference API (package: matrix-admin)
    app/
      main.py             App factory: docs at "/", /openapi.json, routers, lifespan
      models.py           SQLModel tables (squashed schema, see §4): Machine,
                          ProviderAgent, ProviderType, ProviderDefinition,
                          ProviderInstance, ResponseRecord, TokenUsageSample
      api/
        admin/            /admin/api/*  (health, provider registration)
        v1/               /v1/*         (public inference)
        ws.py             /provider/ws  (provider WebSocket)
      services/
        scheduler.py      InferenceScheduler: in-process FIFO + Redis mirror (§6)
        connection_manager.py  WS registry, auth, epochs, command/ack (§5)
        presence_sweep.py      Background stale-connection sweeper
        sse.py                 SSEEmitter: client-facing stream framing (§7)
        alias_registry.py      litellm model-alias registration (§7)
        metrics_service.py     Machine metrics ownership assignment (§8)
        config_update.py       Phase 9 provider.config.update push + fingerprint self-heal
        wire.py                Admin mirror of the provider wire envelope
        redis_keys.py          Redis key layout (§9)
    alembic/              ONE squashed initial migration
    docs/redis-keys.md    Redis key reference
  frontend/               React + Bun + TanStack Router UI (Vite base: "/admin")
provider/
  lib/                    Shared provider library (package: provider_lib)
    provider_lib/
      admin_client.py     Agent registration (machine+type+agent_id), one WS
                          dial multiplexing N backends, bearer auth, backoff
      wire.py             Canonical wire envelope (Frame, FrameKind, Ack)
      backend.py          BackendDriver ABC + BackendLifecycle (slot mgmt)
      app_factory.py      FastAPI app: the agent's single env-port /v1 surface, routed to a backend by model
      config.py           ProviderSettings (env-derived)
      downloader.py       Model downloads with progress events
      metrics.py          GPU / RAM / CPU / storage collectors
    tests/
  mock/                   Mock provider: hardware-free full-path dev
  llama-cpp/              Provider type: llama.cpp backend
  halogen/                Provider type: halogen (ROCm)
  halogen-flash/          Provider type: halogen-flash (NPU)
  gufo/                   Provider type: gufo
spike/litellm-fidelity/   Phase 0 spike: litellm Responses streaming fidelity
docs/                     ws-protocol.md (canonical RFC), integration-testing.md
compose.yml               postgres + redis + admin + provider-mock
Dockerfile                Builds admin (frontend + FastAPI) image
```

(The pre-overhaul `legacy/` tree was removed in the final cleanup; see
git history.)

Provider packages are uv workspace members (`admin/backend`,
`provider/lib`, `provider/mock`, `provider/llama-cpp`, `provider/gufo`,
`provider/halogen`, `provider/halogen-flash`, `spike/litellm-fidelity`).
The admin image is built with
`uv sync --no-install-workspace --package matrix-admin`, so it does **not**
ship provider packages — hence `app/services/wire.py` deliberately
duplicates the frame shape from `provider_lib.wire` (keep them in sync;
`docs/ws-protocol.md` is canonical).

---

## 3. URL Layout

| Path | Served by | Auth | Notes |
| --- | --- | --- | --- |
| `/` | admin | none | Built-in Swagger UI (`docs_url="/"`) |
| `/openapi.json` | admin | none | Full OpenAPI schema of the admin app |
| `/admin/*` | admin (static SPA) | none | React UI, Vite `base: "/admin"` |
| `/admin/api/health` | admin | none | Health check (used by compose healthcheck) |
| `/admin/api/providers/register` | admin | **machine secret** | Provider **agent** registration (§5). Body carries `machine_uid`, `agent_id`, `provider_type`, `version`, `base_port`, `hardware`, `metrics_categories`, `schema`. Authenticated by the shared `Machine.registration_secret` (not a per-definition token). `base_port` is the agent's single published admin-facing `/v1` port (its env `PROVIDER_PORT`); the admin does **not** police port uniqueness — each agent owns its own env port and routes to its backends internally by model. |
| `/admin/api/machines*` | admin | none (trusted LAN) | Machine CRUD (Phase 9; delete refused while agents attached). Phase 16: the machine row carries the shared `registration_secret` shown/rotated here. |
| `/admin/api/agents*` | admin | none (trusted LAN) | **ProviderAgent** reads + placement (Phase 16): list agents per machine/type, see hosted backends. Cherry-picked definition placement targets reference agent ids here. |
| `/admin/api/definitions*` | admin | none (trusted LAN) | ProviderDefinition CRUD (Phase 9; `backend_config`/`capacity` PATCH pushes `provider.config.update` to the backends hosting it; explicit null on required fields → 422; Phase 12: `backend_config` validated against the committed JSON Schema of its `ProviderType` → 422 with per-field errors). **Phase 16:** `provider_type` is required at create (no shells); placement is `any_of_type` or an explicit set of `agent_ids` (link table); placement/config edits push `agent.assignments.update` / `provider.config.update` over the affected agents' sockets. |
| `/admin/api/provider-types` `/admin/api/provider-types/{name}` | admin | none (trusted LAN) | ProviderType registry reads (Phase 12; UI renders the backend-config form from the committed schema) |
| `/admin/api/provider-types/{name}/pending/commit` `.../pending/dismiss` | admin | none (trusted LAN) | Operator override of the schema-consensus state (Phase 12) |
| `/admin/api/huggingface/search` `/admin/api/huggingface/files` | admin | none (trusted LAN) | HF proxy for the `hf-file` picker widget (Phase 12; ported from legacy `huggingface.py` routes) |
| `/admin/api/instances/{id}/cache/clear` `/admin/api/instances/{id}/storage/prune` | admin | none (trusted LAN) | Instance storage actions (Phase 9) |
| `/admin/api/instances/{id}/backend/start` `.../backend/stop` `.../backend/restart` `/admin/api/instances/{id}/initialize` | admin | none (trusted LAN) | Manual control of **one backend** (`ProviderInstance`), routed over its **agent's** WS (`provider_lib.ops`): start/restart/initialize answer **202 accepted** by default and the transition runs at the agent (body `{"wait_for_running": true}` waits instead, bounded by `BACKEND_BOOT_TIMEOUT_SECONDS`); the agent's `backend_in_use` / `boot_in_progress` NAKs surface as 502 with `step` + `retry_after` |
| `/admin/api/instances` `/admin/api/instances/{id}` | admin | none (trusted LAN) | Provider-instance reads for the UI (Phase 10) |
| `/admin/api/instances/{id}/logs` | admin | none (trusted LAN) | Backend/provider log tail (Phase 13; Redis-backed, `kind`/`since`/`limit` cursor) |
| `/admin/api/responses` `/admin/api/stats/usage` `/admin/api/stats/overview` | admin | none (trusted LAN) | Response log + usage/dashboard reads (Phase 10; overview reads the scheduler Redis mirror, observability only) |
| `/v1/*` | admin | none (trusted LAN) | Public OpenAI-compatible inference |
| `/provider/ws` | admin | bearer (**agent secret**) | Provider **agents** dial in — one socket per agent, frames addressed by backend `ProviderInstance` id (§5) |

Each **agent** publishes exactly **one** admin-facing OpenAI-compatible HTTP
port — its container env `PROVIDER_PORT`, recorded on the agent as `base_port`
— reachable from the admin at `http://{machine.reachable_address()}:{agent.base_port}/v1/...`.
The agent serves a single `/v1` surface on that port and dispatches each request
to the correct backend by the request's `model` field (the definition `alias`);
each backend's engine process binds a private internal port inside the container
that the admin never learns or dials. `Machine` exposes
`dns`/`host`/`ip`; `reachable_address()` prefers `dns or host or ip`.

---

## 4. Data Model

All tables are created by **one squashed Alembic initial migration**
(`admin/backend/alembic/versions/0140d6ad9f48_initial_squashed_schema.py`),
applied by `scripts/prestart.sh` and by CI; Phase 12 adds the
`provider_types` table (plus the two instance columns) in a follow-up
migration. **Phase 16 adds `provider_agents` and a `definition_agents` link
table, moves the registration secret onto `machines`, re-keys
`provider_instances` onto `(agent_id, provider_definition_id)`, drops
`provider_definitions.registration_token`, and adds
`provider_types.max_running_backends`.** The provider agent itself has **no
database** — it derives everything from env + the registration response,
mirrored here by the admin.

### Machine
A host that provider agents run on. **Created in the admin UI before
any agent registers against its `uid`.**

| Field | Notes |
| --- | --- |
| `uid` | Stable identifier supplied by the operator, referenced by the agent's `MACHINE_UID` env. Unique. |
| `name` | Display name. Unique. |
| `registration_secret` | **Phase 16.** Shared secret every agent on this machine authenticates registration with (replaces the per-definition `registration_token`). Shown/rotated in the UI. |
| `host` / `dns` / `ip` | How the admin reaches the agents (and thus their backends, via each agent's single env-port `/v1`) on this machine. |
| `total_vram_bytes` | Admission budget; the **auto-sum of the `hardware.gpus` union** (recomputed on each registration and on agent delete). A manually-set value survives only until an agent reports a `gpus` list. |
| `hardware` | JSON inventory: `{"gpus": [{"uuid","vendor","name","total_vram_bytes"}, ...], "cpu": {...}, "ram": {...}}`. `gpus` is the **union by uuid across every agent on the machine** (latest report wins per uuid); `cpu`/`ram` stay last-writer-wins. |

An agent registering with an unknown `MACHINE_UID` is **rejected (404)**; a
wrong `registration_secret` is **rejected (401)**. UIDs must not be reused
across physical hosts (see Known Limitations).

### ProviderAgent
**Phase 16.** One hardware-local container: a `(machine, provider_type,
agent_id)` triple. The agent owns 1..N backends (`ProviderInstance`s) of its
single type and holds the one WebSocket the admin commands it over.

| Field | Notes |
| --- | --- |
| `machine_id` | FK → Machine. The host this agent runs on. |
| `provider_type` | FK → ProviderType. **All** the agent's backends are this type. Immutable while backends are attached. |
| `agent_id` | Operator-supplied stable id (the container's `AGENT_ID` env) that discriminates multiple agents sharing a `(machine, provider_type)`. Unique on `(machine_id, provider_type, agent_id)`. |
| `base_port` | The agent's **single** published admin-facing `/v1` port (its container env `PROVIDER_PORT`). The admin reaches every backend of this agent at `machine.reachable_address():base_port` and the agent routes by model. No per-backend offsets. |
| `version` | Agent version (commit id until first release); hard-fail gate vs admin `VERSION`. |
| `agent_status` | `registering` `initializing` `running` `unhealthy` `error` `disconnected` — the container-level state (was the old per-instance `instance_status`). |
| `websocket_connected` / `epoch` / `last_seen` | Agent-level WS liveness mirror (authoritative liveness is the Redis presence key). |
| `reported_schema_fingerprint` | Schema fingerprint presented at the agent's last registration attempt (drives `waiting_schema` + the consensus voter roster). |
| `assigned_gpus` | Agent-reported GPU uuid strings; used for the machine `hardware.gpus` union / survivor-aware stale-drop — not full descriptors. |

The agent's per-connection secret (`im:ws:secret:{agent_id}`) is minted at
registration and stored in Redis only (see §5, §9, §12).

### ProviderType
A registered provider **type** (Phase 12), created by the first
registration of that type. Owns the JSON Schema (2020-12) that describes
the type's `backend_config`; every `ProviderDefinition.provider_type`
must reference a registered type.

| Field | Notes |
| --- | --- |
| `name` | Type id (`llama-cpp`, `halogen`, `halogen-flash`, `gufo`, `mock`, …). Unique. |
| `schema` | Committed JSON Schema (2020-12) for this type's `backend_config`. Drives admin validation on write **and** the UI form render. |
| `max_running_backends` | **Phase 16.** Per-agent cap on simultaneously `running` backends of this type, declared in the shipped `schema.json` (top-level `x-max-running-backends`, default unlimited). `1` for halogen-flash (single NPU). Enforced by the scheduler (§6). |
| `serves_modalities` | **Phase 18.** Which client-facing endpoint kinds this type can host, declared in the shipped `schema.json` (top-level `x-serves-modalities`, default `["llm"]`). Values ∈ `llm` \| `embedding` (`audio` reserved). A `ProviderDefinition`'s `modality` must be in this list or create/PATCH is refused 422. `llama-cpp` (vulkan + cuda) declares `["llm","embedding"]`; every other type stays `["llm"]`. Read into the column at every schema-commit point exactly like `max_running_backends`. |
| `schema_fingerprint` | SHA-256 of canonical `schema` (same canonicalization as `config_fingerprint`). |
| `pending_schema` / `pending_fingerprint` | Staged schema awaiting consensus (null when none). |
| `pending_voters` | JSON list of **agent ids** that have registered presenting `pending_fingerprint` (Phase 16: voters are agents, not backends). |
| `status` | `active` \| `consensus_pending` \| `conflict`. |

**Schema consensus** (all known **agents** of the type must agree):
registration with the committed fingerprint proceeds normally. A
different fingerprint stages it as pending and the registration is
**refused 409 `schema_pending`** — the agent keeps retrying (waiting
for all agents to update). Repeat registrations with the pending
fingerprint add voters; when voters cover **every `ProviderAgent`
row of that type**, the pending schema is committed. A third distinct
fingerprint while pending → 409 `schema_conflict`. Operator override:
force-commit (e.g. a permanently dead machine can never vote) or
dismiss. Force-commit applies to **future registrations and new/edited
definitions only** — connected old-schema agents are not force-
converged. Full algorithm in `docs/ws-protocol.md` §2.

The `PROVIDER_TYPES` constant is gone; the registry is the source of
truth. The static `schema.json` lives in each provider package
(`provider/<type>/provider_<type>/schema.json`) and is shipped by the
agent at registration — the admin never holds provider code.

### ProviderDefinition
A client-facing model: how to boot a backend and how to schedule it. One
definition may be hosted by many agents (fan-out), each running its own
backend (`ProviderInstance`).

| Field | Notes |
| --- | --- |
| `alias` | Public model name clients use in `/v1/models` and the `model` field. Unique. |
| `provider_type` | Must reference a registered `ProviderType`. **Phase 16: required at create — the Phase 14 shell (NULL type adopted at registration) is removed.** A type change is refused 409 while backends are attached. |
| `modality` | **Phase 18.** The endpoint kind this definition serves: `llm` (default) \| `embedding` (`audio` reserved). The admin's **routing key**: `/v1/embeddings` accepts only `embedding` aliases; `/v1/responses` + `/v1/chat/completions` accept only `llm` aliases (the other kind is a clean 404). Must be in the `ProviderType.serves_modalities` list or create/PATCH is refused 422. Pushed to the provider on the assignment / `provider.config.update` / registration payloads so the engine boots correctly (llama-cpp adds `--embedding` for `embedding`). Immutable while backends are attached (same gate as `provider_type`). |
| `backend_config` | JSON handed to the agent to start a backend for this definition: model artifacts (main GGUF + mmproj + draft, each with source), engine args, engine options. **Validated against the committed JSON Schema of its `ProviderType` on create/PATCH (Phase 12)**; the UI form is rendered from that schema (collapseable sections, `hf-file` artifact widget). Required (no shells). |
| `agent_placement` | **Phase 16.** `any_of_type` (host on every agent whose `provider_type` matches) or `specific` (host only on the agents listed in `definition_agents`). Drives which agents receive `agent.assignments.update` and which backends the scheduler may pick. |
| `agents` (link) | **Phase 16.** `definition_agents` join table (`provider_definition_id`, `agent_id`) — populated only when `agent_placement = specific`. |
| `vram_required_bytes` | Scheduler admission hint (per backend). |
| `idle_timeout_seconds` | Admin-driven idle stop (reaper — §6). |
| `capacity` | Concurrent inference **slots within one backend** (≥1). Distinct from `max_running_backends` (how many backends of the type run per agent). |
| `model_metadata` | OpenAI model metadata discovered at init. |
| `enabled` | Disabled definitions are excluded from scheduling. |

> **Phase 16 removals:** `registration_token` (auth moves to the shared
> `Machine.registration_secret`) and the shell/`awaiting_config` machinery
> from Phase 14 (a definition is always typed and configured at create).

### ProviderInstance
**One backend**: one `ProviderDefinition` running on one `ProviderAgent`.
Unique on `(agent_id, provider_definition_id)` — an agent cannot run two
backends of the same definition. Phase 16 moves the container-level fields
(`instance_status`, `version`, `websocket_connected`, `epoch`,
`reported_schema_fingerprint`) up onto `ProviderAgent`; this row is purely
the per-backend state the scheduler and lifecycle act on.

| Field | Notes |
| --- | --- |
| `agent_id` | FK → ProviderAgent (was `machine_id`). The backend's owning container. |
| `backend_status` | `stopped` `initializing` `starting` `running` `in_use` `stopping` `error` |
| `last_request_at` | Idle tracking (liveness is the agent's). |
| `backend_loaded_at` | When the backend last entered `running`/`in_use` (stamped by the status ingest on transition; cleared when it leaves the loaded set or the agent's socket dies). The idle reaper's window baseline is `max(last_request_at, backend_loaded_at)` so a freshly booted, never-requested backend is not reaped against a stale clock. |
| `config_fingerprint` | SHA-256 of the applied `backend_config`; drives auto cache-clear and the `provider.config.update` push (admin PATCH + reconnect self-heal). |
| `assigned_gpus` | Reserved — currently unused (VRAM is accounted per booted instance; metrics attribution is per-agent). |

### ResponseRecord
Stored OpenResponses turn (spec `ResponseResource`), chained by
`previous_response_id`. The admin owns `response_id` (`resp_<uuid>`);
litellm's affinity-wrapped upstream id is stored in `parameters` for turn
affinity while the DB chain stays ours. `input_items`/`output_items` hold
spec-shaped payloads; token counts and `store`/`status`/error fields
mirror the spec. Since Phase 7 this table also stores chat-completions
turns: `response_id` is the minted `chatcmpl-<uuid>`,
`parameters.api_format="chat_completions"` marks the format, and
`previous_response_id` is always `NULL` (chat is stateless client-side —
the caller resends the full `messages`, which are stored as
`input_items`).

### TokenUsageSample
Per-request prompt/cached/completion token + rate telemetry, FK to
provider instance + definition.

### Dropped by the overhaul
`agents`, `server_instances`, `models` (folded into `backend_config`),
`inference_leases` (replaced by the scheduler + provider admission),
`prompt_cache` (provider-local via fingerprint), `download_jobs`
(replaced by `download.progress` events), `benchmark_*`, `files`,
`batch_jobs`, `audio_jobs`, `users`/`api_keys`.

---

## 5. Registration & Connection Protocol

Full wire-level detail lives in **`docs/ws-protocol.md`** (canonical RFC).
Summary:

### Provider agent environment (all env-derived, no DB)
| Variable | Required | Meaning |
| --- | --- | --- |
| `MACHINE_UID` | yes | UID of the Machine this container runs on |
| `MACHINE_SECRET` | yes | The shared `Machine.registration_secret` proving the container belongs to that machine (**replaces `PROVIDER_REGISTRATION_TOKEN`**) |
| `AGENT_ID` | yes | Stable operator-set id discriminating multiple agents that share a `(machine, provider_type)` |
| `PROVIDER_TYPE` | yes | The single backend type this agent runs (must match a registered `ProviderType`) |
| `ADMIN_BASE_URL` | yes | e.g. `http://admin:8000` |
| `PROVIDER_PORT` | no (8081) | The agent's **single** admin-facing `/v1` port (published to the bridge network). The agent routes each request to a backend by model on this one port. Engine (backend) ports are internal to the container and OS-assigned by default (bind `127.0.0.1:0`); the admin never learns or dials them. |
| `CACHE_DIR` | yes (`/cache`) | Prompt caches, persisted `provider_config.json` |
| `MODELS_DIR` | yes (`/models`) | Model artifact storage |
| `METRICS_CATEGORIES` | no | Space-delimited: `gpu_usage vram os_ram cpu storage`. Inference metrics are **always** enabled and must not appear here. |

`LLAMA_SERVER_PATH` and other backend-binary paths come from the
environment, never from `backend_config`.

### Sequence
```
Provider agent                      Admin
    │ POST /admin/api/providers/register
    │  {machine_uid, machine_secret, agent_id, provider_type, schema,
    │   version, base_port, hardware, metrics_categories}
    │ ──────────────────────────────▶
    │  validations (401/404/409):
    │    machine_uid exists + machine_secret matches   (401/404)
    │    provider_type registered in ProviderType   (Phase 12;
    │      unknown type → created from this schema, committed)
    │    schema consensus gate                      (Phase 12; voters =
    │      agents of the type; mismatch → 409 schema_pending/conflict)
    │    version == admin settings.VERSION   (HARD FAIL 409)
   │  effects:
   │    merge hardware into Machine; upsert ProviderAgent
   │    resolve placement → upsert one ProviderInstance per assigned
   │      definition (agent_id, definition_id), stopped
   │    issue per-agent secret → Redis im:ws:secret:{agent_id}
   │ ◀──────────────────────────────
     │  {agent_id, agent_secret, backends:[{instance_id,
     │   definition:{alias, modality, backend_config, config_fingerprint, ...}}, ...]}
   │ write provider_config.json to CACHE_DIR
   │ dial ws(s)://{admin}/provider/ws   (ONE socket for the agent)
   │   Authorization: Bearer {agent_secret}
   │ ──────────────────────────────▶  auth vs Redis (constant-time)
   │                                  accept → bump im:ws:epoch:{agent_id}
   │   ◀── provider.hello {epoch}    claim im:ws:owner:{agent_id}
   │                                  agent_status → running
   │  (proactive init warm-up, one backend at a time — see below)
   │   ── backend.status per backend as the scheduler boots them ──▶
```

- **Version hard fail:** agent versions must match the admin exactly
  (409). Admin and provider images are deployed together (§10).
- **Placement resolution:** the admin computes the agent's assigned
  definitions = enabled `ProviderDefinition`s of the agent's type whose
  `agent_placement` is `any_of_type`, plus those `specific`-placed on this
  agent (via `definition_agents`). Each becomes a `ProviderInstance`
  (backend) row, initially `stopped`.
- **Proactive init warm-up (one at a time):** after the socket is up, the
  scheduler walks the agent's assigned backends and boots them **one at a
  time** (serialized per agent), each gated by machine VRAM admission and
  the type's `max_running_backends`, until they no longer fit; the rest
  stay `stopped` and boot on demand later. The idle reaper still applies to
  warm backends that go idle.
- **Schema gate (Phase 12):** each agent ships a `schema.json` and sends
  it with the registration body. A fingerprint mismatch against the
  committed schema is refused **409 `schema_pending`** — the agent stays
  in a *waiting for consensus* state (visible in the UI as
  `waiting_schema` on the agent) and keeps retrying via the normal
  backoff until every known agent of the type has presented the new
  schema (or the operator force-commits).
- **Socket lifecycle:** if the agent WS dies, the admin marks the agent
  `disconnected` immediately (live path) and via the presence sweep
  (safety net for missed disconnects / admin restarts); all its backends
  become unschedulable while disconnected. The agent reconnects with
  exponential backoff (1s → 30s max). Each accepted reconnect gets a
  strictly greater **epoch**; the old socket is closed with code `4409`.
- **Decommissioning / renaming:** a renamed (`AGENT_ID` changed) or
  decommissioned container leaves a ghost `ProviderAgent` row (its backends may
  still read `running` while disconnected). Operators delete it from the Agents
  UI (`DELETE /admin/api/agents/{agent_id}`), which cascades the agent's
  `ProviderInstance` + `DefinitionAgent` rows and clears its Redis WS/metrics
  keys; the delete is **refused (409) while the agent is connected** — stop or
  redeploy the container first.

### Frame envelope (both directions)
```json
{ "v": 1, "type": "...", "id": "...", "reply_to": null,
  "epoch": 12, "ts": "ISO-8601 UTC", "payload": {} }
```
Commands carry an `id` and require an `ack` frame (`reply_to=<id>`,
payload `{ok, error, detail}`). The admin's
`ConnectionManager.send_command(agent_id, type, payload, timeout)` sends on
the agent's socket and awaits the matching ack (default 30s). **Per-backend
commands/events carry the target `instance_id` inside `payload`** (the
socket is agent-level; the backend is addressed by id). Every frame carries
the agent's epoch; stale-epoch frames are discarded by both sides.

### Event / command catalog
Provider agent → admin: `provider.status` (agent-level),
`backend.status` (per `instance_id`), `backend.boot_requested`,
`metrics.machine`, `backend.logs` (per `instance_id`), `provider.logs`
(agent-level), `download.progress`, `backend.metadata`, `ping`
(`metrics.inference` is defined but reserved — see §8).

Admin → provider agent: `provider.hello`, `agent.assignments.update`
(**Phase 16** — add/remove backends the agent should host), `backend.start`,
`backend.stop`, `backend.restart`, `provider.initialize`,
`provider.config.update`, `metrics.assign`, `metrics.unassign`,
`metrics.category.start`, `cache.clear`, `storage.prune_unused`,
`backend.logs.get` (Phase 13), `pong`. Per-backend commands
(`backend.*`, `provider.config.update`, `cache.clear`, `storage.prune_unused`)
carry `instance_id` in the payload.

`backend.start` acks only after the provider's lifecycle reaches
`running` **when it asks to wait** (`wait_for_running: true`, the
scheduler's default), so a successful return means the provider's `/v1` is
live. Manual control sends `wait_for_running: false`: the provider accepts,
boots in the background, heartbeats `backend.status initializing` (a cold
halogen-flash boot downloads its checkpoint and companions inside that
wait — tens of GB), and reports the terminal state as events.
`backend.restart` is the same with a drain check first (refuses while slots
are held), and `provider.initialize` re-runs the whole init lifecycle
(re-register → re-resolve placement → reboot → publish `backend.metadata`) and always
acks before the boot. `backend.stop` stops the backend plainly
(→ STOPPING → STOPPED): in-flight streams are not force-cancelled and there
is no in_use refusal on this command — their producer tasks release their
slots as the upstream closes (a client may see the stream end early). Real
drain semantics live in `provider.config.update` (below) and in
`backend.restart`, which refuse to stop while slots are held.

`provider.config.update` (Phase 9, live) carries the new
`backend_config` + `config_fingerprint` + scheduling fields. The
provider adopts a differing `capacity` in place first (no restart),
no-ops a same-fingerprint update (always safely ackable under load),
then drains via `stop_if_idle()` — the busy check and the STOPPING
transition are atomic under the lifecycle lock — clears the old
fingerprint's prompt cache, applies the config, restarts the backend,
and acks with the applied fingerprint plus discovered `model_metadata` —
the admin persists both. Admin retries only the drain-refused case (3
attempts, 10s apart, 300s per-instance timeout); other failures are
reported per-instance and are not fatal to the admin row. A stale
fingerprint on reconnect is healed automatically to the **stale
instance only** (connect-path background task + presence sweep, with a
per-instance in-flight push guard). Full semantics in
`docs/ws-protocol.md` §4 and `provider/README.md`.

`agent.assignments.update` (**Phase 16 slice 5**, live) is the agent-level
counterpart to `provider.config.update`: when a definition's placement changes
(created / PATCHed `agent_placement`/`agents`/`enabled`/`alias`), the admin
recomputes the agent's placed set, reconciles its `ProviderInstance` rows
(create stopped rows for newly-placed defs — the assignment carries only
alias/modality/config, the agent binds its own internal engine port;
retire de-placed rows **busy-safe** — a `running`/`in_use` row is left in place
and reported refused, pruned only once it stops), and pushes the **full**
current assignment set over the socket. Config edits to an already-assigned
definition still ride `provider.config.update` (addressed by `instance_id`).

Payload (admin → provider):

```json
{
  "assignments": [
    {"instance_id": "<uuid>", "provider_definition_id": "<uuid>",
     "alias": "my-model", "modality": "llm", "backend_config": { }, "config_fingerprint": "<sha256>",
     "capacity": 1, "idle_timeout_seconds": 300,
     "vram_required_bytes": 0}
  ],
  "max_running_backends": 0
}
```

Ack (provider → admin): `{"ok": true, "added": ["<instance_id>"], "removed":
["<instance_id>"], "refused": [{"instance_id": "<id>", "reason": "..."}]}`. The
provider reconciles its `BackendRegistry` to the pushed set — add a lifecycle
per new `instance_id` (via a package `make_handle` factory), and for each
dropped `instance_id` stop the engine if idle else refuse (busy-safe, mirroring
the admin) and keep it until it frees. A package that cannot host an additional
backend (every real engine today — N-per-process is slice 6) supplies no
factory and refuses the add rather than crashing. On a successful push that
added backends the admin re-triggers the scheduler's proactive warm-up (the
slice-4 seam) so the new backends boot within the VRAM + `max_running_backends`
budget. A disconnected agent still gets its rows reconciled (ghosts pruned) but
receives no frame — its next registration re-resolves authoritatively.

Unknown event/command types are logged and ignored (forward compatible).

---

## 6. Scheduler

`admin/backend/app/services/scheduler.py` — **in-process FIFO with a
Redis mirror**. The admin runs a single uvicorn worker, so the
authoritative queue is a per-alias waiter `deque` guarded by an
`asyncio.Lock` (serializes admission) and an `asyncio.Condition`
(wait/wake). The Redis keys (§9) are a best-effort observable mirror,
and the `im:sched:lock` contract is honored around boot/admission
mutations so a future cross-worker Redis-queue implementation can swap in
behind the same `acquire`/`release` interface.

### Boot budget

Booting is bounded by `BACKEND_BOOT_TIMEOUT_SECONDS` (default 1h), not the
30s command-ack default: `backend.start` from the scheduler sends
`wait_for_running: true` and the provider's ack means `/v1` is live, which
for a cold halogen-flash instance is behind an engine-side download of its
checkpoint and companions. A waiting client still gives up at its own
`queue_timeout` (→ 504) while the boot continues at the provider — the
instance heartbeats `backend.status initializing`, the DB mirror follows,
and the next request adopts the now-running backend (the `elif instance_id
not in self._booted` adoption path). The idle reaper only ever targets
`running`/`in_use`, so it never kills a downloading boot. Manual control
avoids the wait entirely: the admin routes accept (202) and poll.

### VRAM ledger — per booted instance

VRAM is held **per booted instance, not per request**. A loaded backend
keeps its weights resident between requests, so the old per-slot ledger
(`release` cleared the hold → an idle backend counted as 0) was
physically wrong and made eviction impossible (nothing to free). The
authoritative ledger is `self._booted: {instance_id: (machine_uid,
vram_bytes)}`, merged (max per instance) into `held_on(machine)` from
three sources so no hold is ever under-counted:

- `self._booted` (backends this process booted/adopted),
- the Redis `im:vram:used:{machine}` mirror (holds recorded before an
  admin restart), and
- connected instances whose DB `backend_status` is `running`/`in_use`,
  at their definition's `vram_required_bytes` (a running backend holds
  even when this process lost all trace of it).

`release` frees the **capacity slot only** — the booted hold persists
until the backend is actually stopped (idle reaper or eviction). The
Redis mirror is refreshed on boot/stop and its TTL kept alive by the
reaper while booted (see §9 / `redis-keys.md`). **Out-of-band stops
(Phase 6 review):** because only scheduler-initiated stops clear
`_booted` directly, `connection_manager` prunes the hold
(`scheduler.note_backend_stopped`) whenever a `backend.status` /
`provider.status` frame reports `stopped` or `error` — otherwise a UI
stop or crash would leave a stale hold that the reaper's TTL refresh
keeps alive forever and permanently over-counts the machine.
`stopping` deliberately does NOT prune (weights are still resident until
the stop completes; pruning early would let a concurrent boot
transiently oversubscribe).

`acquire(alias, request_id)` semantics:
1. Zero connected+enabled candidates → `NoProviderAvailable` (503)
   immediately — never queue what can never run.
2. FIFO fairness: the request is appended to the alias waiter deque and
   only admitted at the head when a slot is available.
3. A slot = `active_on_instance < definition.capacity`. An
   **already-booted** target needs no new VRAM and admits on capacity
   alone. A target that **needs a boot** is gated on free VRAM =
   `Machine.total_vram_bytes − held_on(machine)` with the target's own
   hold excluded.
4. **Eviction (Phase 6).** If a boot doesn't fit, idle **different-alias**
   backends on the same machine are stopped (`backend.stop`) **LRU-first**
   — oldest `last_request_at` (null = oldest) — until room is made.
   Victims must have **zero active in-process slots** (never evict busy)
   and a **positive VRAM hold** (evicting something that frees nothing
   can't help). The DB `backend_status` is *not* a victim filter: any
   instance with a positive hold in the merged ledger is evictable,
   including `_booted` instances whose DB mirror is stale. The
   requesting alias's own instances are never victims. A refused stop
   (NAK, e.g. a drain race) moves to the next candidate; if nothing
   works the request stays queued (FIFO/timeout) — busy backends are
   never force-killed. The whole evict+boot runs under one
   `im:sched:lock:{alias}` acquisition plus an in-process `_evicting`
   set, so two concurrent acquires (even different aliases sharing the
   machine) can't both evict the same victim. **Admit guard (Phase 6
   review):** because the evictor holds the *requesting* alias's lock —
   never the victim's — `_try_admit` skips any candidate instance
   present in `_evicting` on both the already-booted and needs-boot
   paths: a request is never admitted onto a backend with an in-flight
   stop (eviction or idle reaper). Evictions log at WARNING.
5. A `stopped` candidate is booted with `backend.start` under the same
   admission lock; boot failure/timeout falls through to the next
   candidate. A successful boot is remembered in `_booted` so a lagging
   DB mirror never causes a redundant boot.
6. Waiting is bounded by `queue_timeout` (default 300s) → `QueueTimeout`
   (504).
7. Admission records `im:sched:active` and returns an
   `Admission{instance_id, base_url, machine_uid}`, where `base_url` is the
   **agent's** env-port base (`http://{machine.reachable_address()}:{agent.base_port}`)
   — litellm then routes to the specific backend by the `model` (= alias).

`release(alias, request_id)` is idempotent and cancellation-safe
(`asyncio.shield` around cleanup so a client-disconnect-mid-stream still
releases and wakes the next waiter). It does **not** clear the booted
instance's VRAM hold.

### Agent-scoped admission & `max_running` (Phase 16)

Candidates for an alias are the `ProviderInstance` backends whose
definition is placed on a **connected** agent (per `agent_placement`:
`any_of_type` or `specific`), still filtered by enabled + authored config.
The VRAM ledger stays **per booted backend** (unchanged) and the admin
remains the sole VRAM authority — an agent never boots a backend the admin
hasn't admitted.

Two new constraints layer on top of the existing per-machine VRAM gate:

1. **`max_running_backends` (per agent).** Before booting a backend of type
   `T` on agent `A`, the scheduler counts `A`'s backends of type `T`
   already `running`/`in_use`. If the count is at `ProviderType.max_running_backends`
   (e.g. `1` for halogen-flash), it **hot-swaps**: evict the LRU running
   backend of that type on that agent (subject to the same busy/zero-active-
   slot guard as VRAM eviction) before admitting the new one. A singleton
   type therefore never has two backends resident on one agent at once.
2. **Serialized init warm-up.** On agent (re)connect the scheduler enqueues
   the agent's assigned backends and boots them **one at a time** under the
   `im:sched:lock` + a per-agent `_warming` guard (each boot also takes the
   per-alias in-process `state.lock`, exactly as the request path does, so a
   warm-up boot and a concurrent request for the same alias can never
   double-boot), each gated by VRAM + `max_running`, stopping at the first
   that doesn't fit. This is proactive (not request-driven) but bounded; the
   remaining backends boot on demand exactly as before.

### Idle-timeout reaper (Phase 6)

`_idle_reaper` runs every `IDLE_REAPER_INTERVAL_SECONDS` (default 15.0,
injectable via the `InferenceScheduler` constructor for tests), started
by `start_background` and stopped by `stop`. Each tick:

- Refreshes the `im:vram:used` TTL for every hold this process owns (so
  a long boot outliving the 60s key TTL keeps its mirror alive).
- Stops every **connected** instance whose `backend_status` is
  `running`/`in_use`, which has **zero active in-process slots**, and
  whose idle clock — `max(last_request_at, backend_loaded_at)` (a boot
  re-arms the window at load time), falling back to `created_at` when
  both are null — is at least the definition's
  `idle_timeout_seconds` old. `idle_timeout_seconds == 0` means **never
  idle-stop**.
- On a successful stop the VRAM hold is cleared (ledger + mirror + DB
  `backend_status=stopped`). A NAK/exception is logged and simply
  retried next tick. The whole tick is wrapped in try/except so one bad
  pass never kills the task.

---

## 7. Inference Flow (request path)

Example: streamed Responses API against `https://matrix.thelink.family`.

```
1. Client POST /v1/responses {model: alias, input: [...],
     previous_response_id: "resp_abc", stream: true}
2. Admin resolves the enabled ProviderDefinition by alias (else 404),
   mints client_response_id = "resp_" + uuid4().hex.
3. Admin builds the litellm input: on previous_response_id, load the
   prior ResponseRecord and prepend its input_items + output_items to the
   new input. The admin owns the chain; previous_response_id is NOT
   passed to litellm.
4. ensure_registered(alias): register the alias with litellm
   (supports_native_streaming=True, mode="chat",
   litellm_provider="openai"). REQUIRED — without it litellm "fake
   streams" and fails with a confusing APIError (see FINDINGS).
   mode="chat" (not "responses") is deliberate: acompletion bridges to
   /v1/responses when mode=="responses" (responses_api_bridge_check),
   while aresponses keys native streaming only on
   supports_native_streaming and ignores mode — so one registration
   serves both public APIs (see §7 chat specifics).
5. scheduler.acquire(alias, request_id) → Admission with base_url.
   Transport-specific error contract (Phase 6 keepalive):
   - **Non-stream:** `NoProviderAvailable` → 503, `QueueTimeout` → 504
     (JSON; the stream never starts).
   - **Stream:** the SSE response starts immediately and emits
     `: keep-alive` comment lines every `KEEPALIVE_INTERVAL_SECONDS`
     (default 10s) while `acquire` blocks on a cold boot (download +
     load) or the FIFO wait, so proxies (Traefik) don't drop the
     connection before any event. The cheap zero-candidates check stays
     pre-stream (HTTP 503 — never open a stream that can't respond),
     but scheduler errors that arrive *inside* the stream surface as the
     spec `response.failed` + `error` frames + `[DONE]` (the admin
     synthesizes terminal frames — FINDINGS §2) and a failed
     `ResponseRecord`; no HTTP status is possible once bytes are sent.
   Fast pre-stream checks are HTTP for both: unknown/disabled alias →
   404, missing `model`/`input` → 400.
 6. litellm.aresponses(model=alias,
      api_base=f"{admission.base_url}/v1", custom_llm_provider="openai",
      stream=True, tools=[client tools + platform local tools], input=...)
 7. The agent routes the request (by `model`) to the target backend, whose
    translation layer normalizes its output to spec on the agent's single
    env-port `/v1` surface; the slot is held for the inbound connection
    lifetime; backend.status → in_use.
8. Admin's SSEEmitter re-frames litellm events for the client:
   - replaces response.id with the admin-owned resp_<uuid> on every
     lifecycle frame (created/in_progress/completed/failed/incomplete),
   - reassigns sequence_number monotonically (0,1,2,...),
   - passes everything else (usage, output[], non-canonical events like
     response.reasoning_text.delta) through untouched.
9. litellm exceptions (MidStreamFallbackError etc.) → SSEEmitter.failed()
   synthesizes the spec response.failed + error frames. This covers a
   **call-time** `aresponses` raise too (connection refused, APIError
   before the first event — Phase 6 review): the awaited call is inside
   the same try as the `async for`, so the stream always terminates with
   response.failed + error + [DONE] instead of truncating after the
   keepalives. A litellm transport error is not a `SchedulerError` and
   never reaches the admission-error branch.
10. finally (cancellation-safe): close upstream stream, scheduler.release,
    persist ResponseRecord + TokenUsageSample once.
```

Cancellation: client disconnect → FastAPI cancels the emitter task →
litellm stream closed → provider sees TCP close → slot freed. No separate
lease-renewal machinery. On the `/v1/responses` stream the cancel can
also arrive during the cold-boot keepalive phase (before any `Admission`
exists); the generator's shielded `finally` cancels the pending
`acquire` (its own shielded `_drop_waiter` removes the waiter) before
releasing, so no slot or waiter is orphaned. The keepalive + in-stream
error contract is specific to `/v1/responses`; the Phase 7
`/v1/chat/completions` route keeps the pre-stream 503/504 acquire.

### Public endpoint scope (this overhaul)
| Endpoint | Status |
| --- | --- |
| `/v1/responses` | **Full** (stream + non-stream) — Phase 6 |
| `/v1/chat/completions` | **Full** (stream + non-stream) — Phase 7, via `litellm.acompletion`; admin-owned `chatcmpl-<uuid>` id on every chunk, data-only SSE, persisted as `ResponseRecord` with `parameters.api_format="chat_completions"` |
| `/v1/models` | **Implemented** — Phase 7; derived from enabled `ProviderDefinition`s (alias asc, `owned_by`=provider_type, `model_metadata` merged). **Phase 18:** lists `llm` **and** `embedding` definitions; each object carries a `modality` marker so clients can filter. |
| `/v1/embeddings` | **Implemented** — Phase 18; non-streaming only, via `litellm.aembedding` against the same scheduler admission + the agent's env-port `/v1` surface. Accepts only `modality=embedding` aliases (an `llm` alias → 404). Returns the spec `CreateEmbeddingResponse`; persists a `TokenUsageSample` (prompt tokens) only — no `ResponseRecord` (embeddings are not conversation turns). |
| `/v1/completions` (legacy), `/v1/rerank`, `/v1/moderations`, `/v1/decisions`, `/v1/audio/*`, `/v1/files`, `/v1/batches` | **501 stubs** — accepted regressions (§12). `/v1/audio/*` is the reserved landing spot for the future `audio` modality (§4). |
| Responses-over-WebSocket transport | **Dropped** (not part of the OpenResponses spec) |
| Benchmarks | **Dropped entirely** |

### Chat completions specifics (Phase 7)

Same flow as §7 steps 5–10 with `aresponses` → `acompletion`, plus:

- **Alias registration is shared** (`ensure_registered`): the litellm
  model entry is `mode: "chat"` + `supports_native_streaming: True`.
  `aresponses` keys native streaming off `supports_native_streaming`
  only (never `mode`), while `acompletion` **bridges to `/v1/responses`
  whenever `mode == "responses"`** — so the Phase 6 `mode: "responses"`
  registration would hijack chat calls. One `mode: "chat"` registration
  serves both APIs natively; the chat route also passes
  `_skip_responses_api_bridge=True` as a guard against litellm's gpt-5
  conditional bridge.
- **Id ownership**: admin mints `chatcmpl-<uuid>` and replaces `id` on
  every streamed chunk / the non-stream response (litellm passes the raw
  upstream chat id through un-wrapped; it is captured in
  `parameters.litellm_id`, never surfaced).
- **Usage**: the admin forces `stream_options.include_usage` on the
  streaming path so token counts persist.
- **Persistence**: reuses the Phase 6 `persist_turn` — request
  `messages` → `input_items`; aggregated assistant message(s) (content
  + merged tool-call deltas) → `output_items`; client disconnect before
  the terminal chunk persists `status="failed"` with
  `error.code="client_disconnected"`.
- **Error framing**: pre-stream errors → HTTP (400/404/503/504, or 502
  JSON for upstream failures on the non-stream path); mid-stream errors →
  chat-style `data: {"error": {...}}` + `data: [DONE]` (no
  `response.failed` — that's the responses spec).
- **Provider side**: the agent's env-port `/v1/chat/completions` surface now
   honors `stream=false` by aggregating the driver's chunk stream into a
  `chat.completion` JSON (same fix class as Phase 6's responses
  non-stream path).

### Embeddings specifics (Phase 18)

`POST /v1/embeddings` is the first **non-chat modality** endpoint. It reuses
the Phase 6/7 admission + litellm discipline but is **non-streaming only**
(the OpenAI embeddings response is a single JSON object):

1. Client `POST /v1/embeddings {model: alias, input: "text" | ["t1","t2"], dimensions?, user?}`.
2. Admin resolves the enabled `ProviderDefinition` by alias; **404 if the
   definition's `modality != embedding`** (an `llm` alias cannot embed).
   Missing/empty `input` → 400.
3. `ensure_registered(alias, mode="embedding")` — embeddings are a distinct
   litellm `mode`; the alias registry keys its process cache by alias (an
   alias is exactly one modality). litellm's `aembedding` for
   `custom_llm_provider="openai"` posts to `{api_base}/embeddings` (verified
   against litellm 1.103.2: `make_openai_embedding_request` →
   `client.post("/embeddings", …)`), so the backend must serve that path.
4. `scheduler.acquire(alias, request_id)` → `Admission` (identical FIFO +
   VRAM + `max_running` path as chat; the embedding backend is just another
   booted instance).
5. `litellm.aembedding(model=alias, custom_llm_provider="openai",
   api_key="unused", api_base=f"{admission.base_url}/v1", input=…, dimensions=…)`
   → spec `CreateEmbeddingResponse`.
6. Persist a `TokenUsageSample` from `usage.prompt_tokens` (embedding
   telemetry only — **no `ResponseRecord`**, embeddings are not turns).
7. `finally` (cancellation-safe, shielded): `scheduler.release`.
8. Error contract mirrors the chat **non-stream** path (no SSE to frame):
   pre-acquire 400/404, scheduler 503/504, upstream failure → 502 JSON with
   the OpenAI error envelope.

**Provider side**: `provider_lib` gains an optional `BackendDriver.embeddings()`
(default raises `NotImplementedError` → 501) and a slot-admitted
`POST /v1/embeddings` route on the agent's env-port `/v1` surface (routed to
the backend by model). llama-cpp boots
`llama-server --embedding` (auto from the pushed `modality`) and proxies the
upstream `/v1/embeddings`; `--pooling` is a schema option. The mock serves
deterministic fake vectors so the whole path runs with no GPU.

---

## 8. Metrics

### Machine metrics (split by category)
Multiple provider agents share one machine; categories split into two
ownership classes (Phase 17):

* **GPU categories** (`vram` / `gpu_usage`) are per-device. On a
  device-isolated box each container sees only its own GPUs, so **every**
  agent that declares them emits them, filtered to its assigned GPUs
  (`ASSIGNED_GPU_UUIDS`; empty = implicit visible==owned). The admin stores
  each frame as a per-agent partial `im:metrics:machine:{machine_uid}:agent:{agent_id}`
  (TTL 30s) and **merges them per-GPU-uuid on read** (`read_machine_metrics`,
  served by `GET /admin/api/machines/{id}/metrics`); a dead agent's partial
  self-expires with its TTL.
* **Machine-wide categories** (`os_ram` / `cpu` / `storage`) are visible from
  any container and stay single-owner. `metrics_service.py` assigns ownership
  via Redis `im:metrics:owner:{machine_uid}` (SET NX, TTL 30s, refreshed on
  each owner `metrics.machine` receipt) — but only when the agent declares a
  machine-wide category (a GPU-only agent never claims the lease). The owner's
  snapshot is stored at `im:metrics:machine:{machine_uid}`; non-owner
  machine-wide sections are dropped. On connect the admin assigns the lease
  (subject to `im:metrics:cats:{agent_id}`); on expiry/disconnect another agent
  can take over. Epoch fencing makes failover safe against half-open sockets.

### Inference metrics (never deduped)
The `metrics.inference` frame kind is **defined and reserved** (available
slots, max slots, token speed, prompt-processing speed, in-flight counts)
but **not yet emitted** by any provider and not yet handled by the admin —
it lands with Phase 8 (see the Reserved row in `docs/ws-protocol.md` §4).
When it lands it is always-on per instance and never deduped.

### Categories
`gpu_usage`, `vram`, `os_ram` (OS usage only — APUs share RAM with VRAM),
`cpu`, `storage` (cache + model dirs). Collectors in
`provider_lib/metrics.py` (NVML / rocm-sysfs / psutil).

### Log streams (Phase 13)
`backend.logs` / `provider.logs` carry captured lines from the backend
subprocess (stdout/stderr, tagged per stream) and from the provider's
own logger. The provider buffers in a bounded ring
(`provider_lib.log_ring.CursorLogRing`, `LOG_RING_LINES` default 2000),
flushes throttled batches (~1s / 100 lines per frame) while connected,
and answers `backend.logs.get` with the current buffer + `dropped`
counter for catch-up after a reconnect. `install_log_streaming` gives
the backend and provider rings a **shared `SeqCounter`** so
`kind=all` resumes on one cursor. The admin stores **Redis-only**
capped lists (`im:logs:backend:{instance_id}`, ~2000 lines / TTL 1h)
with a per-entry **ingest seq** (`im:logs:seq:{id}`, Redis `INCRBY`,
1-based) — logs are ephemeral ops telemetry, never persisted to
Postgres — and serves them via `GET /admin/api/instances/{id}/logs`.
The provider ring seq and the admin ingest seq are **unrelated spaces**
(see `docs/ws-protocol.md` §4); never mix their cursors. The read
response carries `gap`/`oldest_seq`/`unseen_total` so the UI can warn
about dropped or skipped lines rather than silently losing them.

---

## 9. Redis Key Layout

Postgres is the source of truth for configuration and stored responses.
Redis holds fast-changing runtime state that can be rebuilt from Postgres +
a fresh provider registration. All keys namespaced `im:`. Full table in
`admin/backend/docs/redis-keys.md`.

| Key | Type | TTL | Purpose |
| --- | --- | --- | --- |
| `im:ws:secret:{agent_id}` | String | 30d | Per-**agent** WS secret (trusted LAN, plaintext; not in Postgres) |
| `im:ws:epoch:{agent_id}` | Counter | none | Monotonic connection epoch (INCR per accepted socket) |
| `im:ws:owner:{agent_id}` | String | none | Connection token of the currently accepted socket |
| `im:ws:presence:{agent_id}` | String | 60s | Liveness marker; absence ⇒ sweep marks the agent disconnected |
| `im:metrics:owner:{machine_uid}` | String | 30s | Which **agent** emits the **machine-wide** metrics (`os_ram`/`cpu`/`storage`) for the machine (GPU categories are not gated by this lease) |
| `im:metrics:machine:{machine_uid}` | String | 30s | Latest **machine-wide** owner snapshot JSON |
| `im:metrics:machine:{machine_uid}:agent:{agent_id}` | String | 30s | Per-agent **GPU** partial JSON (`vram`/`gpu_usage`), written by every agent that reports them; merged per-GPU-uuid on read |
| `im:metrics:cats:{agent_id}` | String | — | Agent's declared metrics categories (JSON list) |
| `im:sched:queue:{alias}` | List | — | Queued request ids (mirror of in-process deque) |
| `im:sched:wait:{req_id}` | Hash | 1h | position / enqueued_at / status |
| `im:sched:active:{alias}` | Set | — | Admitted request ids (cardinality ≤ capacity) |
| `im:sched:lock:{alias}` | String | 5s (SET NX PX) | Admission lock contract |
| `im:vram:used:{machine_uid}` | Hash | 60s | `{instance_id}` → bytes held **per booted instance** (§6); refreshed on boot/stop, TTL kept alive by the reaper |
| `im:logs:backend:{instance_id}` | List | 1h | Backend stdout/stderr tail (Phase 13; capped ~2000 lines, newest left) — **per backend** |
| `im:logs:provider:{agent_id}` | List | 1h | Agent's own log tail (Phase 13; same cap) — **per agent** |

Schema-consensus state (pending schema, fingerprint, voters) lives in the
`provider_types` Postgres row — not Redis — because the admin is a
single writer and the state must survive a Redis flush (§9 preamble).

---

## 10. Deployment

| Container | Contents | Notes |
| --- | --- | --- |
| `admin` | built SPA + FastAPI, port 8000 behind Traefik at `matrix.thelink.family` | **single uvicorn worker** (in-process scheduler authority) |
| `postgres` 16 | schema via squashed migration at prestart | |
| `redis` 7 | scheduler/WS/metrics runtime state | |
| provider images | one per provider type from `provider/<type>/Dockerfile` | run on hardware hosts as **agents** (one container = one machine+type, 1..N backends); mock for local dev |

**Deploy order (mandatory, due to version hard-fail):** admin first, then
all provider agents. A version-mismatched agent refuses to start with a
clear 409. Accepted constraint of the solo-operator model.

---

## 11. Mock Provider

`provider/mock` implements the full provider contract with no hardware:
- Registers with a fake GPU inventory.
- `backend.start` transitions `initializing → starting → running` almost
  instantly.
- Serves fake but spec-compliant `/v1/responses` + `/v1/chat/completions`
  with streaming deltas, tool-call events, and usage.
- Emits all metric categories with synthetic values.

Purpose: the entire real path (registration → WS → scheduler → VRAM
admission → boot handshake → litellm → SSE → persistence) runs locally
with `docker compose up` and no GPU. This is the base for integration
tests and UI development.

---

## 12. Security Model — Trusted LAN (Known Limitation)

**The system assumes a trusted network.** Consciously chosen for solo
operation:

- `/admin/api/*` and `/v1/*` are **unauthenticated**. Do not expose them
  beyond the LAN / a trusted reverse proxy without adding auth first.
- The **only** secrets are the per-**machine** `registration_secret`
  (plaintext in agent env — possession proves the container belongs to that
  machine; **replaces the per-definition `registration_token`**) and the
  per-**agent** WS `agent_secret` issued at registration (Redis).
- Secret rotation is manual: edit the machine's secret, redeploy the agents
  on it. No revocation short of deleting the machine/agent.
- Threat accepted: anyone on the network can call inference and the admin
  API; anyone with a machine's `registration_secret` + `MACHINE_UID` can
  register a rogue agent (and thus backends) on that machine.

---

## 13. Known Limitations & Accepted Regressions

1. Trusted-LAN only (§12).
2. Version hard-fail requires coordinated admin+provider deploys (§10).
3. Legacy completions, rerank, moderations, decisions, audio,
   files, batches: 501 stubs (§7). **Embeddings landed in Phase 18**
   (`/v1/embeddings`, `modality=embedding`). **Do not "restore" old
   implementations from git history** — re-implement against the new
   provider/scheduler model when needed.
4. No prompt-cache tracking table; cache is provider-local via
   fingerprint.
5. Multi-file model artifacts are JSON (`backend_config`), not relational
   — no cross-definition artifact dedup.
6. `MACHINE_UID` misuse (same UID on two physical hosts) poisons VRAM
   admission; there is no host-fingerprint check.
7. **Schema consensus (Phase 12)** requires every known
   `ProviderAgent` row of a type to present the new schema before it
   commits (Phase 16: voters are agents). A machine that is permanently gone
   (agent row never deleted) blocks consensus forever unless the operator
   **force-commits** the pending schema. Stale agent rows should be deleted
   when hardware is decommissioned.
 12. **Phase 16 / port-model overhaul — agent fan-out & ports.** Multiple
    agents may share a `(machine, provider_type)`; they are discriminated only
    by the operator-supplied `AGENT_ID`. Each agent publishes exactly **one**
    admin-facing `/v1` port (its env `PROVIDER_PORT`, recorded as `base_port`)
    and routes to its backends internally by model; engine ports are private to
    the container and OS-assigned by default. The admin no longer allocates or
    polices ports (no `port_conflict` rejection). Ensuring two agents on one
    machine publish distinct `PROVIDER_PORT`s (bridge networking) is a manual,
    per-deployment responsibility.
8. Scheduler VRAM eviction is implemented (§6): idle different-alias
   backends are stopped LRU-first to make room. Remaining nuance:
   eviction is per-machine (no cross-machine rebalancing) and victims are
   chosen purely by LRU idle time — there is no priority/weight scheme,
   so a just-booted large model can be evicted for a newly arriving
   request if it happens to be the oldest idle hold.
9. The idle-timeout reaper is implemented (§6). Remaining nuance: idle
   detection is admin-side from the idle clock
   (`max(last_request_at, backend_loaded_at)`) + in-process slot
    counts, so a backend kept busy by traffic that bypasses the admin
    (direct agent env-port calls) is invisible to the reaper.
10. Benchmarks removed; performance testing is out-of-band.
11. Config-update retry is bounded (3 attempts for drain-refused only);
    a provider that stays busy past the retries surfaces the failure in
    the PATCH response and relies on the reconnect/sweep self-heal for
    eventual consistency. The admin does not queue config pushes.
13. **Overlapping GPU visibility (Phase 17).** Two agents that both see the
    same GPU uuid (e.g. both `--gpus all`) with no `ASSIGNED_GPU_UUIDS`
    resolve by per-uuid last-writer-wins — values stay correct (same physical
    GPU) but attribution may flap between reporters. The explicit
    `ASSIGNED_GPU_UUIDS` env disambiguates which agent owns which GPU.

---

## 14. Testing & Conformance

- **Exit criterion for every inference-path phase:** the openresponses.org
  conformance (Zod) suite passes against `/v1/responses` (see
  `docs/integration-testing.md`).
- Phase 0 fidelity spike results in `spike/litellm-fidelity/FINDINGS.md`
  are the design basis for the §7 emitter: native streaming preserves the
  full event set (dual-name reasoning, `sequence_number`, populated
  `output[]`, usage); the admin owns the emitter and the `resp_` id;
  litellm exceptions must be mapped to spec terminal frames.
- Admin services have unit tests (`admin/backend/tests/`: scheduler, SSE
  emitter, v1 responses, connection manager, presence sweep, metrics).
- Provider lib + each provider package have their own tests.
- Mock provider integration tests cover registration, boot, streaming,
  failover (kill socket → instant `disconnected` → metrics reassignment).

---

## 15. References

- `docs/ws-protocol.md` — canonical Admin ⇄ Provider wire protocol RFC
- `admin/backend/docs/redis-keys.md` — Redis key reference
- `provider/README.md` — provider authoring guide
- `spike/litellm-fidelity/FINDINGS.md` — litellm streaming fidelity decision
- `IMPLEMENTATION_STATUS.md` — per-phase feature status
- `AGENTS.md` — commands, workflow, high-risk gotchas
