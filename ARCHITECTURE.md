# Inference Matrix — Architecture

Inference Matrix is an OpenAI-compatible inference broker that schedules GGUF
model inference across one or more hardware machines. The admin container is
the brain (database, scheduler, client API); provider instances are
hardware-local containers that each own exactly one inference backend
process.

**Status: this document describes the litellm-based architecture on the
`litellm-architecture-overhaul` branch.** It supersedes the legacy
broker/agent design; the pre-overhaul code was removed in the final
cleanup and lives in git history only.

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
                            │  /provider/ws  (provider dials in)
        ┌───────────────────┼───────────────────────┐
        ▼                   ▼                       ▼
  ┌───────────┐       ┌───────────┐          ┌───────────┐
  │ MACHINE A │       │ MACHINE B │   ...    │ MACHINE N │
  │ provider  │       │ provider  │          │ provider  │
  │ instance  │       │ instance  │          │ instance  │
  │  ┌─────┐  │       │  ┌─────┐  │          │  ┌─────┐  │
  │  │backend│ │       │  │backend│ │          │  │backend│ │
  │  └─────┘  │       │  └─────┘  │          │  └─────┘  │
  │  :8081 ▲──┼───────┼──▲ :8081 │          │           │
  └────────┼──┘       └──┼───────┘          └───────────┘
           │  litellm targets http://<machine>:<port>/v1 │
           └─────────────────────────────────────────────┘
```

### Components

| Component | Role |
| --- | --- |
| **Admin** (`admin/backend`) | Owns Postgres + Redis, the scheduler, conversation state, the client-facing `/v1` API, the admin UI/API, and the provider WebSocket registry. Runs **litellm** to drive inference. Knows **nothing** provider-specific. |
| **Provider instance** (`provider/<type>`) | Hardware-local container. Owns exactly **one** backend subprocess, its lifecycle, metrics, logs, and downloads. Serves a fully OpenAI/OpenResponses-spec-compliant HTTP API on `PROVIDER_PORT` and dials the admin over a WebSocket. |
| **Backend** | The provider's inference software (llama-server, halogen, halogen-flash, gufo). A private detail of the provider instance; never seen by the admin. |
| **Machine** | A physical/virtual host with a unique `uid`, pre-registered in the admin UI, with VRAM capacity tracked for scheduler admission. |
| **Redis** | Scheduler queues/mirrors, VRAM ledger, WS presence/secrets/epochs, metrics-ownership leases. |
| **Postgres** | Source of truth for configuration (machines, provider definitions, instances) and stored responses. |

### Key invariants

1. A provider instance has **exactly one** backend.
2. The provider instance is in full control of the backend lifecycle and
   inference-slot admission. Slot capacity is enforced **at the provider**,
   tied to the inbound connection lifecycle — a dead admin/litellm
   connection cannot leak a slot.
3. The admin never proxies raw provider-quirk traffic. Everything between
   admin and provider is either (a) the WS control protocol or (b)
   standard OpenAI-compatible HTTP driven by litellm against the
   provider's spec-compliant port.
4. The client-facing `/v1/responses` SSE stream must pass the
   openresponses.org conformance suite. That client boundary is the
   fidelity contract; internal layers are free as long as it holds.
5. The admin owns the client-facing `resp_<uuid>` id and the conversation
   chain. litellm-side session state is never relied upon.

---

## 2. Repository Layout

```
admin/
  backend/                FastAPI admin + public inference API (package: matrix-admin)
    app/
      main.py             App factory: docs at "/", /openapi.json, routers, lifespan
      models.py           SQLModel tables (squashed schema, see §4)
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
      admin_client.py     Registration, WS dial, bearer auth, reconnect/backoff
      wire.py             Canonical wire envelope (Frame, FrameKind, Ack)
      backend.py          BackendDriver ABC + BackendLifecycle (slot mgmt)
      app_factory.py      FastAPI app serving the provider port + /v1 surface
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
| `/admin/api/providers/register` | admin | registration token | Provider registration (§5) |
| `/admin/api/machines*` | admin | none (trusted LAN) | Machine CRUD (Phase 9; delete refused while instances attached) |
| `/admin/api/definitions*` | admin | none (trusted LAN) | ProviderDefinition CRUD (Phase 9; `backend_config`/`capacity` PATCH pushes `provider.config.update` to connected instances; explicit null on required fields → 422; `provider_type` change refused 409 while instances attached; Phase 12: `backend_config` validated against the committed JSON Schema of its `ProviderType` → 422 with per-field errors) |
| `/admin/api/provider-types` `/admin/api/provider-types/{name}` | admin | none (trusted LAN) | ProviderType registry reads (Phase 12; UI renders the backend-config form from the committed schema) |
| `/admin/api/provider-types/{name}/pending/commit` `.../pending/dismiss` | admin | none (trusted LAN) | Operator override of the schema-consensus state (Phase 12) |
| `/admin/api/huggingface/search` `/admin/api/huggingface/files` | admin | none (trusted LAN) | HF proxy for the `hf-file` picker widget (Phase 12; ported from legacy `huggingface.py` routes) |
| `/admin/api/instances/{id}/cache/clear` `/admin/api/instances/{id}/storage/prune` | admin | none (trusted LAN) | Instance storage actions (Phase 9) |
| `/admin/api/instances` `/admin/api/instances/{id}` | admin | none (trusted LAN) | Provider-instance reads for the UI (Phase 10) |
| `/admin/api/instances/{id}/logs` | admin | none (trusted LAN) | Backend/provider log tail (Phase 13; Redis-backed, `kind`/`since`/`limit` cursor) |
| `/admin/api/responses` `/admin/api/stats/usage` `/admin/api/stats/overview` | admin | none (trusted LAN) | Response log + usage/dashboard reads (Phase 10; overview reads the scheduler Redis mirror, observability only) |
| `/v1/*` | admin | none (trusted LAN) | Public OpenAI-compatible inference |
| `/provider/ws` | admin | bearer (instance secret) | Provider instances dial in (§5) |

The provider instance serves its own OpenAI-compatible HTTP API on
`PROVIDER_PORT` (default **8081**), reachable from the admin at
`http://{machine.reachable_address()}:{port}/v1/...`. `Machine` exposes
`dns`/`host`/`ip`; `reachable_address()` prefers `dns or host or ip`.

---

## 4. Data Model

All tables are created by **one squashed Alembic initial migration**
(`admin/backend/alembic/versions/0140d6ad9f48_initial_squashed_schema.py`),
applied by `scripts/prestart.sh` and by CI; Phase 12 adds the
`provider_types` table (plus the two instance columns) in a follow-up
migration. The provider instance itself has **no database** — it derives
everything from env + the registration response, mirrored here by the
admin.

### Machine
A host that provider instances run on. **Created in the admin UI before
any provider registers against its `uid`.**

| Field | Notes |
| --- | --- |
| `uid` | Stable identifier supplied by the operator, referenced by the provider's `MACHINE_UID` env. Unique. |
| `name` | Display name. Unique. |
| `host` / `dns` / `ip` | How the admin reaches instances on this machine. |
| `total_vram_bytes` | Admission budget; merged/refreshed from provider-reported hardware. |
| `hardware` | JSON inventory: `{"gpus": [{"uuid","vendor","name","total_vram_bytes"}, ...], "cpu": {...}, "ram": {...}}`. Union of reports from all instances on the machine. |

A provider registering with an unknown `MACHINE_UID` is **rejected (404)**.
UIDs must not be reused across physical hosts (see Known Limitations).

### ProviderType
A registered provider **type** (Phase 12), created by the first
registration of that type. Owns the JSON Schema (2020-12) that describes
the type's `backend_config`; every `ProviderDefinition.provider_type`
must reference a registered type.

| Field | Notes |
| --- | --- |
| `name` | Type id (`llama-cpp`, `halogen`, `halogen-flash`, `gufo`, `mock`, …). Unique. |
| `schema` | Committed JSON Schema (2020-12) for this type's `backend_config`. Drives admin validation on write **and** the UI form render. |
| `schema_fingerprint` | SHA-256 of canonical `schema` (same canonicalization as `config_fingerprint`). |
| `pending_schema` / `pending_fingerprint` | Staged schema awaiting consensus (null when none). |
| `pending_voters` | JSON list of instance ids that have registered presenting `pending_fingerprint`. |
| `status` | `active` \| `consensus_pending` \| `conflict`. |

**Schema consensus** (all known instances of the type must agree):
registration with the committed fingerprint proceeds normally. A
different fingerprint stages it as pending and the registration is
**refused 409 `schema_pending`** — the agent keeps retrying (waiting
for all agents to update). Repeat registrations with the pending
fingerprint add voters; when voters cover **every `ProviderInstance`
row of that type**, the pending schema is committed. A third distinct
fingerprint while pending → 409 `schema_conflict`. Operator override:
force-commit (e.g. a permanently dead machine can never vote) or
dismiss. Force-commit applies to **future registrations and new/edited
definitions only** — connected old-schema instances are not force-
converged. Full algorithm in `docs/ws-protocol.md` §2.

The `PROVIDER_TYPES` constant is gone; the registry is the source of
truth. The static `schema.json` lives in each provider package
(`provider/<type>/provider_<type>/schema.json`) and is shipped by the
agent at registration — the admin never holds provider code.

### ProviderDefinition
A client-facing model: how to boot a backend and how to schedule it.

| Field | Notes |
| --- | --- |
| `alias` | Public model name clients use in `/v1/models` and the `model` field. Unique. |
| `provider_type` | Must reference a registered `ProviderType` (Phase 12). Known types after rollout: `llama-cpp`, `halogen`, `halogen-flash`, `gufo`, `mock` — but the registry, not a constant, is the source of truth. |
| `backend_config` | JSON handed to the provider to start its backend: model artifacts (main GGUF + mmproj + draft, each with source), engine args, engine options. **Validated against the committed JSON Schema of its `ProviderType` on create/PATCH (Phase 12)**; the UI form is rendered from that schema (collapseable sections, `hf-file` artifact widget). |
| `vram_required_bytes` | Scheduler admission hint. |
| `idle_timeout_seconds` | Admin-driven idle stop (reaper — Phase 6 TODO). |
| `capacity` | Concurrent backend slots (≥1). |
| `registration_token` | Secret presented at registration; binds instance→definition and cross-checked against the container's provider type. Unique. |
| `model_metadata` | OpenAI model metadata discovered at init. |
| `enabled` | Disabled definitions are excluded from scheduling. |

### ProviderInstance
One backend on one Machine for one ProviderDefinition. Unique on
`(machine_id, provider_definition_id)` — a machine cannot run two backends
of the same definition.

| Field | Notes |
| --- | --- |
| `port` | The provider's own port (default 8081). |
| `version` | Provider instance version (commit id until first release). |
| `instance_status` | `registering` `initializing` `running` `unhealthy` `error` `disconnected` |
| `backend_status` | `stopped` `initializing` `starting` `running` `in_use` `stopping` `error` |
| `websocket_connected` | DB mirror; authoritative liveness is the Redis presence key. |
| `epoch` | Connection epoch: bumped on every accepted socket; stale-epoch frames ignored (fencing). |
| `last_seen` / `last_request_at` | Liveness + idle tracking. |
| `config_fingerprint` | SHA-256 of the applied `backend_config`; drives auto cache-clear and the Phase 9 `provider.config.update` push (admin PATCH + reconnect self-heal). |
| `reported_schema_fingerprint` | Schema fingerprint the instance presented at its last registration attempt (Phase 12; drives the `waiting_schema` badge and the pending voter roster). |
| `assigned_gpus` | Instance-reported GPU UUIDs this backend is bound to; VRAM accounting + metrics dedup. |

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

### Provider environment (all env-derived, no DB)
| Variable | Required | Meaning |
| --- | --- | --- |
| `MACHINE_UID` | yes | UID of the Machine this container runs on |
| `PROVIDER_REGISTRATION_TOKEN` | yes | Token of the ProviderDefinition it serves |
| `ADMIN_BASE_URL` | yes | e.g. `http://admin:8000` |
| `PROVIDER_PORT` | no (8081) | Port the spec-compliant API listens on |
| `CACHE_DIR` | yes (`/cache`) | Prompt caches, persisted `provider_config.json` |
| `MODELS_DIR` | yes (`/models`) | Model artifact storage |
| `METRICS_CATEGORIES` | no | Space-delimited: `gpu_usage vram os_ram cpu storage`. Inference metrics are **always** enabled and must not appear here. |

`LLAMA_SERVER_PATH` and other backend-binary paths come from the
environment, never from `backend_config`.

### Sequence
```
Provider                          Admin
    │ POST /admin/api/providers/register
    │  {machine_uid, registration_token, provider_type, schema,
    │   version, port, hardware, metrics_categories}
    │ ──────────────────────────────▶
    │  validations (401/403/404/409):
    │    registration_token → ProviderDefinition exists
    │    provider_type registered in ProviderType   (Phase 12;
    │      unknown type → created from this schema, committed)
    │    schema consensus gate                      (Phase 12;
    │      mismatch → 409 schema_pending/schema_conflict)
    │    definition.provider_type == container's provider_type
    │    version == admin settings.VERSION   (HARD FAIL)
    │    machine_uid exists
   │  effects:
   │    merge hardware into Machine, upsert ProviderInstance
   │    issue per-instance secret → Redis im:ws:secret:{id}
   │ ◀──────────────────────────────
   │  {instance_id, instance_secret, provider config, fingerprint}
   │ write provider_config.json to CACHE_DIR
   │ dial ws(s)://{admin}/provider/ws
   │   Authorization: Bearer {instance_secret}
   │ ──────────────────────────────▶  auth vs Redis (constant-time)
   │                                  accept → bump im:ws:epoch:{id}
   │   ◀── provider.hello {epoch}    claim im:ws:owner:{id}
   │                                  instance_status → running
```

- **Version hard fail:** provider versions must match the admin exactly
  (409). Admin and provider images are deployed together (§10).
- **Schema gate (Phase 12):** each agent ships a `schema.json` and sends
  it with the registration body. A fingerprint mismatch against the
  committed schema is refused **409 `schema_pending`** — the agent stays
  in a *waiting for consensus* state (visible in the UI as
  `waiting_schema` on the instance) and keeps retrying via the normal
  backoff until every known instance of the type has presented the new
  schema (or the operator force-commits).
- **Socket lifecycle:** if the WS dies, the admin marks the instance
  `disconnected` immediately (live path) and via the presence sweep
  (safety net for missed disconnects / admin restarts). The provider
  reconnects with exponential backoff (1s → 30s max). Each accepted
  reconnect gets a strictly greater **epoch**; the old socket is closed
  with code `4409`.

### Frame envelope (both directions)
```json
{ "v": 1, "type": "...", "id": "...", "reply_to": null,
  "epoch": 12, "ts": "ISO-8601 UTC", "payload": {} }
```
Commands carry an `id` and require an `ack` frame (`reply_to=<id>`,
payload `{ok, error, detail}`). The admin's
`ConnectionManager.send_command(instance_id, type, payload, timeout)`
awaits the matching ack (default 30s). Every frame carries the epoch;
stale-epoch frames are discarded by both sides.

### Event / command catalog
Provider → admin: `provider.status`, `backend.status`,
`backend.boot_requested`, `metrics.machine`, `backend.logs`,
`provider.logs`, `download.progress`, `backend.metadata`, `ping`
(`metrics.inference` is defined but reserved — see §8).

Admin → provider: `provider.hello`, `backend.start`, `backend.stop`,
`backend.restart`, `provider.initialize`, `provider.config.update`,
`metrics.assign`, `metrics.unassign`, `metrics.category.start`,
`cache.clear`, `storage.prune_unused`, `backend.logs.get` (Phase 13),
`pong`.

`backend.start` acks only after the provider's lifecycle reaches
`running`, so a successful return means the provider's `/v1` is live.
`backend.stop` stops the backend plainly (→ STOPPING → STOPPED):
in-flight streams are not force-cancelled and there is no in_use refusal
on this command — their producer tasks release their slots as the
upstream closes (a client may see the stream end early). Real drain
semantics live in `provider.config.update` (below), which refuses to
stop while slots are held.

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
   `Admission{instance_id, base_url, machine_uid}`.

`release(alias, request_id)` is idempotent and cancellation-safe
(`asyncio.shield` around cleanup so a client-disconnect-mid-stream still
releases and wakes the next waiter). It does **not** clear the booted
instance's VRAM hold.

### Idle-timeout reaper (Phase 6)

`_idle_reaper` runs every `IDLE_REAPER_INTERVAL_SECONDS` (default 15.0,
injectable via the `InferenceScheduler` constructor for tests), started
by `start_background` and stopped by `stop`. Each tick:

- Refreshes the `im:vram:used` TTL for every hold this process owns (so
  a long boot outliving the 60s key TTL keeps its mirror alive).
- Stops every **connected** instance whose `backend_status` is
  `running`/`in_use`, which has **zero active in-process slots**, and
  whose `last_request_at` is at least the definition's
  `idle_timeout_seconds` old (never-requested instances fall back to
  `created_at`). `idle_timeout_seconds == 0` means **never idle-stop**.
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
7. Provider translation layer normalizes the backend's output to spec on
   the instance port; the slot is held for the inbound connection
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
| `/v1/models` | **Implemented** — Phase 7; derived from enabled `ProviderDefinition`s (alias asc, `owned_by`=provider_type, `model_metadata` merged) |
| `/v1/embeddings`, `/v1/completions` (legacy), `/v1/rerank`, `/v1/moderations`, `/v1/decisions`, `/v1/audio/*`, `/v1/files`, `/v1/batches` | **501 stubs** — accepted regressions (§12) |
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
- **Provider side**: the provider-port `/v1/chat/completions` now
  honors `stream=false` by aggregating the driver's chunk stream into a
  `chat.completion` JSON (same fix class as Phase 6's responses
  non-stream path).

---

## 8. Metrics

### Machine-level metrics (deduped)
Multiple provider instances share one machine; each hardware resource is
emitted by **exactly one** connected instance. `metrics_service.py`
manages ownership via Redis `im:metrics:owner:{machine_uid}` (SET NX,
TTL 30s, refreshed on each `metrics.machine` receipt). On connect the
admin assigns unowned resources (subject to the instance's declared
`im:metrics:cats:{instance_id}`); on expiry/disconnect another instance
on the same machine can take over. Epoch fencing makes failover safe
against half-open sockets.

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
| `im:ws:secret:{instance_id}` | String | 30d | Per-instance WS secret (trusted LAN, plaintext; not in Postgres) |
| `im:ws:epoch:{instance_id}` | Counter | none | Monotonic connection epoch (INCR per accepted socket) |
| `im:ws:owner:{instance_id}` | String | none | Connection token of the currently accepted socket |
| `im:ws:presence:{instance_id}` | String | 60s | Liveness marker; absence ⇒ sweep marks disconnected |
| `im:metrics:owner:{machine_uid}` | String | 30s | Which instance emits machine-level metrics for the machine |
| `im:metrics:machine:{machine_uid}` | String | 30s | Latest machine-level snapshot JSON |
| `im:metrics:cats:{instance_id}` | String | — | Instance's declared metrics categories (JSON list) |
| `im:sched:queue:{alias}` | List | — | Queued request ids (mirror of in-process deque) |
| `im:sched:wait:{req_id}` | Hash | 1h | position / enqueued_at / status |
| `im:sched:active:{alias}` | Set | — | Admitted request ids (cardinality ≤ capacity) |
| `im:sched:lock:{alias}` | String | 5s (SET NX PX) | Admission lock contract |
| `im:vram:used:{machine_uid}` | Hash | 60s | `{instance_id}` → bytes held **per booted instance** (§6); refreshed on boot/stop, TTL kept alive by the reaper |
| `im:logs:backend:{instance_id}` | List | 1h | Backend stdout/stderr tail (Phase 13; capped ~2000 lines, newest left) |
| `im:logs:provider:{instance_id}` | List | 1h | Provider's own log tail (Phase 13; same cap) |

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
| provider images | one per provider type from `provider/<type>/Dockerfile` | run on hardware hosts; mock for local dev |

**Deploy order (mandatory, due to version hard-fail):** admin first, then
all provider instances. A version-mismatched provider refuses to start
with a clear 409. Accepted constraint of the solo-operator model.

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
- The **only** secrets are the per-definition `registration_token`
  (plaintext in provider env — possession proves entitlement) and the
  per-instance WS `instance_secret` issued at registration (Redis).
- Token rotation is manual: edit the definition, redeploy the provider.
  No revocation short of deleting the definition/instance.
- Threat accepted: anyone on the network can call inference and the admin
  API; anyone with `registration_token` + a valid `MACHINE_UID` can
  register a rogue instance.

---

## 13. Known Limitations & Accepted Regressions

1. Trusted-LAN only (§12).
2. Version hard-fail requires coordinated admin+provider deploys (§10).
3. Embeddings, legacy completions, rerank, moderations, decisions, audio,
   files, batches: 501 stubs (§7). **Do not "restore" old
   implementations from git history** — re-implement against the new
   provider/scheduler model when needed.
4. No prompt-cache tracking table; cache is provider-local via
   fingerprint.
5. Multi-file model artifacts are JSON (`backend_config`), not relational
   — no cross-definition artifact dedup.
6. `MACHINE_UID` misuse (same UID on two physical hosts) poisons VRAM
   admission; there is no host-fingerprint check.
7. **Schema consensus (Phase 12)** requires every known
   `ProviderInstance` row of a type to present the new schema before it
   commits. A machine that is permanently gone (row never deleted) blocks
   consensus forever unless the operator **force-commits** the pending
   schema. Stale instance rows should be deleted when hardware is
   decommissioned.
7. Scheduler VRAM eviction is implemented (§6): idle different-alias
   backends are stopped LRU-first to make room. Remaining nuance:
   eviction is per-machine (no cross-machine rebalancing) and victims are
   chosen purely by LRU idle time — there is no priority/weight scheme,
   so a just-booted large model can be evicted for a newly arriving
   request if it happens to be the oldest idle hold.
8. The idle-timeout reaper is implemented (§6). Remaining nuance: idle
   detection is admin-side from `last_request_at` + in-process slot
   counts, so a backend kept busy by traffic that bypasses the admin
   (direct provider-port calls) is invisible to the reaper.
9. Benchmarks removed; performance testing is out-of-band.
10. Config-update retry is bounded (3 attempts for drain-refused only);
    a provider that stays busy past the retries surfaces the failure in
    the PATCH response and relies on the reconnect/sweep self-heal for
    eventual consistency. The admin does not queue config pushes.

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
