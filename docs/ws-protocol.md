# Inference Matrix — Admin ⇄ Provider Wire Protocol

This document specifies the registration handshake and the provider WebSocket
protocol between a **provider agent** (a hardware-local container —
`provider/mock`, `provider/llama-cpp`, `provider/halogen`,
`provider/halogen-flash`, `provider/gufo`, `provider/talkies`) and the **admin**
(`admin/backend`, the stateless broker).

**Phase 16 (machine-scoped agents) is the live model.** A provider container is
a **ProviderAgent**: it is bound to exactly one **machine** + one
**provider type** + a stable operator-supplied **`AGENT_ID`**, and it hosts
**1..N backends** (one per placed `ProviderDefinition`, each a
`ProviderInstance`). The agent holds **one** WebSocket to the admin that
multiplexes control frames for all of its backends; per-backend frames carry
the target `instance_id` in their payload. `ARCHITECTURE.md` §5 is the
canonical overview; this document is the wire-level detail. Where the two
disagree, this document is more specific about frames and this is the RFC.

Auth model: **trusted LAN**. The shared **machine secret**
(`Machine.registration_secret`, presented as `MACHINE_SECRET`) gates
registration; the per-**agent** `agent_secret` (minted at registration, stored
only in Redis) gates the WebSocket. `/admin/api` and `/v1` are unauthenticated
by design.

> **Data plane note (Phase 24).** The agent's single `base_port` `/v1` surface
> also carries the `/v1/audio/*` routes (speech / transcriptions / live-ASR WS /
> voices). These ride the same HTTP data plane as `/v1/responses` — **no new WS
> frame kinds** were added; the control plane is unchanged.

The frame envelope, `FrameKind` set, and `Ack` shape are canonical in
`provider/lib/provider_lib/wire.py` and mirrored in
`admin/backend/app/services/wire.py` (the admin image does not ship provider
packages). **If you change one, change the other** — a drift test guards them.

---

## 1. Frame envelope

Every WebSocket message is a single JSON object ("frame"):

```json
{
  "v": 1,
  "type": "<event_or_command_name>",
  "id": "<message id, unique per sender connection>",
  "reply_to": "<id of the message this replies to, or null>",
  "epoch": 3,
  "ts": "2026-10-04T12:00:00+00:00",
  "payload": { }
}
```

| Field | Meaning |
| --- | --- |
| `v` | Protocol version, currently `1`. |
| `type` | Frame kind, e.g. `ping`, `provider.status`, `agent.assignments.update`. See §4. |
| `id` | Unique id chosen by the sender for this frame (UUID recommended). |
| `reply_to` | Set on replies/acks to the `id` of the frame being answered; `null` otherwise. |
| `epoch` | The connection epoch assigned by the admin when this agent's socket was accepted (see §3). |
| `ts` | ISO-8601 UTC timestamp from the sender. |
| `payload` | Kind-specific JSON object. |

Commands carry an `id` and require an `ack` frame (`type="ack"`,
`reply_to=<command id>`) whose payload validates as `Ack`:

```json
{"ok": true, "error": null, "detail": { }}
```

The admin's `ConnectionManager.send_command(agent_id, type, payload, timeout)`
sends on the agent's socket and awaits the matching ack (default
`COMMAND_TIMEOUT_SECONDS = 30`). **The socket is agent-level; a per-backend
command puts the target `instance_id` inside `payload`.** Every frame carries
the agent's epoch; stale-epoch frames are discarded by both sides (§3).

---

## 2. Registration (HTTP)

`POST {ADMIN_BASE_URL}/admin/api/providers/register`

Called by the provider **agent** container at startup, before dialing the
WebSocket. It identifies the container as a `(machine_uid, provider_type,
agent_id)` triple and authenticates with the machine's shared secret.

### Request body

```json
{
  "machine_uid": "gpu-box-1",
  "machine_secret": "<Machine.registration_secret>",
  "agent_id": "llama-a",
  "provider_type": "llama-cpp",
  "schema": { },
  "version": "dev",
  "base_port": 8081,
  "hardware": {
    "gpus": [
      {"uuid": "gpu-1", "vendor": "nvidia", "name": "RTX 4090",
       "total_vram_bytes": 25769803776}
    ],
    "total_vram_bytes": 25769803776
  },
  "metrics_categories": ["cpu", "gpu_usage", "os_ram", "storage", "vram"],
  "registered_at": "2026-10-04T12:00:00+00:00"
}
```

| Field | Notes |
| --- | --- |
| `machine_uid` | The `Machine.uid` this container runs on (env `MACHINE_UID`). Must be pre-created in the admin UI. |
| `machine_secret` | The shared `Machine.registration_secret` (env `MACHINE_SECRET`). Possession proves the container belongs to that machine. **Replaces the retired per-definition `registration_token`.** |
| `agent_id` | Operator-supplied stable id (env `AGENT_ID`) discriminating multiple agents that share a `(machine, provider_type)`. |
| `provider_type` | The single backend type this agent runs (env `PROVIDER_TYPE`). Must match a registered `ProviderType` (or bootstrap it — see the schema gate). |
| `schema` | The provider package's committed `schema.json` (JSON Schema 2020-12 for this type's `backend_config`). The admin derives `schema_fingerprint = sha256(canonical_json(schema))` itself; the agent never sends the fingerprint. Optional only as a Phase 12 transition path (see below). |
| `version` | Must exactly equal the admin `settings.VERSION` (lockstep deploy). |
| `base_port` | The agent's single admin-facing `/v1` port (its `PROVIDER_PORT` env, default 8081). The admin dials `http://{machine.reachable_address()}:{base_port}/v1/...` for **every** backend of this agent; the agent routes each request to the right backend by the request `model` (= definition alias). There are no per-backend port offsets, and the admin does **not** police port uniqueness across agents. |
| `hardware` | Inventory merged into `Machine.hardware` as a per-GPU **union by uuid** (latest report wins per uuid); `total_vram_bytes` is the auto-sum of that union. |
| `metrics_categories` | Space-delimited source list, sent sorted. Machine-level categories only — **never `inference`** (always on). |

### Validation rules (admin side, in order — `app/api/admin/providers.py`)

1. `machine_uid` must reference a `Machine` → otherwise **404**.
2. `machine_secret` must match `Machine.registration_secret` (constant-time)
   → otherwise **401**.
3. **Version gate**: `version == settings.VERSION` → otherwise **409**.
4. If `schema` is present it must parse as JSON Schema (2020-12) → otherwise
   **422** (checked before any mutation).
5. Upsert the `ProviderAgent` for `(machine_id, provider_type, agent_id)`
   (`agent_status="registering"`, `base_port`, `version`, `assigned_gpus` = the
   GPU **uuid strings** from `hardware.gpus`).
6. Merge the hardware report into `Machine` as a per-GPU **union by uuid**
   (latest report wins per uuid; `total_vram_bytes` is the auto-sum of the
   union). Done before the schema gate, so a refused (409) registration still
   refreshes the inventory.
7. **Schema-consensus gate** (Phase 12) — see below. A refusal is **409** with a
   structured `detail`.
8. Resolve the agent's **placed definitions** (`_placed_definitions`): every
    enabled `any_of_type` definition of this type, plus every enabled `specific`
    definition linked to this agent via `definition_agents`; sorted by alias.
    Reconcile one `ProviderInstance` per placed definition (create stopped rows,
    refresh fingerprints, retire de-placed ghosts busy-safe). The admin allocates
    **no** per-backend ports — every backend of this agent is reached through the
    agent's single `base_port` `/v1` surface, routed by `model`, so there is no
    port assignment and no cross-agent port-conflict check.
9. Mint a fresh **per-agent** `agent_secret` (`secrets.token_urlsafe(32)`),
   store it **only in Redis** under `im:ws:secret:{agent_id}` (30-day TTL), and
   return it once. Persist the declared `metrics_categories` under
   `im:metrics:cats:{agent_id}`.

> **`{agent_id}` in every Redis key below is the `ProviderAgent` primary-key
> uuid** (`str(agent.id)`) handed back in the registration response — **not**
> the operator-supplied `AGENT_ID` string. The admin resolves the operator
> `agent_id` to its PK row at registration and uses the PK everywhere.

### Schema consensus (Phase 12)

The admin keeps a `ProviderType` row per type holding the **committed** schema +
fingerprint and (optionally) a **pending** schema + fingerprint + voter list.
**The voter universe is every `ProviderAgent` row of this type** (agents ship
the schema), regardless of connection state. On a registration that passes
rules 1–6:

| Case | Admin action | Result |
| --- | --- | --- |
| Type not registered | Create `ProviderType` with `schema` committed; `status=active`; read `x-max-running-backends` from the schema. | 200 (bootstrap). |
| `sha256(schema)` == committed fingerprint | No change (resolves a stale conflict to `active` if nothing pending). | 200. |
| `sha256(schema)` != committed, **no pending staged** | If this agent is the only member of the universe → commit immediately (unanimous). Otherwise stage `pending_schema`/`pending_fingerprint`, `pending_voters=[this agent]`, `status=consensus_pending`. | 200 when sole/unanimous; else **409 `schema_pending`** — the agent stays in `waiting_schema` and retries. |
| `sha256(schema)` == pending fingerprint | Add this agent to `pending_voters`. If voters now cover **every `ProviderAgent` of this type** → commit (`committed = pending`, clear pending, `status=active`). | **409 `schema_pending`** while incomplete; **200** once the last voter registers. |
| `sha256(schema)` != committed and != staged pending | Record `reported_schema_fingerprint`; leave pending untouched. | **409 `schema_conflict`**. |

Notes:

- **TRANSITION (Phase 12):** `schema` is optional until every provider package
  ships its `schema.json`. When omitted: unknown type → the admin bootstraps a
  permissive committed schema `{"type": "object"}` (`status=active`) and
  proceeds (200); known type → the gate is skipped and the registration proceeds
  (200) with `reported_schema_fingerprint = None`.
- The agent's `reported_schema_fingerprint` is written on **every** registration
  attempt (success or 409) so the UI can show who is on what schema.
- **Force-commit** (`POST /admin/api/provider-types/{name}/pending/commit`)
  promotes the pending schema immediately (affects future registrations and
  new/edited definitions only); **dismiss** drops the pending state. A
  permanently-dead agent row can block consensus until force-commit — delete
  decommissioned agent rows.
- A definition's `backend_config` is validated against the type's **committed**
  schema on create/PATCH (admin API, not the WS).

### Effects (summary)

- Upsert `ProviderAgent` (container-level state) + reconcile its
  `ProviderInstance` backend rows (per-backend state).
- Merge hardware into `Machine`; refresh `total_vram_bytes`.
- Mint the per-agent WS secret (Redis only, never Postgres).

### Response `200 OK`

```json
{
  "agent_id": "<ProviderAgent PK uuid str>",
  "agent_secret": "<token_urlsafe(32)>",
  "machine": {
    "id": "<uuid str>", "uid": "gpu-box-1", "name": "...",
    "host": "...", "dns": "...", "ip": "...",
    "total_vram_bytes": 25769803776, "hardware": { }
  },
  "backends": [
    {
      "instance_id": "<ProviderInstance PK uuid str>",
      "definition": {
        "id": "<uuid str>", "alias": "my-model", "provider_type": "llama-cpp",
        "backend_config": { }, "config_fingerprint": "<sha256 hex>",
        "idle_timeout_seconds": 300, "capacity": 1,
        "vram_required_bytes": 8589934592, "model_metadata": { },
        "served_models": [
          {"name": "my-model", "modality": "llm", "backend_config": { }, "enabled": true}
        ]
      }
    }
  ]
}
```

`backends` is the agent's full placed set — one entry per assigned
`ProviderDefinition`, each with its own `instance_id` and `definition` (whose
`alias` is the `model` name the agent's single `/v1` surface routes on). No
per-backend `port` is sent — every backend is reached through the agent's single
`base_port` `/v1`. A single-backend agent gets exactly one entry. The
agent persists this response to `CACHE_DIR/provider_config.json` and builds one
hosted backend (`BackendLifecycle`) per entry.

> **Phase 25 (additive).** `definition.served_models` is the canonical served-model
> list `[{name, modality, backend_config, enabled}]` — always present (a
> single-model definition sends exactly one entry). New agents prefer it and route
> on **any enabled** served name; **old agents ignore the key** and keep routing
> on `alias` (single-model behavior unchanged).

The client derives the WS URL from `ADMIN_BASE_URL` (scheme swap `http→ws`,
`https→wss`) and appends `?agent_id=<ProviderAgent PK uuid>`; a machine echo
may optionally carry `admin_ws_url` to override the base.

Errors are plain FastAPI `{"detail": "..."}` bodies, **except** the schema-gate
409s, whose `detail` is structured:

```json
{
  "detail": {
    "error": "schema_pending",
    "provider_type": "llama-cpp",
    "committed_fingerprint": "<sha256 hex>",
    "pending_fingerprint": "<sha256 hex>",
    "voted": ["<agent_id>", "..."],
    "waiting_on": ["<agent_id>", "..."]
  }
}
```

(`error` is `schema_conflict` in the conflicting-pending case, with the same
fields; `voted`/`waiting_on` are `ProviderAgent` PK uuid strings.) The agent
raises `SchemaPendingError` from this and keeps retrying via the normal backoff.

---

## 3. WebSocket connect, auth, and epoch fencing

### Endpoint

`GET /provider/ws?agent_id=<ProviderAgent PK uuid>` — mounted at the **app
root** (not under `/admin/api`). One socket per agent; it multiplexes all of
that agent's backends.

### Connect handshake (`app/api/ws.py`)

1. Agent dials the WS with header `Authorization: Bearer <agent_secret>` and
   the `agent_id` query param (the PK uuid from registration).
2. Admin verifies the secret against `im:ws:secret:{agent_id}` with
   `secrets.compare_digest`. Missing header / unknown agent / mismatch → close
   **4401** before accept. Redis unavailable at auth time → close **1013**.
3. Admin **INCRs** `im:ws:epoch:{agent_id}` (monotonic, never deleted) to mint
   the new connection **epoch**, and sets `im:ws:owner:{agent_id}` to a unique
   connection token (no TTL).
4. Admin sends the **hello** frame first:

   ```json
   {
     "v": 1, "type": "provider.hello", "id": "", "reply_to": null,
     "epoch": 4, "ts": "...",
     "payload": {"epoch": 4, "server_time": "..."}
   }
   ```

   The agent reads `payload.epoch` from the first frame and stamps all
   subsequent frames with it.
5. Admin marks the `ProviderAgent` row connected (`websocket_connected=true`,
   `epoch`, `last_seen`). `agent_status` stays `registering` until the agent
   emits its first `provider.status`.
6. Background tasks are spawned off the accept path (each exception-suppressed
   so a slow one never blocks the socket):
   - **metrics ownership** — try to make this agent the machine's metrics
     reporter (`metrics_service.assign_ownership`).
   - **config self-heal** — push `provider.config.update` to any of the agent's
     backends whose stored `config_fingerprint` lags its definition
     (`heal_agent_stale_fingerprints`).
   - **proactive warm-up** — boot the agent's assigned backends one at a time,
     bounded by VRAM + `max_running_backends`
     (`scheduler.warm_up_agent`), gated by `settings.SCHEDULER_WARMUP_ON_CONNECT`
     (default on; off in the test app).
7. Presence: `im:ws:presence:{agent_id}` (TTL **60s**) is refreshed on every
   accepted inbound frame and on every outbound command/pong. A background
   sweep marks any agent whose presence key expired `disconnected` (and its
   backends unschedulable) — the safety net for missed disconnects / admin
   restarts. The agent keeps the socket warm with a `ping` every 20s
   (`AdminClient.PING_INTERVAL_SECONDS`), well under the 60s TTL.

### Epoch fencing rule

Every non-hello frame carries the sender's current `epoch`. The admin drops
(logs and ignores) any inbound frame where `frame.epoch != connection.epoch` —
lower means a stale, already-replaced socket; higher indicates a protocol
violation. This prevents a zombie socket from mutating state or consuming
command replies after the agent reconnected.

### Reconnect / disconnect

The agent uses exponential backoff (1s → 30s max, `AdminClient.run_forever`).
Each accepted reconnect gets a strictly greater epoch; the previous socket for
the same agent is closed with code **4409** (conflict). On disconnect the admin
clears `im:ws:owner:{agent_id}` (only if still owned by the dying connection)
and marks the `ProviderAgent` `disconnected`, clearing each backend's
`backend_loaded_at` so the next loaded status re-arms the idle clock. The
epoch key is **not** deleted.

**Connect-time snapshot reconcile.** The admin deliberately leaves each
`ProviderInstance.backend_status` untouched on socket teardown, and a restarted
agent container starts every lifecycle in `stopped` while the lifecycle only
emits `backend.status` on transitions — so without a reconcile the mirror stays
stale (`running`) and the scheduler keeps routing to a proxy that answers 503.
To heal it, on **every** accepted connection the agent emits one `backend.status`
frame per hosted backend carrying the lifecycle's current status
(`{"instance_id", "backend_status", "reason": "connect snapshot"}`, via
`provider_lib.ops.emit_backend_status_snapshot`, called from each provider's
`on_connected` right after `provider.status`). The admin persists it exactly like
any other transition frame (updating `backend_status` and pruning scheduler VRAM
holds for a reported `stopped`). Epoch fencing drops any snapshot from a
superseded socket.

### Close codes

| Code | Meaning |
| --- | --- |
| 4401 | Authentication failed (missing/bad secret, unknown agent). |
| 4409 | Connection replaced by a newer socket for the same agent. |
| 1013 | Admin not ready (Redis unavailable at auth time). |

---

## 4. Frame catalog

### Provider agent → admin (events)

Events do **not** require an ack; the admin persists/acts on them. Per-backend
events carry `instance_id`; agent-level events do not.

| type | payload | status | handling |
| --- | --- | --- | --- |
| `ping` | `{}` | Live | Admin replies `pong` with `reply_to = ping.id`. Also serves as the presence keepalive. |
| `provider.status` | `{"agent_status": "...", "error_message": "..."}` | Live | **Agent-level.** Updates `ProviderAgent.agent_status` / `error_message`, `last_seen=now`. The admin also accepts the legacy key `instance_status` (emitted by `provider_lib.ops`/`config_update`) in place of `agent_status`, and `error` in place of `error_message`. |
| `backend.status` | `{"instance_id": "...", "backend_status": "...", "error_message": "..."}` | Live | **Per-backend.** Requires `instance_id` (a frame without it is ignored). Updates that `ProviderInstance.backend_status` (+ optional error), `last_seen=now`, and the idle-reaper load clock. A cross-agent guard drops a frame naming an instance this agent does not own. `status` is accepted as an alias for `backend_status`. |
| `metrics.machine` | GPU categories (`vram`/`gpu_usage`, filtered to the agent's `ASSIGNED_GPU_UUIDS`, with `assigned_gpus`) from **every** agent + machine-wide categories (`os_ram`/`cpu`/`storage`) from the owner | Live | GPU sections stored per-agent at `im:metrics:machine:{machine_uid}:agent:{agent_id}` (merged per-uuid on read); machine-wide sections stored at `im:metrics:machine:{machine_uid}` only from the owner (refreshes the lease). Emitted by `MachineMetricsEmitter` from connect; `set_owned` toggles only the machine-wide categories. |
| `download.progress` | `{filename, progress_percent, bytes_downloaded, total_bytes, speed_mbps, phase}` | Live | Emitted by `provider_lib.downloader` (throttled). |
| `backend.logs` | `{"instance_id": "...", "lines": [{"ts", "stream": "stdout"\|"stderr", "text"}], "dropped": int}` | **Live (Phase 13)** | Captured backend subprocess stdout/stderr. Provider tees spawn pipes into a bounded ring (`provider_lib.log_ring.CursorLogRing`, `LOG_RING_LINES` default 2000); flushes throttled batches (~1s / max 100 lines) while connected; `dropped` is the cumulative lost-line counter. Admin appends to Redis `im:logs:backend:{instance_id}` (capped ~2000, TTL 1h). Catch-up after reconnect via `backend.logs.get`. |
| `provider.logs` | `{"lines": [{"ts", "stream": "stdout", "text"}], "dropped": int}` | **Live (Phase 13)** | **Agent-level.** The agent process's own logger output, captured via a ring-buffer handler and flushed with the same throttle. Admin stores in Redis `im:logs:provider:{agent_id}`. |
| `backend.metadata` | `{"instance_id": "...", "models": [...]}` | **Live** | Published after a boot that scraped the engine (`backend.start` / `backend.restart` / `provider.initialize`, via the driver's `list_models()`). The admin stores `{"models": [...]}` verbatim on the definition's `model_metadata`. Best-effort: a scrape failure never fails the boot; an empty list emits nothing. (`provider.config.update` does NOT emit this event — it returns scraped metadata in its ack `detail.model_metadata`.) |
| `backend.boot_requested` | `{}` | Reserved | Observability only; boot is admin-driven (scheduler sends `backend.start`). |
| `metrics.inference` | available/max slots, token speed, prompt-processing speed, in-flight | Reserved (defined in `FrameKind`, not yet emitted; the admin does not handle it yet) | Always-on per-backend telemetry; lands with Phase 7/8. Never deduped. |

`agent_status` ∈ `registering|initializing|running|unhealthy|error|disconnected`
(`InstanceStatusValue`). The Phase 14 `awaiting_config` pre-state is **retired**
(no shells). `backend_status` ∈
`stopped|initializing|starting|running|in_use|stopping|error`
(`BackendStatusValue`).

Unknown event types are logged and ignored (forward compatible).

### Admin → provider agent (commands)

Commands carry an `id`; the provider replies with `type="ack"`,
`reply_to=<command id>`, payload validating as `Ack`. **Per-backend commands
(`backend.*`, `provider.config.update`, `cache.clear`, `storage.prune_unused`)
carry the target `instance_id` in the payload**; the provider's
`BackendRegistry` routes the frame to the correct hosted lifecycle and NAKs
`unknown_instance` for an id it does not host.

| type | payload | status | notes |
| --- | --- | --- | --- |
| `agent.assignments.update` | `{"assignments": [...], "max_running_backends": int}` | **Live (Phase 16 slice 5)** | The agent-level counterpart to `provider.config.update`. When a definition's placement changes (created / PATCHed `agent_placement`/`agents`/`enabled`/`alias`), the admin recomputes the agent's placed set, reconciles its `ProviderInstance` rows, and pushes the **full** current assignment set. See the payload/ack detail below. |
| `backend.start` | `{"instance_id": "...", "wait_for_running": bool = true}` | Live | Provider awaits `BackendLifecycle.start()` and — with `wait_for_running: true` (the scheduler's contract, and the default) — acks only once the lifecycle reaches running, with `detail.capacity` + whatever ports the driver exposes (`api_port`/`engine_port`/`backend_port`/`effective_capacity`, omitted when absent) — these are the backend's **private** engine ports (OS-assigned by default; halogen-flash's static pair), reported for information only: the admin never dials them, it reaches the backend via the agent's single `base_port` `/v1` routed by `model`. With `false` the boot runs in a background task and the ack is `{"ok": true, "detail": {"accepted": true, "backend_status": "..."}}` immediately: a cold halogen-flash boot downloads its checkpoint AND companions inside the driver's health wait — tens of GB, tens of minutes — so neither the ack window nor the admin's HTTP request may sit open that long. The outcome then arrives as events: `backend.status initializing` heartbeats, then `running`/`error`, then `backend.metadata`. Waiting while a background boot is already in flight JOINS that boot (never a second spawn); an accept-style call while one is in flight NAKs `boot_in_progress`. Admin timeouts: `BACKEND_BOOT_TIMEOUT_SECONDS` (3600) when waiting, the 60s action window otherwise. |
| `backend.stop` | `{"instance_id": "..."}` | Live | Provider awaits `BackendLifecycle.stop()` (→ STOPPING → STOPPED, emitted) and acks ok. **No forced drain**: in-flight streams are not cancelled — their producer tasks release their slots as the upstream closes (the client may see the stream end early). Use `provider.config.update` or `backend.restart` when drain semantics matter. Refused `boot_in_progress` while an accepted-style transition is running. |
| `backend.restart` | `{"instance_id": "...", "wait_for_running": bool = true}` | **Live** | Stop + start, same wait flag as `backend.start`. **Drain-checked first**: NAKs `{"ok": false, "error": "backend_in_use", "detail": {"step": "restart", "retry_after": 10, "in_flight": N}}` while live requests hold slots, so a restart never cuts a stream. |
| `provider.initialize` | `{"instance_id": "...", "wait_for_running": bool = false}` | **Live** | Full re-provision, driven by the provider package's `re_register` callback: (1) POST `/admin/api/providers/register` again — re-checks the version + schema gates, re-adopts `capacity`/`backend_config`/fingerprint, mints a fresh agent secret and rewrites `CACHE_DIR/provider_config.json`; the live socket is **not** recycled (already authenticated at the current epoch; the new secret is for future reconnects). (2) Drain check — NAKs `backend_in_use` while slots are held. (3) ack (default `{"ok": true, "detail": {"accepted": true, "steps": ["stop","start","metadata"]}}`), then in the background: stop → start (engine downloads/loads, heartbeated as `initializing`) → `list_models()` → `backend.metadata`. Terminal state via `provider.status` (`running` / `error` + `error_message`). |
| `provider.config.update` | `{"instance_id": "...", "backend_config": {...}, "config_fingerprint": "<sha256 hex>", "idle_timeout_seconds": int, "capacity": int}` | **Live (Phase 9)** | Apply a new definition config in place. Provider order matters: (1) **capacity adopt** — a differing `capacity` is applied to `lifecycle.capacity` immediately (no restart); (2) **noop** — received fingerprint == applied → ack `{"ok": true, "detail": {"noop": true, "capacity_adopted": bool, ...}}`, always safely ackable under load; (3) **drain** — `lifecycle.stop_if_idle()` checks busy and transitions STOPPING under the same lifecycle lock with no intervening await; busy → NAK `{"ok": false, "error": "backend_in_use", "detail": {"step": "drain", "retry_after": 10, "in_flight": N}}`; (4) otherwise emits `provider.status initializing`, clears the old fingerprint's prompt cache, `driver.apply_config`, starts (downloads stream `download.progress`), scrapes `list_models()`. **Ok ack detail:** `{"config_fingerprint", "capacity", "model_metadata": [...], "prompt_cache_deleted", "prompt_cache_bytes_freed"}` — the admin persists the echoed fingerprint on the instance and `model_metadata` on the definition. **Failure NAK:** `{"ok": false, "error": "<msg>", "detail": {"step": "validate|drain|cache_clear|apply_config|start"}}`. `idle_timeout_seconds` rides in the payload for observability only — the provider does not consume it (idle reaping is admin-side). Admin retry: 300s per-instance timeout, only `backend_in_use` retried (3 attempts, 10s apart). A stale fingerprint on (re)connect is healed automatically to the **stale instance only** (connect path + presence sweep, per-instance in-flight guard). **Phase 25 (additive):** the payload may carry `served_models` (the canonical list); the admin's definition fingerprint folds `(backend_config, served_models)` so a per-model edit changes it and drives this update, and the provider refreshes the handle's served list via `set_models` **before** the restart. Old agents ignore the key. Full semantics in `provider/README.md`. |
| `metrics.assign` | `{"machine_uid": "...", "categories": [...]}` | Live (Phase 5; split Phase 17) | Grant **machine-wide** metrics ownership (`os_ram`/`cpu`/`storage`); admin `assign_ownership` sends it (only when the agent declares a machine-wide category). The emitter loop already runs from connect — this only flips `set_owned(True)` to include the machine-wide categories; GPU categories emit regardless. |
| `metrics.unassign` | resource list | Provider handler live; admin send not yet wired (ownership currently lapses via Redis lease expiry) | Revoke **machine-wide** metrics ownership (`set_owned(False)`); the loop keeps running so GPU telemetry keeps flowing. |
| `metrics.category.start` | category name | Reserved | Enable a `METRICS_CATEGORIES` category. |
| `cache.clear` | `{"instance_id": "...", "dry_run": bool = false, "force": bool = false}` | **Live (Phase 9)** | **Prompt-cache files only** — never model files. Deletes `CACHE_DIR/prompt_cache` plus the provider's engine cache dirs (`extra_cache_dirs`: gufo `CACHE_DIR/<instance_id>`, halogen-flash `CACHE_DIR/halogen-flash`). **Refused while the backend is `in_use`** (NAK `{"ok": false, "error": "backend_in_use", "detail": {"step": "drain", ...}}`) unless `force: true`; `dry_run` never touches files and is always allowed. Ack detail: `{"dry_run", "deleted": [...], "bytes_freed": N}`. |
| `storage.prune_unused` | `{"instance_id": "...", "dry_run": bool = false}` | **Live (Phase 9)** | Delete `MODELS_DIR` files not referenced by the driver's current `resolved_artifacts` set. **Refuses** (NAK `no_resolved_artifacts`) when the driver has not resolved its set yet — never deletes blind. Ack detail: `{"dry_run", "deleted": [...], "bytes_freed": N, "kept": [...]}`. |
| `backend.logs.get` | `{"kind": "backend"\|"provider"\|"all", "since": <ring seq or null>}` | **Live (Phase 13)** | Ask the provider for its current ring buffer (catch-up after an admin restart / long disconnect). Ack detail: `{"lines": [...], "dropped": int, "seq": int}` — `since` resumes at the ring sequence; omitting it returns the whole buffer. For `kind=all` the backend and provider rings share one `SeqCounter` (installed by `install_log_streaming`), so a single `since`/`seq` cursor resumes both correctly. |

**Two unrelated log seq spaces (Phase 13).** The `seq` in the
`backend.logs.get` ack is the **provider-side ring sequence** (a shared
`SeqCounter` across the backend+provider rings, starting at 0 per provider
process/connect lifetime). The `since`/`cursor` on the admin REST
`GET /admin/api/instances/{id}/logs` is the **admin ingest sequence**
(`im:logs:seq:{agent_id}`, Redis `INCR`, 1-based). These are independent
counters and MUST NOT be passed interchangeably by a client.

### `agent.assignments.update` detail (Phase 16 slice 5)

Payload (admin → provider) — the agent's **full** current assignment set:

```json
{
  "assignments": [
    {"instance_id": "<uuid>", "provider_definition_id": "<uuid>",
     "alias": "my-model", "backend_config": { }, "config_fingerprint": "<sha256>",
     "capacity": 1, "idle_timeout_seconds": 300,
     "vram_required_bytes": 0,
     "served_models": [
       {"name": "my-model", "modality": "llm", "backend_config": { }, "enabled": true}
     ]}
  ],
  "max_running_backends": 0
}
```

> **Phase 25 (additive).** Each entry carries the same canonical `served_models`
> list as the registration `definition` (single-model definitions send one
> entry). Old agents ignore the key; new agents prefer it (via
> `provider_lib.models.models_from_entry`) and fall back to
> `alias`/`modality`/`backend_config` otherwise.

Ack (provider → admin):

```json
{"ok": true, "added": ["<instance_id>"], "removed": ["<instance_id>"],
 "refused": [{"instance_id": "<id>", "reason": "..."}]}
```

The provider reconciles its `BackendRegistry` to the pushed set
(`provider_lib.assignments.install_assignment_handler`):

- **add** — for each assignment whose `instance_id` is not yet hosted, build a
  fresh `BackendHandle` via the package's `make_handle` factory (lifecycle +
  applied-config state seeded from the entry) and register it. Since slice 6
  every provider package supplies a `make_handle` and hosts N backends per
  process, so adds are normally served. An agent built without a factory
  **refuses** an add it cannot serve (`reason: no_handle_factory`) rather than
  crash. An already-hosted backend's lifecycle is **not** recreated here (that
  would drop its running state), but **Phase 25**: a changed `served_models` list
  IS refreshed in place via `registry.set_models` (a live add of a served name
  reaches the driver without a restart where the engine supports it). Other
  config changes still ride `provider.config.update`; assignments.update governs
  which backends exist and their served lists.
- **remove** — for each hosted handle no longer in the set, `stop_if_idle()` and
  drop it. A **busy** backend is refused (`reason: backend_in_use`) and kept
  until it frees — the exact mirror of the admin's busy-safe prune.
- Because the agent's single env-port `/v1` surface resolves the target backend
  from the live registry at request time, an added backend is immediately routable
  and a removed one immediately unroutable — there is **no** per-backend HTTP
  listener to bind/unbind and no registry change-listener to fire.

Admin side (`app/services/assignments.py`): `push_agent_assignments` runs the
shared `reconcile_agent_placement` diff (create stopped rows; retire de-placed
rows busy-safe), then pushes the full set over the socket. The admin allocates
no per-backend ports and does no cross-agent port-clash check — every backend is
reached through the agent's single `base_port` `/v1`, routed by `model`. On a
successful push that added backends the admin re-triggers the scheduler's
proactive warm-up (the slice-4 seam). A
disconnected agent still gets its rows reconciled (ghosts pruned) but receives
no frame — its next registration re-resolves authoritatively.

### Admin → provider agent (informational)

| type | payload | notes |
| --- | --- | --- |
| `provider.hello` | `{"epoch": int, "server_time": iso}` | First frame after a successful authenticated accept. |
| `pong` | `{}` | Reply to provider `ping` (`reply_to` set). |

---

## 5. Redis key summary

Full table + TTLs in `admin/backend/docs/redis-keys.md`. `{agent_id}` is the
`ProviderAgent` primary-key uuid; `{machine_uid}` is the `Machine.uid`.

| Key | Contents | Lifetime |
| --- | --- | --- |
| `im:ws:secret:{agent_id}` | Per-**agent** WS secret (plaintext, trusted LAN). | 30 days TTL; rewritten on each registration. |
| `im:ws:epoch:{agent_id}` | Monotonic connection epoch (counter). | Permanent (never deleted). |
| `im:ws:owner:{agent_id}` | Connection token of the currently accepted socket. | Deleted on disconnect or when superseded (no TTL). |
| `im:ws:presence:{agent_id}` | Liveness marker (timestamp). | 60s TTL, refreshed by traffic; absence ⇒ sweep marks the agent disconnected. |
| `im:metrics:owner:{machine_uid}` | `agent_id` of the single agent reporting the **machine-wide** metrics (`os_ram`/`cpu`/`storage`). | 30s lease (SET NX), refreshed on the owner's `metrics.machine`. |
| `im:metrics:machine:{machine_uid}` | Latest **machine-wide** owner snapshot JSON. | 30s TTL. |
| `im:metrics:machine:{machine_uid}:agent:{agent_id}` | Per-agent **GPU** partial JSON (`vram`/`gpu_usage`), written by every agent that reports them; merged per-uuid on read. | 30s TTL, refreshed per frame. |
| `im:metrics:cats:{agent_id}` | Agent's declared metrics categories (JSON list). | none; written at registration. |
| `im:logs:backend:{instance_id}` | Backend log tail (Phase 13; JSON lines, newest left). | ~2000-line cap, 1h TTL. |
| `im:logs:provider:{agent_id}` | Agent's own log tail (Phase 13; same shape). | ~2000-line cap, 1h TTL. |
| `im:logs:seq:{agent_id}` | Shared monotonic ingest cursor across both kinds on one agent. | 1h TTL. |
| `im:logs:dropped:{kind}:{id}` | Latest provider-reported dropped-line count per kind. | 1h TTL. |
