# Inference Matrix — Admin ⇄ Provider Wire Protocol

This document specifies the registration handshake and the provider
WebSocket protocol between a **provider instance** (hardware-local
container: `provider/mock`, `provider/llama-cpp`, `provider/halogen`,
`provider/halogen-flash`, `provider/gufo`) and the **admin**
(`admin/backend`, the stateless broker).

Auth model: **trusted LAN**. The registration token and the per-instance
secret gate only the provider WebSocket and the registration endpoint.
`/admin/api` and `/v1` are unauthenticated by design.

The frame envelope, registration/connection flow, and the commands/events
below are the live protocol. New command kinds are added here as they land
(see `IMPLEMENTATION_STATUS.md` for what's implemented vs. reserved).

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
| `type` | Frame kind, e.g. `ping`, `provider.status`. See §4. |
| `id` | Unique id chosen by the sender for this frame (UUID recommended). |
| `reply_to` | Set on replies/acks to the `id` of the frame being answered; `null` otherwise. |
| `epoch` | The connection epoch assigned by the admin when this connection was accepted (see §3). |
| `ts` | ISO-8601 UTC timestamp from the sender. |
| `payload` | Kind-specific JSON object. |

The canonical Python definitions live in
`provider/lib/provider_lib/wire.py` (`Frame`, `FrameKind`, `Ack`). The
admin keeps a synced copy in `admin/backend/app/services/wire.py` because
the admin image does not ship provider packages. **If you change one,
change the other.**

---

## 2. Registration (HTTP)

`POST {ADMIN_BASE_URL}/admin/api/providers/register`

Called by the provider container at startup, before dialing the WebSocket.

### Request body

```json
{
  "machine_uid": "gpu-box-1",
  "registration_token": "<token from the ProviderDefinition>",
  "provider_type": "mock",
  "schema": { },
  "version": "dev",
  "port": 8081,
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

`schema` is the provider package's committed `schema.json` (JSON Schema
2020-12 describing this type's `backend_config`). The admin derives
`schema_fingerprint = sha256(canonical_json(schema))` itself (same
canonicalization as `config_fingerprint`); the agent does not send the
fingerprint separately.

### Validation rules (admin side, in order)

1. `registration_token` must match an existing `ProviderDefinition`
   → otherwise **401**.
2. The definition must be `enabled` → otherwise **403**.
3. `provider_type` binding (Phase 14): the definition is typed → its
   `provider_type` must equal the container's → otherwise **409**; the
   definition is a **shell** (`provider_type` NULL) → the container's
   reported type is **adopted** onto the definition (no error). The
   response then carries `"type_adopted": true`.
4. `machine_uid` must reference a `Machine` pre-created in the admin UI
   → otherwise **404**.
5. **Version gate**: `version` must exactly equal the admin's
   `settings.VERSION` → otherwise **409** (lockstep deploy on the LAN).
6. **Schema gate (Phase 12)** — see the consensus section below.
   A `schema` that fails to parse as a JSON Schema (2020-12) → **422**.

### Schema consensus (Phase 12)

The admin keeps a `ProviderType` row per type holding the **committed**
schema + fingerprint and (optionally) a **pending** schema + fingerprint
+ voter list. On a registration that passes rules 1–5:

| Case | Admin action | Result |
| --- | --- | --- |
| Type not registered | Create `ProviderType` with `schema` as committed; `status=active`. | Registration proceeds normally (200). |
| `sha256(schema)` == committed fingerprint | No change. | Registration proceeds normally (200). |
| `sha256(schema)` != committed, **no pending staged** | If the voter universe is just this instance (unanimous) → commit immediately. Otherwise stage `pending_schema`/`pending_fingerprint`, `pending_voters=[this instance]`, `status=consensus_pending`. | **200** when solo/unanimous; else **409 `schema_pending`** — registration refused; the agent stays in `waiting_schema` and retries. |
| `sha256(schema)` == pending fingerprint | Add this instance to `pending_voters`. If voters now cover **every `ProviderInstance` row of this type** → commit: `committed = pending`, clear pending, `status=active`. | **409 `schema_pending`** while incomplete (voter count in detail); **200** once the last outstanding voter registers (the commit happens on that call). |
| `sha256(schema)` != committed and != a staged pending | Record `reported_schema_fingerprint`; do not overwrite pending. | **409 `schema_conflict`**. |

Notes:

- **TRANSITION (Phase 12 → D):** `schema` is optional until every
  provider package ships its `schema.json`. When omitted:
  - unknown type → the admin bootstraps a permissive committed schema
    `{"type": "object"}` (`status=active`) and the registration proceeds
    (200); `reported_schema_fingerprint` is the permissive fingerprint.
  - known type → the consensus gate is skipped and the registration
    proceeds (200); `reported_schema_fingerprint` is left `None` (the
    instance never proved its schema, so the UI must not show it as
    on-committed). This keeps old providers working in the admin-first
    deploy window. Remove this path once Phase D ships real schemas.
- The **voter universe** is every `ProviderInstance` row whose
  `provider_definition.provider_type == this type`, regardless of
  connection state. A decommissioned machine's row must be deleted or
  the operator must force-commit.
- The instance's `reported_schema_fingerprint` is written on **every**
  registration attempt (success or 409) so the UI can show who is on
  what schema.
- **Force-commit** (`POST /admin/api/provider-types/{name}/pending/commit`)
  promotes the pending schema to committed immediately. It affects
  **future registrations and new/edited definitions only** — connected
  old-schema instances keep running until they upgrade and re-register.
  **Dismiss** drops the pending state back to `active`.
- A definition's `backend_config` is validated against the type's
  **committed** schema on create/PATCH (admin API, not the WS); a
  committed schema change therefore does not retroactively invalidate a
  running config — the prestart migration check reports offending rows
  (report-only) and they surface as a UI banner.

### Effects

- Upserts the `ProviderInstance` row keyed by
  `(machine_id, provider_definition_id)`: updates `port`, `version`,
  sets `instance_status="registering"`, and stores
  `config_fingerprint = sha256(canonical_json(backend_config))`
  (canonical = `json.dumps(..., sort_keys=True, separators=(",", ":"))`).
- Merges the hardware report into `Machine.hardware` (latest report
  wins) and refreshes `Machine.total_vram_bytes` if the report contains
  a `total_vram_bytes` integer.
- Mints a **new per-instance secret** (`secrets.token_urlsafe(32)`) on
  every registration. The secret is stored **only in Redis** under
  `im:ws:secret:{instance_id}` (30-day TTL) and returned once to the
  provider. It is deliberately **not** persisted in Postgres
  (trusted-LAN operational secret).

### Response `200 OK`

```json
{
  "instance_id": "<uuid str>",
  "instance_secret": "<token_urlsafe(32)>",
  "type_adopted": false,
  "machine": {
    "id": "<uuid str>", "uid": "gpu-box-1", "name": "...",
    "host": "...", "dns": "...", "ip": "...",
    "total_vram_bytes": 25769803776, "hardware": { }
  },
  "provider_definition": {
    "id": "<uuid str>",
    "alias": "my-model",
    "provider_type": "mock",
    "backend_config": { },
    "config_fingerprint": "<sha256 hex>",
    "idle_timeout_seconds": 300,
    "capacity": 1,
    "vram_required_bytes": 8589934592,
    "model_metadata": { }
  }
}
```

Phase 14: for a shell definition `backend_config` and
`config_fingerprint` are `null` (never the hash of `{}`), and
`type_adopted` is `true` when *this call* set the definition's type
from the container's report.

The client derives the WS URL from `ADMIN_BASE_URL` (scheme swap
`http→ws`, `https→wss`); a definition echo may optionally carry
`admin_ws_url` to override that. Either way the client appends
`?instance_id=<uuid>` to the WS URL.

Errors are plain FastAPI `{"detail": "..."}` bodies, **except** the
Phase 12 schema-gate 409s, whose `detail` is structured:

```json
{
  "detail": {
    "error": "schema_pending",
    "provider_type": "llama-cpp",
    "committed_fingerprint": "<sha256 hex>",
    "pending_fingerprint": "<sha256 hex>",
    "voted": ["<instance_id>", "..."],
    "waiting_on": ["<instance_id>", "..."]
  }
}
```

(`error` is `schema_conflict` in the conflicting-pending case, with the
same fields.) The agent logs `waiting_schema` from this and keeps
retrying via the normal backoff.

---

## 3. WebSocket connect, auth, and epoch fencing

### Endpoint

`GET /provider/ws?instance_id=<uuid>` — mounted at the **app root**
(not under `/admin/api`).

### Connect handshake

1. Provider dials the WS with header:
   `Authorization: Bearer <instance_secret>` (from registration).
2. Admin looks up `im:ws:secret:{instance_id}` in Redis and compares
   with `secrets.compare_digest`. Missing header / unknown instance /
   mismatch → the socket is closed with code **4401** before accept.
3. Admin **INCRs** `im:ws:epoch:{instance_id}` (monotonic, never
   deleted) to mint the new connection **epoch**, and sets
   `im:ws:owner:{instance_id}` to a unique connection token.
4. Admin sends the **hello** frame first:

   ```json
   {
     "v": 1, "type": "provider.hello", "id": "", "reply_to": null,
     "epoch": 4,
     "ts": "...",
     "payload": {"epoch": 4, "server_time": "..."}
   }
   ```

   The provider client reads `payload.epoch` from the first frame it
   receives and stamps all subsequent frames with it.
5. Admin updates the DB row: `websocket_connected=true`, `epoch=<new>`,
   `last_seen=now`. `instance_status` stays `registering` until the
   provider emits its first `provider.status` — **except** when the
   definition is a shell (Phase 14: no authored `backend_config`): the
   admin then sets `awaiting_config` immediately, and provider-reported
   `running`/`initializing`/`registering` statuses are coerced back to
   `awaiting_config` until the config push clears it (§4).
6. Presence: `im:ws:presence:{instance_id}` (TTL 60s) is refreshed on
   every accepted inbound frame and on every outbound command/pong. A
   background sweep (every `INSTANCE_SWEEP_INTERVAL_SECONDS`) marks any
   DB row with `websocket_connected=true` whose presence key has expired
   as `disconnected` — the safety net for missed disconnects and admin
   restarts.

### Epoch fencing rule

Every non-hello frame carries the sender's current `epoch`. The admin
drops (logs and ignores) any inbound frame where
`frame.epoch != connection.epoch` — lower means it came from a stale,
already-replaced connection; higher indicates a protocol violation.
This prevents a zombie socket from a previous connection from mutating
state or consuming command replies after the provider reconnected.

### Reconnect

The provider uses exponential backoff (1s → 30s max). Each accepted
reconnect gets a strictly greater epoch; old sockets are closed with
code **4409** (conflict) if still present in the admin registry. On
disconnect the admin clears `im:ws:owner:{id}` (only if still owned by
the dying connection) and marks the DB row
`websocket_connected=false, instance_status="disconnected"`. The epoch
key is **not** deleted.

**Config self-heal on connect (Phase 9):** after marking the row
connected, the admin compares `ProviderInstance.config_fingerprint`
with `sha256(canonical(definition.backend_config))`; on mismatch it
pushes `provider.config.update` **to that instance only** (never the
definition-wide fan-out — current siblings are not churned) in a
background task (same helper the presence sweep calls), so a config
PATCHed while the provider was down is applied automatically on
reconnect. A per-instance in-flight guard (module-level set in
`app/services/config_update.py`; the admin is a single uvicorn worker,
so an in-process guard is authoritative — always cleared in a `finally`)
ensures a slow (300s) apply is never re-pushed by subsequent 30s sweeps
into the same provider's command queue; a skipped heal retries on the
next sweep.

### Close codes

| Code | Meaning |
| --- | --- |
| 4401 | Authentication failed (missing/bad secret, unknown instance). |
| 4409 | Connection replaced by a newer connection for the same instance. |
| 1013 | Admin not ready (Redis unavailable at auth time). |

---

## 4. Frame catalog

### Provider → Admin (events)

Events do **not** require an ack; the admin persists/acts on them.

| type | payload | status | handling |
| --- | --- | --- | --- |
| `ping` | `{}` | Live | Admin replies `pong` with `reply_to = ping.id` (reply direction: admin → provider). |
| `provider.status` | `{"instance_status": "...", "backend_status": "...", "error_message": "..."}` | Live | Updates `ProviderInstance.instance_status` / `backend_status` / `error_message`, `last_seen=now`. |
| `backend.status` | `{"backend_status": "..." (or "status"), "error_message": "..."}` | Live | Updates `ProviderInstance.backend_status` (+ optional error), `last_seen=now`. |
| `metrics.machine` | machine-level snapshot (vram/gpu_usage/os_ram/cpu/storage) for assigned resources only | Live | Persisted to Redis `im:metrics:machine:{machine_uid}`; refreshes the ownership lease. Emitted by `MachineMetricsEmitter` while owned. |
| `download.progress` | `{filename, progress_percent, bytes_downloaded, total_bytes, speed_mbps, phase}` | Live | Emitted by `provider_lib.downloader` (throttled). |
| `backend.logs` | `{"lines": [{"ts": iso, "stream": "stdout"\|"stderr", "text": "..."}], "dropped": int}` | **Live (Phase 13)** | Captured backend subprocess stdout/stderr. Provider tees the spawn pipes into a bounded ring (`provider_lib.log_ring.CursorLogRing`, `LOG_RING_LINES` default 2000); flushes throttled batches (~1s / max 100 lines per frame) while connected; `dropped` is the cumulative lost-line counter since connect. Admin appends to Redis `im:logs:backend:{instance_id}` (capped ~2000, TTL 1h). Catch-up after reconnect via `backend.logs.get`. |
| `metrics.inference` | available/max slots, token speed, prompt-processing speed, in-flight | Reserved (defined in `FrameKind`, not yet emitted; the admin does not handle it yet) | Always-on per-instance telemetry; lands with Phase 7/8. |
| `backend.boot_requested` | `{}` | Reserved | Observability only; boot is admin-driven (scheduler sends `backend.start`). |
| `provider.logs` | `{"lines": [{"ts": iso, "stream": "stdout", "text": "..."}], "dropped": int}` | **Live (Phase 13)** | The provider process's own logger output, captured via a ring-buffer handler and flushed with the same throttle as `backend.logs`. Admin stores in Redis `im:logs:provider:{instance_id}`. |
| `backend.metadata` | model metadata scraped from the backend | Reserved | Not used: discovered metadata travels in the `provider.config.update` **ack detail** (`model_metadata`) and is persisted by the admin there (Phase 9). |

`instance_status` ∈
`awaiting_config|registering|initializing|running|unhealthy|error|disconnected`
(`InstanceStatusValue`). **`awaiting_config` is admin-owned** (Phase 14):
the provider never emits it; while a definition is unconfigured the admin
coerces provider-reported running/initializing/registering statuses to it.
`backend_status` ∈ `stopped|initializing|starting|running|in_use|stopping|error`
(`BackendStatusValue`).

Unknown event types are logged and ignored (forward compatible).

### Admin → Provider (commands)

Commands carry an `id`; the provider **must** reply with a frame of
`type="ack"` and `reply_to=<command id>` whose payload validates as:

```json
{"ok": true, "error": null, "detail": { }}
```

The admin's `ConnectionManager.send_command(instance_id, type, payload,
timeout)` sends the command and awaits the matching ack (default 30s).

| type | payload | status | notes |
| --- | --- | --- | --- |
| `backend.start` | `{}` | Live | Provider awaits `BackendLifecycle.start()` (STOPPED → STARTING → RUNNING with `backend.status` per transition) and acks ok with `detail.capacity`. Phase 14: NAKs `{"ok": false, "error": "no_config", "detail": {"step": "validate"}}` when no config has been applied (shell definition) — defense-in-depth behind the admin's own gates. |
| `backend.stop` | `{}` | Live | Provider awaits `BackendLifecycle.stop()` (→ STOPPING → STOPPED, emitted) and acks ok. **No forced drain**: in-flight streams are not cancelled — their producer tasks release their slots as the upstream closes (the client may see the stream end early). Use `provider.config.update` when drain semantics matter (it refuses to stop while `in_use`). |
| `backend.restart` | `{}` | Reserved | Stop + start. |
| `provider.initialize` | `{}` | Reserved | Run the full init lifecycle (see `ARCHITECTURE.md` §8 / Phase 9). |
| `provider.config.update` | `{"backend_config": {...}, "config_fingerprint": "<sha256 hex>", "idle_timeout_seconds": int, "capacity": int}` | **Live (Phase 9)** | Apply a new definition config in place. **Phase 14: never pushed for a shell definition** (no authored config — the admin gates every push/heal on `backend_config IS NOT NULL`). Provider order matters: (1) **capacity adopt** — a differing `capacity` is applied to `lifecycle.capacity` immediately (enforced at the provider; no restart needed); (2) **noop** — received fingerprint == applied → ack `{"ok": true, "detail": {"noop": true, "capacity_adopted": bool, ...}}`, always safely ackable even under load; (3) **drain** — `lifecycle.stop_if_idle()` checks busy and transitions STOPPING under the same lifecycle lock with no intervening await (a concurrent `acquire_slot()` can never slip in and get SIGTERMed mid-stream); busy → NAK `{"ok": false, "error": "backend_in_use", "detail": {"step": "drain", "retry_after": 10, "in_flight": N}}`; (4) otherwise emits `provider.status initializing`, clears the old fingerprint's prompt cache, `driver.apply_config`, starts (artifact downloads stream `download.progress`), scrapes `list_models()`. **Ok ack detail:** `{"config_fingerprint", "capacity", "model_metadata": [...], "prompt_cache_deleted", "prompt_cache_bytes_freed"}` — the admin persists the echoed fingerprint on the instance and `model_metadata` (`{"models": [...]}`) on the definition. **Failure NAK:** `{"ok": false, "error": "<msg>", "detail": {"step": "validate|drain|cache_clear|apply_config|start"}}`. `idle_timeout_seconds` is carried in the payload for observability only — the provider does not consume it (idle reaping is admin-side, `InferenceScheduler._idle_reaper`). Admin retry policy: 300s per-instance timeout, instances pushed concurrently; only `backend_in_use` is retried (3 attempts, 10s apart); other failures reported per-instance without rolling back the admin row. A stale fingerprint on (re)connect is auto-healed by the connect path and the presence sweep (**stale instance only**, guarded against duplicate in-flight pushes). See `provider/README.md`. |
| `metrics.assign` | resource list (GPU UUIDs / categories) | Live (Phase 5) | Grant machine-level metrics ownership; admin `assign_ownership` sends it, provider starts its emitter. Carries epoch. |
| `metrics.unassign` | resource list | Provider handler live; admin send not yet wired (ownership currently lapses via Redis lease expiry) | Revoke machine-level metrics ownership. |
| `metrics.category.start` | category name | Reserved | Enable a `METRICS_CATEGORIES` category. |
| `cache.clear` | `{"dry_run": bool = false, "force": bool = false}` | **Live (Phase 9)** | **Prompt-cache files only** — never model files. Deletes `CACHE_DIR/prompt_cache` plus the provider's engine cache dirs (`extra_cache_dirs`: gufo `CACHE_DIR/<MACHINE_UID>`, halogen-flash `CACHE_DIR/halogen-flash`). **Refused while the backend is `in_use`** (NAK `{"ok": false, "error": "backend_in_use", "detail": {"step": "drain", ...}}`) unless `force: true` — clearing engine caches under live streams can cause I/O errors; `dry_run` never touches files and is always allowed. Ack detail: `{"dry_run", "deleted": [...], "bytes_freed": N}`. Admin surface: `POST /admin/api/instances/{id}/cache/clear` (body `{dry_run?, force?}`). |
| `storage.prune_unused` | `{"dry_run": bool = false}` | **Live (Phase 9)** | Delete `MODELS_DIR` files not referenced by the driver's current `resolved_artifacts` set (main/mmproj/draft/tokenizer/NPU pins; a referenced directory protects its subtree). **Refuses** (NAK `no_resolved_artifacts`) when the driver has not resolved its set yet — never deletes blind. Ack detail: `{"dry_run", "deleted": [...], "bytes_freed": N, "kept": [...]}`. Admin surface: `POST /admin/api/instances/{id}/storage/prune`. |
| `backend.logs.get` | `{"kind": "backend"\|"provider"\|"all", "since": <ring seq or null>}` | **Live (Phase 13)** | Ask the provider for its current ring buffer (catch-up after an admin restart / long disconnect; the Redis tail may be older than the ring). Ack detail: `{"lines": [...], "dropped": int, "seq": int}` — `since` resumes at the ring sequence; omitting it returns the whole buffer. For `kind=all` the backend and provider rings share one `SeqCounter` (installed by `install_log_streaming`), so a single `since`/`seq` cursor resumes both correctly. |

**Two unrelated log seq spaces (Phase 13).** The `seq` in the
`backend.logs.get` ack is the **provider-side ring sequence** (a shared
`SeqCounter` across the backend+provider rings, starting at 0 per
provider process/connect lifetime). The `since`/`cursor` on the admin
REST `GET /admin/api/instances/{id}/logs` is the **admin ingest
sequence** (`im:logs:seq:{instance_id}`, Redis `INCRBY`, 1-based).
These are independent counters and MUST NOT be passed interchangeably by
a client. The admin `read_logs` response also carries `gap`/`oldest_seq`
(eviction below the retained window) and `unseen_total` (pre-trim count
of `seq > since`) so the UI can warn when lines were skipped between
polls rather than silently dropping them.

The provider lifecycle, `BackendDriver` interface, slot admission, and the
release-on-upstream-close invariant are documented in `provider/README.md`.

### Admin → Provider (informational)

| type | payload | notes |
| --- | --- | --- |
| `provider.hello` | `{"epoch": int, "server_time": iso}` | First frame after a successful authenticated accept. |
| `pong` | `{}` | Reply to provider `ping` (`reply_to` set). |

---

## 5. Redis key summary

| Key | Contents | Lifetime |
| --- | --- | --- |
| `im:ws:secret:{instance_id}` | Per-instance WS secret (plaintext, trusted LAN). | 30 days TTL; rewritten on each registration. |
| `im:ws:epoch:{instance_id}` | Monotonic connection epoch (counter). | Permanent (never deleted). |
| `im:ws:owner:{instance_id}` | Connection token of the currently accepted socket. | Deleted on disconnect or when superseded. |
| `im:ws:presence:{instance_id}` | Liveness marker (timestamp). | 60s TTL, refreshed by traffic; absence ⇒ sweep marks disconnected. |
| `im:logs:backend:{instance_id}` | Backend log tail (Phase 13; JSON lines, newest left via LPUSH+LTRIM). | ~2000-line cap, 1h TTL. |
| `im:logs:provider:{instance_id}` | Provider log tail (Phase 13; same shape). | ~2000-line cap, 1h TTL. |
