# Inference Matrix — Admin ⇄ Provider Wire Protocol (Phase 3)

This document specifies the registration handshake and the provider
WebSocket protocol between a **provider instance** (hardware-local
container: `provider/mock`, later `provider/llama-cpp`, etc.) and the
**admin** (`admin/backend`, the stateless broker).

Auth model: **trusted LAN**. The registration token and the per-instance
secret gate only the provider WebSocket and the registration endpoint.
`/admin/api` and `/v1` are unauthenticated by design.

Later phases add inference-path commands (backend boot args, cache clear,
metrics assignment payloads, etc.); the frame envelope and the
registration/connection flow below are stable.

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

### Validation rules (admin side, in order)

1. `registration_token` must match an existing `ProviderDefinition`
   → otherwise **401**.
2. The definition must be `enabled` → otherwise **403**.
3. `provider_type` must equal the definition's `provider_type`
   → otherwise **409**.
4. `machine_uid` must reference a `Machine` pre-created in the admin UI
   → otherwise **404**.
5. **Version gate**: `version` must exactly equal the admin's
   `settings.VERSION` → otherwise **409** (lockstep deploy on the LAN).

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

The client derives the WS URL from `ADMIN_BASE_URL` (scheme swap
`http→ws`, `https→wss`); a definition echo may optionally carry
`admin_ws_url` to override that. Either way the client appends
`?instance_id=<uuid>` to the WS URL.

Errors are plain FastAPI `{"detail": "..."}` bodies.

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
   provider emits its first `provider.status`.
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

### Close codes

| Code | Meaning |
| --- | --- |
| 4401 | Authentication failed (missing/bad secret, unknown instance). |
| 4409 | Connection replaced by a newer connection for the same instance. |
| 1013 | Admin not ready (Redis unavailable at auth time). |

---

## 4. Frame catalog (Phase 3 scope)

### Provider → Admin (events)

Events do **not** require an ack in Phase 3; the admin persists them.

| type | payload | handling |
| --- | --- | --- |
| `ping` | `{}` | Admin replies `pong` with `reply_to = ping.id` (reply direction: admin → provider). |
| `provider.status` | `{"instance_status": "...", "backend_status": "...", "error_message": "..."}` | Updates `ProviderInstance.instance_status` / `backend_status` / `error_message`, `last_seen=now`. |
| `backend.status` | `{"backend_status": "..." (or "status"), "error_message": "..."}` | Updates `ProviderInstance.backend_status` (+ optional error), `last_seen=now`. |

`instance_status` ∈ `registering|initializing|running|unhealthy|error|disconnected`
(`InstanceStatusValue`).
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
timeout)` sends the command and awaits the matching ack.

| type | payload (Phase 3) | notes |
| --- | --- | --- |
| `backend.start` | `{}` | Provider handler awaits `BackendLifecycle.start()` (STOPPED → STARTING → RUNNING with `backend.status` per transition) and acks ok with `detail.capacity`. |
| `backend.stop` | `{}` | Provider handler awaits `BackendLifecycle.stop()` (→ STOPPING → STOPPED, emitted) and acks ok. |

Phase 4 (provider lifecycle + /v1 surface) is documented in
`provider/README.md`: the `BackendDriver` interface, the lifecycle state
machine, slot admission, and the release-on-upstream-close invariant.

Other command kinds (`backend.restart`, `provider.initialize`,
`provider.config.update`, `metrics.assign`, `cache.clear`,
`storage.prune_unused`, ...) are reserved in `FrameKind` for later
phases and are not exercised yet.

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
