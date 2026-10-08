# Redis Keys

The admin uses Redis for the scheduler, live metrics, and WebSocket
connection bookkeeping. PostgreSQL is the source of truth for configuration
and stored responses; Redis holds only fast-changing runtime state that can
be rebuilt from Postgres + a fresh provider registration.

All keys are namespaced under `im:` (Inference Matrix). TTLs are noted per
key. `{...}` are placeholders.

> **Phase 16 (machine-scoped agents).** The WebSocket is keyed by the
> **`ProviderAgent` primary-key uuid** (`str(agent.id)`) — the value the admin
> returns as `agent_id` in the registration response and the agent presents on
> the WS query string. This is **not** the operator-supplied `AGENT_ID` string.
> Throughout this doc `{agent_id}` means that PK uuid. `{machine_uid}` is the
> `Machine.uid`; `{instance_id}` is a `ProviderInstance` (one backend) PK uuid.
> The authoritative source is `app/services/redis_keys.py`.

## Scheduler

The Phase 6 scheduler runs in the **single** admin uvicorn worker and keeps
the **authoritative FIFO in-process** (a per-alias waiter `deque` guarded by
an `asyncio.Lock` + `asyncio.Condition`). The Redis keys below are an
**observability mirror** and the `sched:lock` admission contract; they are
not the queue authority. A future cross-worker design can move the FIFO
into Redis behind the same `acquire`/`release` interface without changing
this key layout.

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:sched:queue:{alias}` | List (mirror of in-process deque) | none | FIFO of queued request ids for a provider alias, rewritten from the authoritative in-process deque on every change. `alias` = ProviderDefinition.alias. |
| `im:sched:wait:{req_id}` | Hash | 1h | Per-request wait state: `position`, `enqueued_at`, `status` (`queued`/`admitted`). |
| `im:sched:active:{alias}` | Set | none | Request ids currently admitted (holding a slot) for the alias. Cardinality ≤ capacity. Mirrors the in-process active map. |
| `im:sched:lock:{alias}` | String (SET NX PX) | 5s | Short admission lock so concurrent admins serialize slot assignment per alias. Honored around boot/admission; with a single worker contention is rare and a lock miss falls back to in-process authority. |

Admission (Phase 4/6): the head waiter is admitted when it reaches the
front of the in-process FIFO **and** a slot is free — `active` count <
capacity **and**, if the target needs a boot, the target machine has
enough free VRAM (see `im:vram:*`). An already-booted target admits on
capacity alone (its VRAM is already held). A stopped candidate backend is
booted (`backend.start`, acked after the provider reaches running) before
admission; boot failure falls through to the next candidate. If the boot
doesn't fit, **idle different-alias backends on the machine are evicted**
(`backend.stop`, LRU by `last_request_at`) under the same lock; if that
still doesn't free enough, the head waits (up to `queue_timeout`) rather
than busy-polling. On completion the request is removed from `active`;
the booted instance's **VRAM hold is NOT released** (the backend is
still loaded — only an actual stop clears it).

**Agent-scoped admission (Phase 16).** Candidates for an alias are the
`ProviderInstance` backends whose definition is placed on a **connected**
agent. Before booting a backend of type `T` on agent `A`, the scheduler counts
`A`'s backends of type `T` already `running`/`in_use`; at
`ProviderType.max_running_backends` (e.g. `1` for halogen-flash) it **hot-swaps**
(evicts the LRU running backend of that type on that agent, subject to the same
busy/zero-active-slot guard as VRAM eviction) before admitting the new one. The
whole evict+boot runs under one `im:sched:lock:{alias}` acquisition plus the
in-process `_evicting` set.

### In-process vs Redis boundary

- **In-process (authority):** waiter FIFO order, active-slot map
  (`request_id -> instance`), the per-booted-instance VRAM hold ledger
  (`self._booted: instance_id -> (machine_uid, vram_bytes)`, so a
  lagging DB `backend_status` mirror never causes a redundant
  `backend.start` and idle backends stay counted), the set of instances
  with an in-flight eviction/stop (`self._evicting`, guards victims
  against concurrent acquires on other aliases), and the `queue_timeout`
  deadline.
- **Redis (mirror + contract):** queue depth, active set, per-request wait
  state, the `sched:lock` admission lock, and the `im:vram:used` ledger.
  All mirror writes are best-effort and TTL-bounded so a crashed admin
  self-heals. `release` pops the in-process slot synchronously (immediate
  `active_count` correctness) and shields only the wake-up + Redis cleanup
  so they land even under caller cancellation (client disconnect).
- **Eviction / idle reaper (Phase 6, implemented):** see
  `ARCHITECTURE.md` §6. Eviction stops idle different-alias backends
  LRU-first to free VRAM for a needed boot; the reaper stops backends
  idle past `idle_timeout_seconds` (0 = never) every
  `IDLE_REAPER_INTERVAL_SECONDS` (15s default) and refreshes the
  `im:vram:used` TTL for this process's booted holds. Both clear the
  ledger entry + mirror on a successful stop. Admission never lands on an
  instance with an in-flight stop: `_try_admit` skips any candidate in
  `self._evicting`. An out-of-band stop (admin UI, provider side,
  crash) is reconciled event-driven: the connection manager calls
  `scheduler.note_backend_stopped` on a `stopped`/`error` status frame,
  dropping the stale hold + mirror field immediately.

## VRAM / Machine Resource

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:vram:used:{machine_uid}` | Hash: `{instance_id}` → bytes | 60s | VRAM held **per booted backend** on the machine — a loaded backend keeps its weights resident between requests, so the entry is written on **boot** (`_mark_booted`) and removed on **stop** (idle reaper, eviction, or an out-of-band `backend.status`/`provider.status` frame reporting `stopped`/`error`, which the connection manager prunes via `scheduler.note_backend_stopped`), *not* per request. Refreshed (value + TTL) on every boot/adopt; the idle reaper re-EXPIREs it each tick while the backend stays booted. If the admin crashes the key lapses in 60s and self-heals: the ledger is rebuilt from the DB (`backend_status` running/in_use × `vram_required_bytes`) on the next admission. |

**Semantic change (Phase 6 close-out).** The earlier ledger recorded one
entry per *request slot* and `hdel`-ed it on `release`, so a
running-but-idle backend appeared to hold 0 VRAM. That is physically
false (a booted llama-server keeps its weights) and made idle-backend
eviction impossible — there was nothing recorded to free. VRAM is now
accounted per booted backend; `release` frees only the capacity slot.
`held_on(machine)` merges the mirror with `self._booted` and the DB's
loaded-instance states (max per instance) so no hold is under-counted.

Free VRAM for a machine = `total - sum(used)` (excluding the requesting
target's own hold), where `total` is read from `Machine.total_vram_bytes`
in Postgres (there is **no** `im:vram:total` Redis key). Eviction picks a
running-but-idle **different-alias** backend to stop when a boot needs the
space.

## Live Metrics

Machine hardware metrics split by category (Phase 17). **GPU categories**
(`vram`/`gpu_usage`) are emitted by **every** connected agent that declares
them, each filtered to its assigned GPUs (`ASSIGNED_GPU_UUIDS`; empty =
implicit visible==owned), and merged per-GPU-uuid by the admin on read.
**Machine-wide categories** (`os_ram`/`cpu`/`storage`) are visible from any
container and stay single-owner: the admin assigns ownership with a Redis
lease.

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:metrics:owner:{machine_uid}` | String (`agent_id`) | 30s | Which **agent** is the **machine-wide** metrics reporter (`os_ram`/`cpu`/`storage`). Value is the `ProviderAgent` PK uuid. SET NX at WS connect only when the agent declares a machine-wide category (or sweep reassignment); refreshed when the owner's `metrics.machine` events arrive; deleted on disconnect/stale sweep. GPU categories are NOT gated by this lease. |
| `im:metrics:machine:{machine_uid}` | String (JSON snapshot) | 30s | Latest **machine-wide** owner snapshot (`os_ram`/`cpu`/`storage` + `owner_agent_id`) stored from the owner's `metrics.machine` event. Non-owner machine-wide sections are dropped. |
| `im:metrics:machine:{machine_uid}:agent:{agent_id}` | String (JSON partial) | 30s | Per-agent **GPU** partial (`vram`/`gpu_usage`), written by **every** agent that reports GPU categories regardless of ownership; refreshed each frame. Merged per-GPU-uuid on read (`read_machine_metrics`); a dead agent's partial self-expires with its TTL. |
| `im:metrics:cats:{agent_id}` | String (JSON list) | none | Agent-declared metrics categories, written at registration, read at ownership assignment (`metrics.assign` payload). |

Flow (`app/services/metrics_service.py`): every agent's emitter loop runs from
connect and writes its GPU partial (`im:metrics:machine:{uid}:agent:{id}`) on
each `metrics.machine` frame regardless of ownership. For the machine-wide
lease: agent connects → admin reads `im:metrics:cats:{agent_id}` → if it
declares a machine-wide category,
`SET im:metrics:owner:{machine_uid} <agent_id> NX EX 30` → sends
`metrics.assign {machine_uid, categories}` over the WS (rollback: lease
released if the command fails). The owner's frames refresh the lease and write
`im:metrics:machine:{uid}`. On
disconnect (`ws.py` finally) or stale sweep (`presence_sweep.py`) the
lease is released; the sweep also reassigns ownerless machines that
still have connected agents. Reads (`read_machine_metrics`, served by
`GET /admin/api/machines/{id}/metrics`) union the live GPU partials per uuid
and overlay the owner machine-wide snapshot.

> **Inference metrics are not in Redis.** The `metrics.inference` frame kind is
> defined but reserved (not yet emitted/handled — Phase 7/8); when it lands it
> is always-on per backend and never deduped.

## WebSocket / Connection Bookkeeping

All keyed by the `ProviderAgent` PK uuid (`{agent_id}`).

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:ws:secret:{agent_id}` | String | 30d | Per-**agent** WS secret (plaintext, trusted LAN; minted at registration, never in Postgres). Read at WS auth. |
| `im:ws:epoch:{agent_id}` | String (int) | none | Monotonic connection epoch. INCR when the admin accepts a new agent socket. Frames with a stale epoch are dropped (fencing). Mirrored on `ProviderAgent.epoch`. |
| `im:ws:owner:{agent_id}` | String (connection token) | none | Connection token of the currently accepted socket. SET by the accepting admin; deleted on disconnect (only if still owned by the dying connection). Lets a future multi-worker admin route/verify without a schema change. |
| `im:ws:presence:{agent_id}` | String (timestamp) | 60s | Liveness heartbeat marker, refreshed on every accepted inbound frame and outbound command/pong. Missing/expired ⇒ the sweep marks the agent `disconnected` (and its backends unschedulable). |

For a single-worker admin (current), `ws:owner` is trivially the one node;
it exists so a future multi-worker admin can route/verify without a schema
change.

## Logs (Phase 13)

Backend/provider log tails are **Redis-only** — ephemeral ops telemetry,
never persisted to Postgres. Newest lines are pushed to the left so
`LRANGE 0 N` reads the tail. Backend logs are **per backend** (keyed by
`instance_id`); provider logs are **per agent** (keyed by `agent_id`).

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:logs:backend:{instance_id}` | List (JSON-line entries) | 1h | Captured backend subprocess stdout/stderr. Each entry: `{"seq", "ts", "stream": "stdout"\|"stderr", "text"}`. Appended in batches from `backend.logs` events via LPUSH + LTRIM to a ~2000-line cap. Served by `GET /admin/api/instances/{id}/logs?kind=backend&since=&limit=`. |
| `im:logs:provider:{agent_id}` | List (JSON-line entries) | 1h | The agent process's own logger output (same entry shape), from `provider.logs` events. Same cap/TTL. |
| `im:logs:seq:{agent_id}` | String (int) | 1h | Shared monotonic ingest cursor (`INCRBY`) across both kinds on one agent, so a single `kind=all` cursor orders the merged tail correctly. 1-based. |
| `im:logs:dropped:{kind}:{id}` | String (int) | 1h | Latest provider-reported dropped-line counter per kind (`{id}` = `instance_id` for backend, `agent_id` for provider). |

- **Cursor**: `since` in the read endpoint is the **admin ingest seq**
  (`im:logs:seq:{agent_id}`), *not* the provider-side ring sequence in the
  `backend.logs.get` ack — the two are unrelated spaces and must never be mixed
  by a client. The UI polls with the last seen cursor to tail; the response also
  carries `gap`/`oldest_seq`/`unseen_total` so the UI can warn about dropped or
  skipped lines.
- **Flush**: On Redis flush the tail is empty until the next provider
  flush; the provider ring buffer (and `backend.logs.get`) is the catch-up
  source. Logs are best-effort and never block the request path.

## Schema consensus state (Phase 12)

**Not in Redis.** The pending schema, its fingerprint, and the voter list
live on the `provider_types` Postgres row (`pending_schema`,
`pending_fingerprint`, `pending_voters`, `status`) because the admin is a
single writer and the consensus state must survive a Redis flush. Per-
**agent** `reported_schema_fingerprint` is likewise a Postgres column
(`ProviderAgent.reported_schema_fingerprint`). Registration-time consensus
evaluation reads/writes that row directly (see `docs/ws-protocol.md` §2).

## Notes

- **Rebuild**: If Redis is flushed, running inference degrades until
  providers re-heartbeat (VRAM/metrics repopulate) and the queue is
  repopulated from new requests. Config in Postgres is unaffected.
- **TTL discipline**: Runtime keys carry TTLs so a crashed provider or
  dropped connection self-heals rather than leaking VRAM/active slots.
  Queue and active-set keys have no TTL and are managed explicitly.
- **No cross-key transactions**: Admission uses the `sched:lock` per-alias
  rather than MULTI/EXEC, so behavior is identical on Redis standalone and
  Redis Cluster.
