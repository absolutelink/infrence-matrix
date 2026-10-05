# Redis Keys

The admin uses Redis for the scheduler, live metrics, and WebSocket
connection bookkeeping. PostgreSQL is the source of truth for configuration
and stored responses; Redis holds only fast-changing runtime state that can
be rebuilt from Postgres + a fresh provider registration.

All keys are namespaced under `im:` (Inference Matrix). TTLs are noted per
key. `{...}` are placeholders.

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
| `im:vram:total:{machine_uid}` | String (int bytes) | 60s | Total VRAM budget, mirrored from Machine.total_vram_bytes; refreshed on heartbeat. |
| `im:vram:used:{machine_uid}` | Hash: `{instance_id}` → bytes | 60s | VRAM held **per booted instance** on the machine — a loaded backend keeps its weights resident between requests, so the entry is written on **boot** (`_mark_booted`) and removed on **stop** (idle reaper, eviction, or an out-of-band `backend.status`/`provider.status` frame reporting `stopped`/`error`, which the connection manager prunes via `scheduler.note_backend_stopped`), *not* per request. Refreshed (value + TTL) on every boot/adopt; the idle reaper re-EXPIREs it each tick while the backend stays booted. If the admin crashes the key lapses in 60s and self-heals: the ledger is rebuilt from the DB (`backend_status` running/in_use × `vram_required_bytes`) on the next admission. |

**Semantic change (Phase 6 close-out).** The earlier ledger recorded one
entry per *request slot* and `hdel`-ed it on `release`, so a
running-but-idle backend appeared to hold 0 VRAM. That is physically
false (a booted llama-server keeps its weights) and made idle-backend
eviction impossible — there was nothing recorded to free. VRAM is now
accounted per booted instance; `release` frees only the capacity slot.
`held_on(machine)` merges the mirror with `self._booted` and the DB's
loaded-instance states (max per instance) so no hold is under-counted.

Free VRAM for a machine = `total - sum(used)` (excluding the requesting
target's own hold). Eviction picks a running-but-idle **different-alias**
instance to stop when a boot needs the space.

## Live Metrics

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:metrics:instance:{instance_id}` | Hash | 30s | Latest per-instance snapshot: `backend_status`, `active_requests`, `prompt_tps`, `gen_tps`, `vram_bytes`, `updated_at`. |
| `im:metrics:machine:{machine_uid}` | Hash | 30s | Latest machine-level snapshot (aggregated across owned instances). |
| `im:metrics:alias:{alias}` | Hash | 30s | Latest per-model (alias) snapshot for the /v1 UI: queue depth, active, avg latency. |

Metrics are emitted by the admin (it owns inference accounting) and by
providers over the WS. Machine-level metrics ownership is assigned via the
`metrics.assign` WS command (Phase 5) so exactly one instance reports each
machine's hardware.

### Machine Metrics Ownership (Phase 5)

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:metrics:owner:{machine_uid}` | String (instance_id) | 30s | Which instance is the metrics reporter for the machine. SET NX at WS connect (or sweep reassignment); refreshed when the owner's `metrics.machine` events arrive; deleted on disconnect/stale sweep. |
| `im:metrics:machine:{machine_uid}` | String (JSON snapshot) | 30s | Latest machine-level snapshot stored from the owner's `metrics.machine` event. Non-owner events are dropped. |
| `im:metrics:cats:{instance_id}` | String (JSON list) | none | Instance-declared metrics categories, written at registration, read at ownership assignment (`metrics.assign` payload). |

Flow: instance connects → admin reads `im:metrics:cats:{id}` → if
non-empty, `SET im:metrics:owner:{machine_uid} <id> NX EX 30` → sends
`metrics.assign {machine_uid, categories}` over the WS (rollback: lease
released if the command fails). Owner emits `metrics.machine` every
`MACHINE_METRICS_INTERVAL` (default 10s), refreshing the lease. On
disconnect (`ws.py` finally) or stale sweep (`presence_sweep.py`) the
lease is released; the sweep also reassigns ownerless machines that
still have connected instances.

## WebSocket / Connection Bookkeeping

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:ws:epoch:{instance_id}` | String (int) | none | Monotonic connection epoch. Bumped when the admin accepts a new provider socket. Frames with a stale epoch are dropped (fencing). Mirrored on ProviderInstance.epoch. |
| `im:ws:owner:{instance_id}` | String (admin node id) | 30s | Which admin process currently holds the live socket. SET NX by the accepting admin; heartbeat refreshes TTL. Lets other admin nodes know not to expect this instance. |
| `im:ws:presence:{instance_id}` | String | 30s | Liveness heartbeat marker. Missing/expired ⇒ instance considered disconnected. |

For a single-worker admin (current), `ws:owner` is trivially the one node;
it exists so a future multi-worker admin can route/verify without a schema
change.

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
