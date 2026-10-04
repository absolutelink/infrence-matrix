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
capacity **and** the target machine has enough free VRAM (see `im:vram:*`).
A stopped candidate backend is booted (`backend.start`, acked after the
provider reaches running) before admission; boot failure falls through to
the next candidate. If nothing fits, the head waits (up to `queue_timeout`)
rather than busy-polling. On completion the request is removed from
`active` and its `im:vram:used` entry is dropped (unless another active
slot still occupies the same instance).

### In-process vs Redis boundary

- **In-process (authority):** waiter FIFO order, active-slot map
  (`request_id -> instance + vram`), the set of instances this worker has
  booted (`self._booted`, so a lagging DB `backend_status` mirror never
  causes a redundant `backend.start`), and the `queue_timeout` deadline.
- **Redis (mirror + contract):** queue depth, active set, per-request wait
  state, the `sched:lock` admission lock, and the `im:vram:used` ledger.
  All mirror writes are best-effort and TTL-bounded so a crashed admin
  self-heals. `release` pops the in-process slot synchronously (immediate
  `active_count` correctness) and shields only the wake-up + Redis cleanup
  so they land even under caller cancellation (client disconnect).
- **Eviction / idle reaper (Phase 6):** not implemented — see
  `TODO(phase6-eviction)` and `TODO(phase6-idle-reaper)` in
  `app/services/scheduler.py`. When VRAM is insufficient the request
  waits instead of evicting an idle different-alias instance.

## VRAM / Machine Resource

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `im:vram:total:{machine_uid}` | String (int bytes) | 60s | Total VRAM budget, mirrored from Machine.total_vram_bytes; refreshed on heartbeat. |
| `im:vram:used:{machine_uid}` | Hash: `{instance_id}` → bytes | 60s | VRAM currently held by each instance on the machine. Expired entries are swept, so a dead instance frees its VRAM without an explicit release. |

Free VRAM for a machine = `total - sum(used)`. Eviction (Phase 5) picks a
running-but-idle instance to stop when a higher-priority request needs the
space.

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
