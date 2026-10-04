"""Redis key layout for provider WebSocket presence, secrets, epochs, the
scheduler, and the VRAM ledger.

WS keys are prefixed ``im:ws:``. Redis (not Postgres) is the source of truth
for WS liveness and for the per-instance secret, because the secret is short
-lived operational state returned exactly once to the provider at
registration (trusted-LAN design — see docs/ws-protocol.md).

  im:ws:secret:{instance_id}    per-instance WS secret (plaintext, long TTL;
                                written at registration, read at WS auth)
  im:ws:epoch:{instance_id}     monotonic connection epoch (INCR per accepted
                                WS connection; never deleted)
  im:ws:owner:{instance_id}     admin connection token for the currently
                                accepted socket (deleted on disconnect)
  im:ws:presence:{instance_id}  TTL key refreshed on every frame received /
                                pong sent; absence => connection is dead

Machine metrics ownership (Phase 5), prefixed ``im:metrics:``:

  im:metrics:owner:{machine_uid}  instance_id of the single provider
                                  instance reporting machine-level
                                  metrics for that machine (SET NX, TTL
                                  METRICS_OWNER_TTL_SECONDS; refreshed
                                  on each metrics.machine receipt)
  im:metrics:machine:{machine_uid} latest machine-level snapshot JSON
                                  (TTL METRICS_OWNER_TTL_SECONDS)
  im:metrics:cats:{instance_id}   JSON list of the instance's declared
                                  metrics categories (written at
                                  registration, read at assignment)

Scheduler + VRAM ledger (Phase 6), prefixed ``im:sched:`` / ``im:vram:``.
The authoritative FIFO lives in the admin process (single uvicorn worker);
these keys are the observable mirror documented in docs/redis-keys.md:

  im:sched:queue:{alias}   List: queued request ids (mirror; LTRIM'd to
                           the in-process depth on every change)
  im:sched:wait:{req_id}   Hash: position / enqueued_at / status (TTL 1h)
  im:sched:active:{alias}  Set: admitted request ids, cardinality <= capacity
  im:sched:lock:{alias}    SET NX PX 5s admission lock (contract honored)
  im:vram:used:{machine_uid}  Hash {instance_id} -> bytes held; TTL-bounded
                           so a crashed admin self-heals the ledger
"""

SECRET_PREFIX = "im:ws:secret"
EPOCH_PREFIX = "im:ws:epoch"
OWNER_PREFIX = "im:ws:owner"
PRESENCE_PREFIX = "im:ws:presence"

METRICS_OWNER_PREFIX = "im:metrics:owner"
METRICS_MACHINE_PREFIX = "im:metrics:machine"
METRICS_CATS_PREFIX = "im:metrics:cats"

SCHED_QUEUE_PREFIX = "im:sched:queue"
SCHED_WAIT_PREFIX = "im:sched:wait"
SCHED_ACTIVE_PREFIX = "im:sched:active"
SCHED_LOCK_PREFIX = "im:sched:lock"

VRAM_USED_PREFIX = "im:vram:used"

# TTL for the metrics ownership lease and the machine snapshot. The owner
# must refresh it by emitting metrics.machine more often than this; on
# expiry another instance on the same machine can take over.
METRICS_OWNER_TTL_SECONDS = 30

# TTLs for the scheduler mirror keys (see docs/redis-keys.md).
SCHED_WAIT_TTL_SECONDS = 3600
SCHED_LOCK_TTL_MS = 5000
VRAM_USED_TTL_SECONDS = 60


def secret_key(instance_id: str) -> str:
    return f"{SECRET_PREFIX}:{instance_id}"


def epoch_key(instance_id: str) -> str:
    return f"{EPOCH_PREFIX}:{instance_id}"


def owner_key(instance_id: str) -> str:
    return f"{OWNER_PREFIX}:{instance_id}"


def presence_key(instance_id: str) -> str:
    return f"{PRESENCE_PREFIX}:{instance_id}"


def metrics_owner_key(machine_uid: str) -> str:
    return f"{METRICS_OWNER_PREFIX}:{machine_uid}"


def metrics_machine_key(machine_uid: str) -> str:
    return f"{METRICS_MACHINE_PREFIX}:{machine_uid}"


def metrics_cats_key(instance_id: str) -> str:
    return f"{METRICS_CATS_PREFIX}:{instance_id}"


def sched_queue_key(alias: str) -> str:
    return f"{SCHED_QUEUE_PREFIX}:{alias}"


def sched_wait_key(request_id: str) -> str:
    return f"{SCHED_WAIT_PREFIX}:{request_id}"


def sched_active_key(alias: str) -> str:
    return f"{SCHED_ACTIVE_PREFIX}:{alias}"


def sched_lock_key(alias: str) -> str:
    return f"{SCHED_LOCK_PREFIX}:{alias}"


def vram_used_key(machine_uid: str) -> str:
    return f"{VRAM_USED_PREFIX}:{machine_uid}"
