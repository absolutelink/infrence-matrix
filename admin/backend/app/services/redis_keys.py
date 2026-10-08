"""Redis key layout for provider WebSocket presence, secrets, epochs, the
scheduler, and the VRAM ledger.

WS keys are prefixed ``im:ws:``. Redis (not Postgres) is the source of truth
for WS liveness and for the per-agent secret, because the secret is short
-lived operational state returned exactly once to the agent at
registration (trusted-LAN design — see docs/ws-protocol.md).

Phase 16: the WS socket is machine-scoped per agent, so these keys are keyed
by the ``ProviderAgent`` primary-key uuid (``str(agent.id)``) — NOT the
operator-supplied ``agent_id`` string. The admin resolves the operator
``agent_id`` to its PK row at registration and uses the PK everywhere below.

  im:ws:secret:{agent_id}       per-agent WS secret (plaintext, long TTL;
                                written at registration, read at WS auth)
  im:ws:epoch:{agent_id}        monotonic connection epoch (INCR per accepted
                                WS connection; never deleted)
  im:ws:owner:{agent_id}        admin connection token for the currently
                                accepted socket (deleted on disconnect)
  im:ws:presence:{agent_id}     TTL key refreshed on every frame received /
                                pong sent; absence => connection is dead

Machine metrics ownership (Phase 5; per-GPU split in Phase 17), prefixed
``im:metrics:``:

  im:metrics:owner:{machine_uid}  agent_id of the single provider agent
                                  reporting the MACHINE-WIDE metrics
                                  (os_ram/cpu/storage) for that machine
                                  (SET NX, TTL METRICS_OWNER_TTL_SECONDS;
                                  refreshed on each owner metrics.machine
                                  receipt). GPU categories are NOT gated
                                  by this lease.
  im:metrics:machine:{machine_uid} latest machine-wide snapshot JSON
                                  (owner-gated; TTL METRICS_OWNER_TTL_SECONDS)
  im:metrics:machine:{machine_uid}:agent:{agent_id}
                                  per-agent GPU partial JSON (vram/gpu_usage
                                  + assigned_gpus), written by EVERY agent
                                  that reports GPU categories regardless of
                                  ownership (TTL METRICS_OWNER_TTL_SECONDS).
                                  Merged per-GPU-UUID on read; a dead agent's
                                  partial self-expires with its TTL.
  im:metrics:cats:{agent_id}      JSON list of the agent's declared
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
# Per-agent GPU partials live NESTED under the machine snapshot namespace:
# ``{METRICS_MACHINE_PREFIX}:{machine_uid}:agent:{agent_id}`` (Phase 17).
METRICS_AGENT_PARTIAL_INFIX = "agent"

SCHED_QUEUE_PREFIX = "im:sched:queue"
SCHED_WAIT_PREFIX = "im:sched:wait"
SCHED_ACTIVE_PREFIX = "im:sched:active"
SCHED_LOCK_PREFIX = "im:sched:lock"

VRAM_USED_PREFIX = "im:vram:used"

LOGS_BACKEND_PREFIX = "im:logs:backend"
LOGS_PROVIDER_PREFIX = "im:logs:provider"
# Per-instance ingest sequence counter (shared across both kinds so
# kind=all merges by a single monotonic cursor) and the latest
# provider-reported dropped-line count per kind.
LOGS_SEQ_PREFIX = "im:logs:seq"
LOGS_DROPPED_PREFIX = "im:logs:dropped"

# Phase 13 log tail bounds (docs/ws-protocol.md §5).
LOGS_CAP = 2000
LOGS_TTL_SECONDS = 3600

# TTL for the metrics ownership lease and the machine snapshot. The owner
# must refresh it by emitting metrics.machine more often than this; on
# expiry another instance on the same machine can take over.
METRICS_OWNER_TTL_SECONDS = 30

# TTLs for the scheduler mirror keys (see docs/redis-keys.md).
SCHED_WAIT_TTL_SECONDS = 3600
SCHED_LOCK_TTL_MS = 5000
VRAM_USED_TTL_SECONDS = 60


def secret_key(agent_id: str) -> str:
    return f"{SECRET_PREFIX}:{agent_id}"


def epoch_key(agent_id: str) -> str:
    return f"{EPOCH_PREFIX}:{agent_id}"


def owner_key(agent_id: str) -> str:
    return f"{OWNER_PREFIX}:{agent_id}"


def presence_key(agent_id: str) -> str:
    return f"{PRESENCE_PREFIX}:{agent_id}"


def metrics_owner_key(machine_uid: str) -> str:
    return f"{METRICS_OWNER_PREFIX}:{machine_uid}"


def metrics_machine_key(machine_uid: str) -> str:
    return f"{METRICS_MACHINE_PREFIX}:{machine_uid}"


def metrics_agent_partial_prefix(machine_uid: str) -> str:
    """Prefix for every per-agent GPU partial on a machine (for SCAN)."""
    return f"{METRICS_MACHINE_PREFIX}:{machine_uid}:{METRICS_AGENT_PARTIAL_INFIX}:"


def metrics_agent_partial_key(machine_uid: str, agent_id: str) -> str:
    return f"{metrics_agent_partial_prefix(machine_uid)}{agent_id}"


def metrics_cats_key(agent_id: str) -> str:
    return f"{METRICS_CATS_PREFIX}:{agent_id}"


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


def logs_backend_key(instance_id: str) -> str:
    return f"{LOGS_BACKEND_PREFIX}:{instance_id}"


def logs_provider_key(instance_id: str) -> str:
    return f"{LOGS_PROVIDER_PREFIX}:{instance_id}"


def logs_seq_key(instance_id: str) -> str:
    return f"{LOGS_SEQ_PREFIX}:{instance_id}"


def logs_dropped_key(instance_id: str, kind: str) -> str:
    return f"{LOGS_DROPPED_PREFIX}:{kind}:{instance_id}"
