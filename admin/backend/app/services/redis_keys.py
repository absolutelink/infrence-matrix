"""Redis key layout for provider WebSocket presence, secrets, and epochs.

All keys are prefixed ``im:ws:``. Redis (not Postgres) is the source of truth
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
"""

SECRET_PREFIX = "im:ws:secret"
EPOCH_PREFIX = "im:ws:epoch"
OWNER_PREFIX = "im:ws:owner"
PRESENCE_PREFIX = "im:ws:presence"

METRICS_OWNER_PREFIX = "im:metrics:owner"
METRICS_MACHINE_PREFIX = "im:metrics:machine"
METRICS_CATS_PREFIX = "im:metrics:cats"

# TTL for the metrics ownership lease and the machine snapshot. The owner
# must refresh it by emitting metrics.machine more often than this; on
# expiry another instance on the same machine can take over.
METRICS_OWNER_TTL_SECONDS = 30


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
