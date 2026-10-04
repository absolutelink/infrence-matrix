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
"""

SECRET_PREFIX = "im:ws:secret"
EPOCH_PREFIX = "im:ws:epoch"
OWNER_PREFIX = "im:ws:owner"
PRESENCE_PREFIX = "im:ws:presence"


def secret_key(instance_id: str) -> str:
    return f"{SECRET_PREFIX}:{instance_id}"


def epoch_key(instance_id: str) -> str:
    return f"{EPOCH_PREFIX}:{instance_id}"


def owner_key(instance_id: str) -> str:
    return f"{OWNER_PREFIX}:{instance_id}"


def presence_key(instance_id: str) -> str:
    return f"{PRESENCE_PREFIX}:{instance_id}"
