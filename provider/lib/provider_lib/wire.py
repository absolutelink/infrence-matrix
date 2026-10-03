"""Wire envelope for admin <-> provider WebSocket messages.

Every frame is a single JSON object with this shape:

    {
      "v": 1,
      "type": "<event_or_command_name>",
      "id": "<message id, unique per sender connection>",
      "reply_to": "<id of the message this is an ack/response for, or null>",
      "epoch": <provider connection epoch, int>,
      "ts": "<ISO-8601 UTC>",
      "payload": { ... }
    }

Direction and semantics are defined by `type`. See docs/ws-protocol.md.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Frame(BaseModel):
    v: int = PROTOCOL_VERSION
    type: str
    id: str = ""
    reply_to: str | None = None
    epoch: int = 0
    ts: str = Field(default_factory=now_iso)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()


class FrameKind:
    """Command and event type constants (admin <-> provider)."""

    # provider -> admin
    PROVIDER_STATUS = "provider.status"
    BACKEND_STATUS = "backend.status"
    BACKEND_BOOT_REQUESTED = "backend.boot_requested"
    METRICS_MACHINE = "metrics.machine"
    METRICS_INFERENCE = "metrics.inference"
    BACKEND_LOGS = "backend.logs"
    PROVIDER_LOGS = "provider.logs"
    DOWNLOAD_PROGRESS = "download.progress"
    BACKEND_METADATA = "backend.metadata"
    PING = "ping"

    # admin -> provider
    BACKEND_START = "backend.start"
    BACKEND_STOP = "backend.stop"
    BACKEND_RESTART = "backend.restart"
    PROVIDER_INITIALIZE = "provider.initialize"
    PROVIDER_CONFIG_UPDATE = "provider.config.update"
    METRICS_ASSIGN = "metrics.assign"
    METRICS_UNASSIGN = "metrics.unassign"
    METRICS_CATEGORY_START = "metrics.category.start"
    CACHE_CLEAR = "cache.clear"
    STORAGE_PRUNE_UNUSED = "storage.prune_unused"
    PONG = "pong"


class Ack(BaseModel):
    """Standard command acknowledgement payload."""

    ok: bool
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class BackendStatusValue:
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    IN_USE = "in_use"
    STOPPING = "stopping"
    ERROR = "error"


class InstanceStatusValue:
    REGISTERING = "registering"
    INITIALIZING = "initializing"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    ERROR = "error"
    DISCONNECTED = "disconnected"


Direction = Literal["inbound", "outbound"]
