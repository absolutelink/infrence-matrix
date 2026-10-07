"""Admin-side mirror of the provider wire envelope.

This intentionally duplicates ``provider_lib.wire`` (provider/lib) rather
than importing it: the admin image is built with
``uv sync --no-install-workspace --package matrix-admin`` and does not ship
provider packages. The two definitions MUST stay in sync — the frame shape
is canonical in ``provider/lib/provider_lib/wire.py`` and documented in
``docs/ws-protocol.md``.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1

WS_CLOSE_AUTH_FAILED = 4401


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
    PROVIDER_HELLO = "provider.hello"
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
    BACKEND_LOGS_GET = "backend.logs.get"
    PONG = "pong"

    # both directions
    ACK = "ack"


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
    # Phase 16: container-level agent status. The Phase 14
    # ``awaiting_config`` pre-state is retired (no shells). Like
    # ``disconnected``, ``registering``/``unhealthy`` are admin-owned and
    # the provider never emits them.
    REGISTERING = "registering"
    INITIALIZING = "initializing"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    ERROR = "error"
    DISCONNECTED = "disconnected"
