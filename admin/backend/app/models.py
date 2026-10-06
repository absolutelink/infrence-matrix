"""Inference Matrix admin data model (Phase 2).

Six tables replace the old agent/server_instance/model/lease set:

  Machine            a physical/virtual host, pre-registered in the UI by uid.
  ProviderType       registered provider type (Phase 12): the committed JSON
                     Schema for its backend_config + schema consensus state.
  ProviderDefinition the client-facing "model": how to boot a backend +
                     scheduling hints (vram, idle timeout, capacity) + the
                     registration token that binds instances to it.
  ProviderInstance   one running (or registered) backend on one Machine for
                     one ProviderDefinition. Exactly one backend per instance.
  ResponseRecord     stored OpenResponses turn, chained via previous_response_id.
  TokenUsageSample   per-request token/rate telemetry.

The provider instance itself is stateless (no DB); it derives everything from
env + the registration response, mirrored here by the admin.

Auth model: trusted LAN. The registration_token and per-instance secret gate
only the provider WebSocket. See the root ARCHITECTURE.md.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import BigInteger, DateTime, Index, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlmodel import Column, Field, Relationship, SQLModel


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


def _uuid_col() -> UUID:
    return UUID(as_uuid=True)  # type: ignore[call-arg]


# ============================================================================
# Machine
# ============================================================================
class Machine(SQLModel, table=True):
    """A host that provider instances run on.

    Created in the admin UI before any provider registers against its ``uid``.
    Hostname/DNS/IP are what the admin uses to reach the instance's provider
    port. Hardware inventory (GPUs, etc.) is merged in from registration
    reports; total_vram_bytes is the admission budget for the scheduler.
    """

    __tablename__ = "machines"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(_uuid_col(), primary_key=True),
    )

    # Stable identifier supplied by the operator and referenced by the
    # provider instance's MACHINE_UID env var.
    uid: str = Field(max_length=255, unique=True)
    name: str = Field(max_length=255, unique=True)

    # How the admin reaches provider instances on this machine. At least one
    # of host/dns/ip must be set; litellm targets http://{reachable}:{port}/v1.
    host: str | None = Field(default=None, max_length=255)
    dns: str | None = Field(default=None, max_length=255)
    ip: str | None = Field(default=None, max_length=255)

    # Admission budget. Merged/refreshed from provider-reported hardware.
    total_vram_bytes: int = Field(default=0, sa_type=BigInteger)  # type: ignore[call-arg]

    # Full hardware inventory as reported by provider instances:
    #   {"gpus": [{"uuid","vendor","name","total_vram_bytes"}, ...],
    #    "cpu": {...}, "ram": {...}, ...}
    # Admin merges reports from every instance on this machine so the
    # machine record reflects the union of observed hardware.
    hardware: dict = Field(default_factory=dict, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=get_datetime_utc)
    updated_at: datetime | None = None

    instances: list[ProviderInstance] = Relationship(
        back_populates="machine",
        sa_relationship_kwargs={"lazy": "selectin"},
    )

    def reachable_address(self) -> str | None:
        """Best address for the admin to open the instance's provider port."""
        return self.dns or self.host or self.ip


# ============================================================================
# ProviderType (Phase 12)
# ============================================================================
class ProviderType(SQLModel, table=True):
    """A registered provider **type**, created by the first registration
    of that type (ARCHITECTURE.md §4 / docs/ws-protocol.md §2).

    Owns the committed JSON Schema (2020-12) describing this type's
    ``backend_config``. Every ``ProviderDefinition.provider_type`` must
    reference a row here — the registry, not a constant, is the source of
    truth. The schema itself ships inside each provider package
    (``provider/<type>/provider_<type>/schema.json``) and is presented at
    registration; the admin never holds provider code.

    Schema consensus: a new schema must be presented by every known
    ``ProviderInstance`` of the type before it commits. While awaiting
    consensus the staged schema lives in ``pending_*`` and the
    registration is refused 409 ``schema_pending``; a third distinct
    fingerprint while pending flips ``status`` to ``conflict``.
    Force-commit / dismiss are operator overrides exposed under
    ``/admin/api/provider-types/{name}/pending/*``.

    NOTE: the field is named ``schema`` (it shadows the deprecated
    ``BaseModel.schema()`` — SQLModel emits a UserWarning at class-creation
    time; the mapping round-trips cleanly via the explicit ``sa_column``).
    """

    __tablename__ = "provider_types"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(_uuid_col(), primary_key=True),
    )

    # Type id: "llama-cpp", "halogen", "halogen-flash", "gufo", "mock", ...
    name: str = Field(max_length=64, unique=True)

    # Committed JSON Schema (2020-12) for this type's backend_config.
    schema: dict = Field(
        default_factory=dict,
        sa_column=Column("schema", JSON, nullable=False),
    )
    # sha256 of canonical json(schema) — same canonicalization as
    # config_fingerprint (app.services.hashing).
    schema_fingerprint: str = Field(max_length=64)

    # Staged schema awaiting consensus (None when nothing is pending).
    pending_schema: dict | None = Field(default=None, sa_column=Column(JSON))
    pending_fingerprint: str | None = Field(default=None, max_length=64)
    # Instance ids that registered presenting ``pending_fingerprint``.
    pending_voters: list = Field(default_factory=list, sa_column=Column(JSON))

    # active | consensus_pending | conflict
    status: str = Field(default="active", max_length=32)

    created_at: datetime = Field(default_factory=get_datetime_utc)
    updated_at: datetime | None = None


# ============================================================================
# ProviderDefinition
# ============================================================================
class ProviderDefinition(SQLModel, table=True):
    """A client-facing model: how to boot a backend and how to schedule it.

    ``alias`` is the model name clients use in /v1/models and the model field
    of inference requests. ``backend_config`` is the JSON handed to the
    provider instance to start its backend (model artifacts, args, engine
    options) — multi-file artifacts (main GGUF + mmproj + draft) live here.

    ``registration_token`` is the secret a provider container presents at
    registration; it binds the instance to this definition and its
    provider_type is cross-checked against the container's type.
    """

    __tablename__ = "provider_definitions"
    __table_args__ = (Index("idx_provider_definitions_type", "provider_type"),)

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(_uuid_col(), primary_key=True),
    )

    alias: str = Field(max_length=255, unique=True)
    provider_type: str = Field(max_length=64)  # references ProviderType.name (Phase 12)

    # Everything the provider needs to start the backend. Schema documented
    # in provider/README.md. Example:
    #   {"model": {"source": "hf", "repo": "...", "file": "..."},
    #    "mmproj": {...}, "draft": {...},
    #    "args": {"ctx": 8192, "gpu_layers": 35, "flash_attn": "on"},
    #    "engine_options": {...}}
    backend_config: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # Scheduler hints.
    vram_required_bytes: int = Field(default=0, sa_type=BigInteger)  # type: ignore[call-arg]
    idle_timeout_seconds: int = Field(default=300, ge=0)
    capacity: int = Field(default=1, ge=1)  # concurrent backend slots

    # Binding secret for provider registration.
    registration_token: str = Field(max_length=255, unique=True)

    # OpenAI model metadata discovered when the backend was initialized
    # (served model object, context length, capability flags).
    model_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))

    enabled: bool = True
    # running | stopped | initializing | error  (aggregate of instances)
    status: str = Field(default="stopped", max_length=32)

    created_at: datetime = Field(default_factory=get_datetime_utc)
    updated_at: datetime | None = None

    instances: list[ProviderInstance] = Relationship(
        back_populates="provider_definition",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


# ============================================================================
# ProviderInstance
# ============================================================================
class ProviderInstance(SQLModel, table=True):
    """One backend on one Machine for one ProviderDefinition.

    A provider container registers (machine_uid + registration_token) and the
    admin creates/updates this row. The instance dials the admin WebSocket;
    ``epoch`` is bumped on every accepted connection so stale frames from a
    dead connection are discarded (fencing). Machine-level metrics ownership
    lives in Redis, keyed off these rows.

    Uniqueness: one instance per (machine, definition) — a machine cannot run
    two backends of the same provider definition.
    """

    __tablename__ = "provider_instances"
    __table_args__ = (
        Index(
            "idx_provider_instances_machine_def",
            "machine_id",
            "provider_definition_id",
            unique=True,
        ),
        Index("idx_provider_instances_def", "provider_definition_id"),
        Index("idx_provider_instances_instance_status", "instance_status"),
        Index("idx_provider_instances_backend_status", "backend_status"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(_uuid_col(), primary_key=True),
    )

    machine_id: uuid.UUID = Field(foreign_key="machines.id", ondelete="CASCADE")
    provider_definition_id: uuid.UUID = Field(
        foreign_key="provider_definitions.id", ondelete="CASCADE"
    )

    # The provider's own port (default 8081). Admin points litellm at
    # http://{machine.reachable_address()}:{port}/v1.
    port: int = Field(default=8081)

    # Provider instance version (commit id until first release).
    version: str = Field(default="dev", max_length=64)

    # Dual status. instance_status: registering|initializing|running|unhealthy|error|disconnected
    # backend_status:   stopped|initializing|starting|running|in_use|stopping|error
    instance_status: str = Field(default="registering", max_length=32)
    backend_status: str = Field(default="stopped", max_length=32)

    # WebSocket liveness mirror. The authoritative connection lives in the
    # admin process; this is for UI/queries and reconnect reconciliation.
    websocket_connected: bool = False
    # Connection epoch: incremented each time the admin accepts a new socket
    # for this instance. Frames carrying a stale epoch are ignored.
    epoch: int = 0

    last_seen: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    last_request_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )

    # Hash of the backend_config the instance last applied. Drives auto
    # cache-clear on config change (Phase 9).
    config_fingerprint: str | None = Field(default=None, max_length=64)

    # Schema fingerprint the instance presented at its last registration
    # attempt (Phase 12; written on success AND on 409 schema-gate refusals
    # — drives the waiting_schema badge and the pending voter roster).
    reported_schema_fingerprint: str | None = Field(default=None, max_length=64)

    # Instance-reported hardware this backend is bound to (subset of the
    # machine's GPUs), used for VRAM accounting and metrics dedup.
    assigned_gpus: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    error_message: str | None = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(default_factory=get_datetime_utc)
    updated_at: datetime | None = None

    machine: Machine = Relationship(
        back_populates="instances",
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    provider_definition: ProviderDefinition = Relationship(
        back_populates="instances",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


# ============================================================================
# ResponseRecord
# ============================================================================
class ResponseRecord(SQLModel, table=True):
    """Stored OpenResponses turn (spec ResponseResource), chained by id.

    The admin owns the client-facing ``response_id`` (resp_<uuid>). litellm's
    affinity-wrapped upstream id is stored in ``parameters`` so a follow-up
    turn can hand the correct id back to litellm while the DB chain stays
    ours. input_items/output_items hold spec-shaped payloads.
    """

    __tablename__ = "responses"
    __table_args__ = (
        Index("idx_responses_previous_response_id", "previous_response_id"),
        Index("idx_responses_provider_definition_id", "provider_definition_id"),
        Index("idx_responses_created_at", "created_at"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(_uuid_col(), primary_key=True),
    )

    response_id: str = Field(max_length=255, unique=True)
    previous_response_id: str | None = Field(default=None, max_length=255)

    input_items: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    output_items: list[dict] = Field(default_factory=list, sa_column=Column(JSON))

    provider_definition_id: uuid.UUID = Field(
        foreign_key="provider_definitions.id", ondelete="CASCADE"
    )
    provider_instance_id: uuid.UUID | None = Field(
        default=None, foreign_key="provider_instances.id", ondelete="SET NULL"
    )

    # Full ResponseResource echo fields (params, model, instructions, tools...).
    parameters: dict = Field(default_factory=dict, sa_column=Column(JSON))
    response_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))

    status: str = Field(max_length=32)  # completed|failed|incomplete|in_progress
    error_code: str | None = None
    error_message: str | None = None
    incomplete_reason: str | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    store: bool = True
    background: bool = False

    created_at: datetime = Field(default_factory=get_datetime_utc)
    completed_at: datetime | None = None

    provider_definition: ProviderDefinition = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"}
    )


# ============================================================================
# TokenUsageSample
# ============================================================================
class TokenUsageSample(SQLModel, table=True):
    """Per-request token + rate telemetry for live statistics."""

    __tablename__ = "token_usage_samples"
    __table_args__ = (
        Index(
            "idx_token_usage_samples_instance_created",
            "provider_instance_id",
            "created_at",
        ),
        Index("idx_token_usage_samples_created_at", "created_at"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(_uuid_col(), primary_key=True),
    )

    provider_instance_id: uuid.UUID | None = Field(
        default=None, foreign_key="provider_instances.id", ondelete="SET NULL"
    )
    provider_definition_id: uuid.UUID | None = Field(
        default=None, foreign_key="provider_definitions.id", ondelete="SET NULL"
    )

    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    # Backend-reported stage durations (ms; 0 = not reported).
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0

    # Backend-reported stage rates (tokens/s; 0 = not reported).
    prompt_per_second: float = 0.0
    predicted_per_second: float = 0.0

    created_at: datetime = Field(
        default_factory=get_datetime_utc, sa_column=Column(DateTime(timezone=True))
    )
