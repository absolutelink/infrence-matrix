import uuid
from datetime import UTC, datetime

from sqlalchemy import BigInteger, Index
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlmodel import Column, Field, Relationship, SQLModel


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


# ============================================================================
# Legacy Models - For backward compatibility with existing test infrastructure
# ============================================================================
class UserBase(SQLModel):
    """Base user model."""
    email: str = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = None


class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(min_length=8, max_length=40)


class UserUpdate(SQLModel):
    """User update model."""
    email: str | None = None
    password: str | None = Field(default=None, min_length=8, max_length=40)
    full_name: str | None = None
    is_active: bool | None = None


class User(UserBase, table=True):
    """User database model."""
    __tablename__ = "user"
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )
    hashed_password: str
    created_at: datetime = Field(default_factory=get_datetime_utc)
    items: list["Item"] = Relationship(
        back_populates="owner",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


class ItemBase(SQLModel):
    """Base item model."""
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None


class ItemCreate(ItemBase):
    """Item creation model."""
    pass


class ItemUpdate(SQLModel):
    """Item update model."""
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None


class Item(ItemBase, table=True):
    """Item database model."""
    __tablename__ = "item"
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
    )
    owner: "User" = Relationship(
        back_populates="items",
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    created_at: datetime = Field(default_factory=get_datetime_utc)


class TokenPayload(SQLModel):
    """JWT token payload."""
    sub: str | None = None
    exp: int | None = None
    type: str | None = None


class Message(SQLModel):
    """Generic message response."""
    message: str


class ItemsPublic(SQLModel):
    """List of items response."""
    data: list[Item]
    count: int


class ItemPublic(ItemBase):
    """Public item model."""
    id: uuid.UUID
    owner_id: uuid.UUID


class UserPublic(UserBase):
    """Public user model."""
    id: uuid.UUID


class UsersPublic(SQLModel):
    """List of users response."""
    data: list[UserPublic]
    count: int


class Token(SQLModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"


class NewPassword(SQLModel):
    """New password request."""
    token: str
    new_password: str = Field(min_length=8, max_length=40)


class UserRegister(SQLModel):
    """User registration model."""
    email: str = Field(unique=True, index=True, max_length=255)
    password: str = Field(min_length=8, max_length=40)
    full_name: str | None = None
    is_active: bool = True


class UserUpdateMe(SQLModel):
    """User update me model."""
    full_name: str | None = None
    email: str | None = None
    password: str | None = Field(default=None, min_length=8, max_length=40)


class UpdatePassword(SQLModel):
    """Update password model."""
    current_password: str = Field(min_length=8, max_length=40)
    new_password: str = Field(min_length=8, max_length=40)


# ============================================================================
# Agent - Tracks registered inference agents
# ============================================================================
class Agent(SQLModel, table=True):
    __tablename__ = "agents"
    __table_args__ = (
        Index("idx_agents_status", "status"),
        Index("idx_agents_name", "name", unique=True),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    name: str = Field(max_length=255, unique=True, index=True)
    host: str
    port: int

    status: str = "offline"
    gpu_info: dict = Field(default_factory=dict, sa_column=Column(JSON))

    last_seen: datetime | None = None
    websocket_connected: bool = False

    created_at: datetime = Field(default_factory=get_datetime_utc)

    # Relationships
    servers: list["ServerInstance"] = Relationship(
        back_populates="agent",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


# ============================================================================
# Model - Registry of available models
# ============================================================================
class Model(SQLModel, table=True):
    __tablename__ = "models"
    __table_args__ = (
        Index("idx_models_name", "name"),
        Index("idx_models_architecture", "architecture"),
        Index("idx_models_source", "source"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    name: str = Field(max_length=512, unique=True, index=True)
    path: str = Field(max_length=1024)
    size_bytes: int = Field(sa_type=BigInteger)  # type: ignore[call-arg]

    architecture: str
    parameter_count: int | None = Field(default=None, sa_type=BigInteger)  # type: ignore[call-arg]
    quantization: str

    supports_embeddings: bool = False
    supports_vision: bool = False
    context_length: int

    license: str | None = None
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    description: str | None = None

    source: str
    source_repo_id: str | None = None
    source_url: str | None = None
    source_file: str | None = None

    downloaded_at: datetime = Field(default_factory=get_datetime_utc)
    updated_at: datetime | None = None

    conversations: list["Conversation"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    server_instances: list["ServerInstance"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    cache_entries: list["PromptCache"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


class ModelCreate(SQLModel):
    """Model creation model."""
    name: str = Field(max_length=512)
    path: str = Field(max_length=1024)
    size_bytes: int = Field(sa_type=BigInteger)  # type: ignore[call-arg]
    architecture: str
    parameter_count: int | None = Field(default=None, sa_type=BigInteger)  # type: ignore[call-arg]
    quantization: str
    supports_embeddings: bool = False
    supports_vision: bool = False
    context_length: int
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    description: str | None = None
    source: str
    source_repo_id: str | None = None
    source_url: str | None = None
    source_file: str | None = None


class ModelUpdate(SQLModel):
    """Model update model."""
    name: str | None = Field(default=None, max_length=512)
    path: str | None = Field(default=None, max_length=1024)
    size_bytes: int | None = Field(default=None, sa_type=BigInteger)  # type: ignore[call-arg]
    architecture: str | None = None
    parameter_count: int | None = Field(default=None, sa_type=BigInteger)  # type: ignore[call-arg]
    quantization: str | None = None
    supports_embeddings: bool | None = None
    supports_vision: bool | None = None
    context_length: int | None = None
    license: str | None = None
    tags: list[str] | None = None
    description: str | None = None
    source: str | None = None
    source_repo_id: str | None = None
    source_url: str | None = None
    source_file: str | None = None


# ============================================================================
# Conversation - Tree-structured conversation history
# ============================================================================
class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conversations_parent_id", "parent_id"),
        Index("idx_conversations_model_id", "model_id"),
        Index("idx_conversations_response_id", "response_id"),
        Index("idx_conversations_created_at", "created_at"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    parent_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="conversations.id",
        ondelete="CASCADE",
    )
    response_id: str | None = None

    input_items: list[dict[str, str]] = Field(
        default_factory=list,
        sa_column=Column(JSON),
    )
    output_items: list[dict[str, str]] = Field(
        default_factory=list,
        sa_column=Column(JSON),
    )

    model_id: uuid.UUID = Field(
        foreign_key="models.id",
        ondelete="CASCADE",
    )
    parameters: dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )
    conversation_metadata: dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )

    status: str
    error_message: str | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    created_at: datetime = Field(default_factory=get_datetime_utc)
    completed_at: datetime | None = None

    model: "Model" = Relationship(
        back_populates="conversations",
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    children: list["Conversation"] = Relationship(
        back_populates="parent",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "remote_side": "Conversation.id",
        },
    )
    parent: "Conversation" = Relationship(
        back_populates="children",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


# ============================================================================
# APIKey - API authentication keys
# ============================================================================
class APIKey(SQLModel, table=True):
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("idx_api_keys_key_hash", "key_hash"),
        Index("idx_api_keys_is_active", "is_active"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    name: str = Field(max_length=255)
    key_hash: str = Field(max_length=512, index=True)
    key_prefix: str = Field(max_length=10)

    permissions: list[str] = Field(
        default_factory=lambda: ["chat", "embeddings", "files"],
        sa_column=Column(JSON),
    )

    rate_limit_requests: int | None = None
    rate_limit_tokens: int | None = None

    is_active: bool = True
    last_used_at: datetime | None = None
    expires_at: datetime | None = None

    created_at: datetime = Field(default_factory=get_datetime_utc)
    created_by: uuid.UUID

    total_requests: int = 0
    total_tokens: int = 0


# ============================================================================
# PromptCache - Metadata for prompt caching
# ============================================================================
class PromptCache(SQLModel, table=True):
    __tablename__ = "prompt_cache"
    __table_args__ = (
        Index("idx_prompt_cache_cache_key", "cache_key"),
        Index("idx_prompt_cache_model_id", "model_id"),
        Index("idx_prompt_cache_expires_at", "expires_at"),
        Index("idx_prompt_cache_conversation_id", "conversation_id"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    cache_key: str = Field(max_length=512, index=True)
    cache_type: str

    model_id: uuid.UUID = Field(
        foreign_key="models.id",
        ondelete="CASCADE",
    )
    content_hash: str = Field(max_length=64)

    llama_cache_id: str | None = None

    hits: int = 0
    size_bytes: int
    token_count: int

    ttl_seconds: int
    created_at: datetime = Field(default_factory=get_datetime_utc)
    expires_at: datetime
    last_accessed_at: datetime = Field(default_factory=get_datetime_utc)

    conversation_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="conversations.id",
        ondelete="SET NULL",
    )
    agent_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="agents.id",
        ondelete="SET NULL",
    )
    cache_path: str

    model: "Model" = Relationship(
        back_populates="cache_entries",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


# ============================================================================
# DownloadJob - Tracks model download progress
# ============================================================================
class DownloadJob(SQLModel, table=True):
    __tablename__ = "download_jobs"
    __table_args__ = (
        Index("idx_download_jobs_status", "status"),
        Index("idx_download_jobs_model_id", "model_id"),
        Index("idx_download_jobs_created_at", "created_at"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    model_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="models.id",
        ondelete="SET NULL",
    )

    source: str
    repo_id: str
    filename: str
    download_url: str

    status: str
    progress_percent: float = 0.0
    bytes_downloaded: int = 0
    total_bytes: int | None = None

    current_speed: int = 0
    average_speed: int = 0
    eta_seconds: int | None = None

    pause_token: str | None = None

    retry_count: int = 0
    max_retries: int = 3
    last_error: str | None = None
    last_error_at: datetime | None = None

    created_at: datetime = Field(default_factory=get_datetime_utc)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    destination_path: str


# ============================================================================
# ServerInstance - Tracks running llama-server processes
# ============================================================================
class ServerInstance(SQLModel, table=True):
    __tablename__ = "server_instances"
    __table_args__ = (
        Index("idx_server_instances_model_id", "model_id"),
        Index("idx_server_instances_status", "status"),
        Index("idx_server_instances_port", "port"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    model_id: uuid.UUID = Field(
        foreign_key="models.id",
        ondelete="CASCADE",
    )

    port: int = Field(unique=True)
    pid: int | None = None
    process_command: str

    config: dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )

    status: str
    health_status: str = "unknown"

    started_at: datetime | None = None
    last_request_at: datetime | None = None
    last_health_check: datetime | None = None

    total_requests: int = 0
    total_tokens_generated: int = 0
    average_response_time_ms: float = 0.0

    vram_usage_bytes: int | None = None
    ram_usage_bytes: int | None = None
    cpu_usage_percent: float | None = None

    inactivity_timeout_seconds: int
    auto_shutdown_at: datetime | None = None

    error_message: str | None = None
    restart_count: int = 0

    agent_id: uuid.UUID = Field(
        foreign_key="agents.id",
        ondelete="CASCADE",
    )
    proxy_url: str | None = None

    agent: "Agent" = Relationship(
        back_populates="servers",
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    model: "Model" = Relationship(
        back_populates="server_instances",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


# ============================================================================
# AudioJob - Tracks audio processing jobs
# ============================================================================
class AudioJob(SQLModel, table=True):
    __tablename__ = "audio_jobs"
    __table_args__ = (
        Index("idx_audio_jobs_type", "job_type"),
        Index("idx_audio_jobs_status", "status"),
        Index("idx_audio_jobs_input_file", "input_file_id"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    job_type: str

    input_file_id: uuid.UUID = Field(
        foreign_key="files.id",
        ondelete="CASCADE",
    )
    output_file_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="files.id",
        ondelete="SET NULL",
    )

    model_id: uuid.UUID = Field(
        foreign_key="models.id",
        ondelete="CASCADE",
    )

    parameters: dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )

    status: str
    progress_percent: float = 0.0
    error_message: str | None = None

    result_text: str | None = None

    created_at: datetime = Field(default_factory=get_datetime_utc)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    processing_time_ms: int | None = None

    input_file: "File" = Relationship(
        back_populates="audio_jobs",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "[AudioJob.input_file_id]",
        },
    )
    output_file: "File" = Relationship(
        back_populates="audio_output_jobs",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "[AudioJob.output_file_id]",
        },
    )
    model: "Model" = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"},
    )


# ============================================================================
# BatchJob - Tracks batch processing jobs
# ============================================================================
class BatchJob(SQLModel, table=True):
    __tablename__ = "batch_jobs"
    __table_args__ = (
        Index("idx_batch_jobs_status", "status"),
        Index("idx_batch_jobs_expires_at", "expires_at"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    input_file_id: uuid.UUID = Field(
        foreign_key="files.id",
        ondelete="CASCADE",
    )
    output_file_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="files.id",
        ondelete="SET NULL",
    )
    results_file_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="files.id",
        ondelete="SET NULL",
    )

    endpoint: str

    completion_window_hours: int = 24

    status: str

    total_requests: int = 0
    processed_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    created_at: datetime = Field(default_factory=get_datetime_utc)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime

    error_message: str | None = None

    input_file: "File" = Relationship(
        back_populates="batch_jobs",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "[BatchJob.input_file_id]",
        },
    )
    output_file: "File" = Relationship(
        back_populates="batch_output_jobs",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "[BatchJob.output_file_id]",
        },
    )
    results_file: "File" = Relationship(
        back_populates="batch_results_jobs",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "[BatchJob.results_file_id]",
        },
    )


# ============================================================================
# File - Tracks uploaded files
# ============================================================================
class File(SQLModel, table=True):
    __tablename__ = "files"
    __table_args__ = (
        Index("idx_files_purpose", "purpose"),
        Index("idx_files_created_at", "created_at"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        sa_type=UUID(as_uuid=True),  # type: ignore[call-arg,arg-type]
    )

    filename: str = Field(max_length=512)
    path: str = Field(max_length=1024)

    size_bytes: int
    mime_type: str
    checksum_sha256: str = Field(max_length=64)

    purpose: str

    file_metadata: dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )

    status: str
    error_message: str | None = None

    created_at: datetime = Field(default_factory=get_datetime_utc)
    expires_at: datetime | None = None

    audio_jobs: list["AudioJob"] = Relationship(
        back_populates="input_file",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "AudioJob.input_file_id",
        },
    )
    audio_output_jobs: list["AudioJob"] = Relationship(
        back_populates="output_file",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "AudioJob.output_file_id",
        },
    )
    batch_jobs: list["BatchJob"] = Relationship(
        back_populates="input_file",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "BatchJob.input_file_id",
        },
    )
    batch_output_jobs: list["BatchJob"] = Relationship(
        back_populates="output_file",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "BatchJob.output_file_id",
        },
    )
    batch_results_jobs: list["BatchJob"] = Relationship(
        back_populates="results_file",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "BatchJob.results_file_id",
        },
    )


# ============================================================================
# Update relationships for AudioJob and BatchJob
# ============================================================================
AudioJob.model_rebuild()
BatchJob.model_rebuild()
File.model_rebuild()
