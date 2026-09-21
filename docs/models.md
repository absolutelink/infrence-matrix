# Database Models Reference

This document describes all PostgreSQL database models used by Inference Matrix.

## Models Overview

```
┌─────────────────────┐
│       Model         │  Model registry and metadata
├─────────────────────┤
│    Conversation     │  Tree-structured conversation history
├─────────────────────┤
│   PromptCache       │  Prompt cache metadata
├─────────────────────┤
│   DownloadJob       │  Model download tracking
├─────────────────────┤
│  ServerInstance     │  llama-server process tracking
├─────────────────────┤
│     AudioJob        │  Audio processing jobs
├─────────────────────┤
│     BatchJob        │  Batch processing jobs
├─────────────────────┤
│       File          │  Uploaded file metadata
└─────────────────────┘
```

---

## Model

Registry of available models with full metadata.

```python
class Model(SQLModel, table=True):
    __tablename__ = "models"
    
    # Primary Key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Basic Info
    name: str = Field(max_length=512, unique=True)  # Filename
    path: str = Field(max_length=1024)  # Full filesystem path
    size_bytes: int  # File size in bytes
    
    # Model Architecture
    architecture: str  # e.g., "llama", "mistral", "qwen"
    parameter_count: Optional[int]  # Total parameters
    quantization: str  # e.g., "Q4_K_M", "Q5_K_M"
    
    # Capabilities
    supports_embeddings: bool = False
    supports_vision: bool = False
    context_length: int  # Max context window
    
    # Metadata
    license: Optional[str]
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    description: Optional[str]
    
    # Source Information
    source: str  # "huggingface" or "modelscope"
    source_repo_id: Optional[str]  # e.g., "TheBloke/Llama-2-7B"
    source_url: Optional[str]
    source_file: Optional[str]  # Specific file in repo
    
    # Timestamps
    downloaded_at: datetime
    updated_at: Optional[datetime]
    
    # Relationships
    conversations: list[Conversation] = Relationship(back_populates="model")
    server_instances: list[ServerInstance] = Relationship(back_populates="model")
    cache_entries: list[PromptCache] = Relationship(back_populates="model")
```

**Indexes:**
- `idx_models_name` - Fast lookup by name
- `idx_models_architecture` - Filter by architecture
- `idx_models_source` - Filter by source

---

## Conversation

Tree-structured conversation history for `/v1/responses` endpoint.

```python
class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"
    
    # Primary Key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Tree Structure
    parent_id: Optional[UUID] = Field(foreign_key="conversations.id", index=True)
    response_id: Optional[str]  # OpenAI-compatible response ID
    
    # Content
    input_items: list[dict] = Field(sa_column=Column(JSON))  # Input items
    output_items: list[dict] = Field(sa_column=Column(JSON))  # Output items
    
    # Configuration
    model_id: UUID = Field(foreign_key="models.id", index=True)
    parameters: dict = Field(default_factory=dict, sa_column=Column(JSON))
    # {temperature, top_p, max_tokens, tools, etc.}
    
    # Metadata
    metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    # User-provided metadata (max 16 key-value pairs)
    
    # Status
    status: str  # "in_progress", "completed", "failed"
    error_message: Optional[str]
    
    # Usage
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # Timestamps
    created_at: datetime
    completed_at: Optional[datetime]
    
    # Relationships
    model: Model = Relationship(back_populates="conversations")
    children: list[Conversation] = Relationship(
        back_populates="parent",
        sa_relationship_kwargs=dict(remote_side="Conversation.id")
    )
    parent: Optional[Conversation] = Relationship(
        back_populates="children"
    )
```

**Indexes:**
- `idx_conversations_parent_id` - Tree traversal
- `idx_conversations_model_id` - Filter by model
- `idx_conversations_response_id` - Lookup by response ID
- `idx_conversations_created_at` - Time-based queries

**Tree Structure Example:**
```
Conversation 1 (root)
├── Conversation 2 (child of 1)
│   └── Conversation 4 (child of 2)
└── Conversation 3 (child of 1)
```


## PromptCache

Metadata for prompt caching with llama.cpp.

```python
class PromptCache(SQLModel, table=True):
    __tablename__ = "prompt_cache"
    
    # Primary Key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Cache Identification
    cache_key: str = Field(max_length=512, index=True)
    # Unique identifier for this cache entry
    
    cache_type: str  # "conversation" or "hierarchical"
    # conversation: Chat completions caching
    # hierarchical: Responses API hierarchical caching
    
    # Content Identification
    model_id: UUID = Field(foreign_key="models.id", index=True)
    content_hash: str = Field(max_length=64)
    # SHA256 hash of cached content
    
    # llama.cpp Integration
    llama_cache_id: Optional[str]
    # Internal llama.cpp cache identifier
    
    # Statistics
    hits: int = 0
    size_bytes: int
    token_count: int
    
    # TTL
    ttl_seconds: int  # Time-to-live in seconds
    created_at: datetime
    expires_at: datetime  # created_at + ttl_seconds
    last_accessed_at: datetime
    
    # Metadata
    conversation_id: Optional[UUID] = Field(
        foreign_key="conversations.id",
        index=True
    )
    # Associated conversation (for conversation-based caching)
    
    cache_path: str
    # Path to cache file on disk (if using file-based cache)
```

**Indexes:**
- `idx_prompt_cache_cache_key` - Fast cache lookup
- `idx_prompt_cache_model_id` - Filter by model
- `idx_prompt_cache_expires_at` - Cleanup expired entries
- `idx_prompt_cache_conversation_id` - Link to conversations

**Cache Cleanup:**
```sql
DELETE FROM prompt_cache WHERE expires_at < NOW();
```

---

## DownloadJob

Tracks model download progress and state.

```python
class DownloadJob(SQLModel, table=True):
    __tablename__ = "download_jobs"
    
    # Primary Key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Target Model
    model_id: Optional[UUID] = Field(
        foreign_key="models.id",
        index=True
    )
    # Null until download completes
    
    # Source Information
    source: str  # "huggingface" or "modelscope"
    repo_id: str  # e.g., "TheBloke/Llama-2-7B-GGUF"
    filename: str  # Specific file to download
    download_url: str
    
    # Progress Tracking
    status: str  # "pending", "downloading", "paused", "completed", "failed"
    progress_percent: float = 0.0
    bytes_downloaded: int = 0
    total_bytes: Optional[int]
    
    # Speed Tracking
    current_speed: int  # Bytes per second
    average_speed: int
    eta_seconds: Optional[int]
    
    # Pause/Resume
    pause_token: Optional[str]
    # Token for resumable downloads
    
    # Retry Logic
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str]
    last_error_at: Optional[datetime]
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Destination
    destination_path: str
    # Where file will be saved
```

**Indexes:**
- `idx_download_jobs_status` - Filter by status
- `idx_download_jobs_model_id` - Link to models
- `idx_download_jobs_created_at` - Order by creation

**State Transitions:**
```
pending → downloading → completed
              ↓
           paused → downloading
              ↓
           failed → (retry) → downloading
```

---

## ServerInstance

Tracks running llama-server processes.

```python
class ServerInstance(SQLModel, table=True):
    __tablename__ = "server_instances"
    
    # Primary Key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Model Association
    model_id: UUID = Field(foreign_key="models.id", index=True)
    
    # Process Information
    port: int = Field(unique=True)
    pid: Optional[int]  # Process ID
    process_command: str  # Full command used to start
    
    # Configuration
    config: dict = Field(sa_column=Column(JSON))
    # {gpu_layers, context_size, batch_size, etc.}
    
    # Status
    status: str  # "starting", "running", "stopping", "stopped", "failed"
    health_status: str  # "healthy", "unhealthy", "unknown"
    
    # Timing
    started_at: Optional[datetime]
    last_request_at: Optional[datetime]
    last_health_check: Optional[datetime]
    
    # Statistics
    total_requests: int = 0
    total_tokens_generated: int = 0
    average_response_time_ms: float = 0.0
    
    # Resource Usage
    vram_usage_bytes: Optional[int]
    ram_usage_bytes: Optional[int]
    cpu_usage_percent: Optional[float]
    
    # Auto-shutdown
    inactivity_timeout_seconds: int
    auto_shutdown_at: Optional[datetime]
    # Calculated: last_request_at + inactivity_timeout
    
    # Error Tracking
    error_message: Optional[str]
    restart_count: int = 0
```

**Indexes:**
- `idx_server_instances_model_id` - Find server for model
- `idx_server_instances_status` - Filter by status
- `idx_server_instances_port` - Port lookup

**Lifecycle:**
1. Created with status="starting"
2. Updated to "running" when healthy
3. Auto-shutdown when `auto_shutdown_at < NOW()`
4. Status="stopped" after graceful shutdown

---

## AudioJob

Tracks audio processing jobs (transcription, translation, speech).

```python
class AudioJob(SQLModel, table=True):
    __tablename__ = "audio_jobs"
    
    # Primary Key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Job Type
    job_type: str  # "transcription", "translation", "speech"
    
    # File References
    input_file_id: UUID = Field(foreign_key="files.id")
    output_file_id: Optional[UUID] = Field(foreign_key="files.id")
    
    # Model
    model_id: UUID = Field(foreign_key="models.id")
    # Whisper model for transcription/translation
    # TTS model for speech
    
    # Parameters
    parameters: dict = Field(sa_column=Column(JSON))
    # {language, prompt, response_format, temperature, voice, speed}
    
    # Status
    status: str  # "pending", "processing", "completed", "failed"
    progress_percent: float = 0.0
    error_message: Optional[str]
    
    # Result
    result_text: Optional[str]
    # For transcription/translation
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Processing Time
    processing_time_ms: Optional[int]
```

**Indexes:**
- `idx_audio_jobs_type` - Filter by job type
- `idx_audio_jobs_status` - Filter by status
- `idx_audio_jobs_input_file` - Lookup by input file

---

## BatchJob

Tracks batch processing jobs.

```python
class BatchJob(SQLModel, table=True):
    __tablename__ = "batch_jobs"
    
    # Primary Key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # File References
    input_file_id: UUID = Field(foreign_key="files.id")
    output_file_id: Optional[UUID] = Field(foreign_key="files.id")
    results_file_id: Optional[UUID] = Field(foreign_key="files.id")
    
    # Configuration
    endpoint: str  # Target endpoint
    # e.g., "/v1/chat/completions"
    
    completion_window_hours: int = 24
    # Maximum time to complete batch
    
    # Status
    status: str  # "validating", "in_progress", "completed", "failed", "cancelled"
    
    # Progress
    total_requests: int = 0
    processed_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Timing
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    expires_at: datetime  # created_at + completion_window
    
    # Errors
    error_message: Optional[str]
```

**Indexes:**
- `idx_batch_jobs_status` - Filter by status
- `idx_batch_jobs_expires_at` - Cleanup expired jobs

---

## File

Tracks uploaded files for batch processing and retrieval.

```python
class File(SQLModel, table=True):
    __tablename__ = "files"
    
    # Primary Key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # File Info
    filename: str = Field(max_length=512)
    path: str = Field(max_length=1024)
    # Filesystem path
    
    size_bytes: int
    mime_type: str
    checksum_sha256: str = Field(max_length=64)
    
    # Purpose
    purpose: str  # "batch", "retrieval", "assistants", "audio"
    
    # Metadata
    metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Status
    status: str  # "uploaded", "processing", "ready", "error"
    error_message: Optional[str]
    
    # Timestamps
    created_at: datetime
    expires_at: Optional[datetime]
    # Optional expiration for temporary files
    
    # Relationships
    audio_jobs: list[AudioJob] = Relationship()
    batch_jobs: list[BatchJob] = Relationship()
```

**Indexes:**
- `idx_files_purpose` - Filter by purpose
- `idx_files_created_at` - Time-based queries

---

## Relationships Diagram

```
┌──────────────┐       ┌──────────────┐
│    Model     │◄──────│ Conversation │
└──────────────┘       └──────────────┘
       │                      │
       │                      │ (parent_id self-reference)
       │                      ▼
       │              ┌──────────────┐
       │              │ Conversation │
       │              └──────────────┘
       │
       ├──────►┌──────────────┐
       │       │ServerInstance│
       │       └──────────────┘
       │
       ├──────►┌──────────────┐
       │       │ PromptCache  │
       │       └──────────────┘
       │
       └──────►┌──────────────┐
               │  DownloadJob │
               └──────────────┘

┌──────────────┐       ┌──────────────┐
│     File     │       │   AudioJob   │
└──────────────┘       └──────────────┘
                              │
                              └──────►┌──────────────┐
                                      │   BatchJob   │
                                      └──────────────┘
```

---

## Migration Notes

All models use SQLModel for ORM. Migrations are managed by Alembic.

**Creating a migration:**
```bash
cd backend
alembic revision --autogenerate -m "Add new field"
alembic upgrade head
```

**Rolling back:**
```bash
alembic downgrade -1
```
