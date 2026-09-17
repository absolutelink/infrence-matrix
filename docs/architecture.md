# Inference Matrix Architecture

## Overview

Inference Matrix is an OpenAI API-compatible inference server that routes requests to llama.cpp servers. It's designed for single-user home server deployments with full local model management.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WebUI (React)                          │
│            Model Management & Monitoring Dashboard          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              OpenAI-Compatible API                    │  │
│  │  /v1/models                                           │  │
│  │  /v1/chat/completions (SSE streaming)                 │  │
│  │  /v1/completions                                      │  │
│  │  /v1/embeddings                                       │  │
│  │  /v1/responses (tree-structured conversations)        │  │
│  │  /v1/files                                            │  │
│  │  /v1/batches                                          │  │
│  │  /v1/audio/transcriptions                             │  │
│  │  /v1/audio/translations                               │  │
│  │  /v1/audio/speech                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Management Services                      │  │
│  │  • llama-server Process Manager                       │  │
│  │  • Model Download Manager (HF + ModelScope)           │  │
│  │  • Prompt Cache Manager                               │  │
│  │  • GPU Configuration Manager                          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────────────────┐
│   PostgreSQL    │          │      llama.cpp Servers          │
│  • Models       │          │  (One subprocess per model)     │
│  • Conversations│          │  • Auto start on demand         │
│  • API Keys     │          │  • Auto shutdown on inactivity  │
│  • Cache Meta   │          │  • Configurable GPU layers      │
│  • DownloadJobs │          │  • Global + per-model config    │
│  • ServerInstances       │
│  • AudioJobs    │          └─────────────────────────────────┘
│  • BatchJobs    │
│  • Files        │
└─────────────────┘
```

## Core Components

### 1. FastAPI Backend

**Technology Stack:**
- Python 3.14+
- FastAPI for REST API
- SQLModel for ORM
- PostgreSQL for metadata storage
- JWT + API Key authentication (optional)

**Responsibilities:**
- OpenAI-compatible API endpoints
- Request routing to llama-server instances
- Model lifecycle management
- Conversation state management
- File storage management

### 2. llama-server Process Manager

**Architecture:**
- One llama-server subprocess per loaded model
- Dynamic port allocation
- Automatic lifecycle management:
  - Start on first request
  - Auto-shutdown after configurable inactivity timeout
- Health monitoring and automatic restart on failure

**Configuration:**
- Global defaults (context size, GPU layers, batch size)
- Per-model overrides stored in PostgreSQL
- User-configurable via WebUI

### 3. Model Management

**Supported Formats:**
- GGUF only (llama.cpp native format)

**Download Sources:**
- HuggingFace Hub (primary)
- ModelScope (secondary)
- Extensible architecture for additional sources

**Features:**
- Progress tracking
- Pause/resume support
- Automatic retry on failure
- Model validation after download
- Metadata extraction (size, architecture, capabilities)

**Storage:**
- Local filesystem only
- Structured directory hierarchy
- Database-tracked metadata

### 4. Prompt Cache Manager

**Caching Strategies:**

**Chat Completions (`/v1/chat/completions`):**
- Conversation-based caching
- Cache invalidated on new messages
- Automatic management

**Responses API (`/v1/responses`):**
- Hierarchical caching
- System prompt + conversation segments
- Tree-structure aware

**Implementation:**
- Native llama.cpp prompt caching
- Hybrid tracking in PostgreSQL
- Cache metadata (hit rates, size, TTL)

### 5. GPU Configuration

**Features:**
- Auto-detection of available GPUs (CUDA, Metal, Vulkan)
- User-configurable via WebUI
- Per-model GPU layer allocation
- VRAM monitoring

### 6. Audio Processing

**Implementation:**
- System-level Whisper installation
- Separate from llama.cpp inference
- Endpoints:
  - `/v1/audio/transcriptions` - Speech-to-text
  - `/v1/audio/translations` - Audio translation
  - `/v1/audio/speech` - Text-to-speech (future TTS integration)

## Data Models

### PostgreSQL Schema

```
Models:
  - id, name, path, size, architecture, quantization
  - capabilities, license, tags, benchmarks
  - source, source_id, downloaded_at

Conversations:
  - id, parent_id (tree structure), response_id
  - input_items, output_items
  - model_id, metadata, created_at

APIKeys:
  - id, key_hash, name, permissions
  - created_at, last_used_at, expires_at

PromptCache:
  - id, cache_key, cache_type (conversation/hierarchical)
  - model_id, content_hash, llama_cache_id
  - hits, size_bytes, created_at, expires_at

DownloadJobs:
  - id, model_id, source, status
  - progress_percent, bytes_downloaded, total_bytes
  - pause_token, retry_count, error_message

ServerInstances:
  - id, model_id, port, pid
  - config_json, status, started_at, last_request_at

AudioJobs:
  - id, job_type (transcription/translation/speech)
  - file_path, model_id, status, result_path
  - created_at, completed_at

BatchJobs:
  - id, input_file_id, output_file_id
  - endpoint, status, created_at, completed_at

Files:
  - id, filename, path, size, mime_type
  - purpose (batch/retrieval/etc.), created_at
```

## API Design

### Authentication

**Optional Authentication:**
- API keys stored in PostgreSQL (OpenAI-style)
- JWT tokens for admin UI access
- Configurable enforcement per endpoint

### Streaming

**Server-Sent Events (SSE):**
- Primary streaming method for chat completions
- OpenAI-compatible event format
- Supports `stream_options.include_usage`

### Error Handling

**Response Format:**
```json
{
  "error": {
    "message": "Human-readable error message",
    "type": "error_type",
    "code": "machine_readable_code",
    "param": "parameter_name"
  }
}
```

**HTTP Status Codes:**
- 200: Success
- 400: Bad Request (invalid parameters)
- 401: Unauthorized (invalid API key)
- 404: Not Found (model/resource not found)
- 429: Rate Limit Exceeded
- 500: Internal Server Error
- 503: Service Unavailable (model loading)

## Deployment Architecture

### Single-User Home Server

**Resource Management:**
- Configurable model concurrency limits
- Memory-aware model loading
- GPU memory monitoring
- Automatic model unloading under pressure

**Storage:**
- Models: Local filesystem (`/models`)
- Files: Local filesystem (`/files`)
- Database: PostgreSQL (Docker volume)
- Cache: llama.cpp native + PostgreSQL metadata

### Docker Compose Setup

```yaml
services:
  backend:
    # FastAPI application
    volumes:
      - ./models:/models
      - ./files:/files
      - ./cache:/cache
    depends_on:
      - postgres
  
  postgres:
    # PostgreSQL database
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  frontend:
    # React WebUI
    # Served by backend in production
```

## Monitoring & Observability

**Metrics Exposed:**
- VRAM usage per model
- Tokens/second generation rate
- Request queue depth
- Cache hit/miss rates
- Download progress
- Server instance status

**Export Formats:**
- Prometheus metrics endpoint
- JSON API endpoints
- WebSocket real-time updates (WebUI)

## Backup & Recovery

**Backup Components:**
- PostgreSQL database (all metadata)
- Model configurations
- Conversation history
- API keys
- Download job state

**Recovery Process:**
1. Restore PostgreSQL from backup
2. Verify model files exist
3. Re-register models in database
4. Resume interrupted downloads

## Security Considerations

**Local Deployment Focus:**
- Optional authentication (trusted network assumed)
- API keys hashed in database (argon2)
- No secrets in logs
- HTTPS termination at Traefik (if exposed)

**Model Security:**
- Model file validation after download
- Hash verification (when available from source)
- Quarantine for untrusted models

## Extensibility

**Plugin Architecture:**
- Model download sources (extensible)
- Audio processing backends
- Monitoring exporters
- Backup storage providers

**Future Enhancements:**
- Multi-user support with quotas
- Cluster deployment (distributed inference)
- Additional model formats (beyond GGUF)
- Native TTS integration
