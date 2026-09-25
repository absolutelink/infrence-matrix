# Inference Matrix Architecture

## Overview

Inference Matrix is an OpenAI API-compatible inference server with a distributed architecture. It consists of two services:

1. **Frontend Service** - WebUI, API endpoints, database, and orchestration
2. **Agent Service** - Hardware-local inference management with llama.cpp or Halogen

Designed for single-user home server deployments with support for multiple inference agents.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    FRONTEND SERVICE                              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 React WebUI                                │  │
│  │       Model Management & Monitoring Dashboard              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              FastAPI Backend                               │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │           OpenAI-Compatible API                      │  │  │
│  │  │  /v1/models                                          │  │  │
│  │  │  /v1/chat/completions (SSE streaming)                │  │  │
│  │  │  /v1/completions                                     │  │  │
│  │  │  /v1/embeddings                                      │  │  │
│  │  │  /v1/responses                                       │  │  │
│  │  │  /v1/files                                           │  │  │
│  │  │  /v1/batches                                         │  │  │
│  │  │  /v1/audio/*                                         │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                              │                               │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │           Management Services                        │  │  │
│  │  │  • Agent Manager (registration, discovery)           │  │  │
│  │  │  • Model Download Manager (HF + ModelScope)          │  │  │
│  │  │  • Prompt Cache Manager                              │  │  │
│  │  │  • Conversation Manager                              │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│         PostgreSQL Database  │                                     │
│  • Models                    │                                     │
│  • Conversations             │                                     │
│  • APIKeys                   │                                     │
│  • PromptCache (metadata)    │                                     │
│  • DownloadJobs              │                                     │
│  • ServerInstances           │                                     │
│  • Agents                    │                                     │
│  • AudioJobs, BatchJobs, Files                                     │
└──────────────────────────────────────────────────────────────────┘
         │
         │ REST + WebSocket
         │ (Agent API)
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    AGENT SERVICE (can be multiple)               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │           Agent Management API                             │  │
│  │  • POST /servers/start - Start llama.cpp                   │  │
│  │  • POST /servers/stop - Stop server                        │  │
│  │  • GET /servers - List running servers                     │  │
│  │  • GET /gpu/info - GPU information                         │  │
│  │  • GET /models - List model files                          │  │
│  │  • POST /models/download - Download model                  │  │
│  │  • DELETE /models/{id} - Delete model                      │  │
│  │  • WS /ws/status - Real-time events                        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │           Inference Proxy                                │  │
│  │  • Routes to llama.cpp or Halogen per server               │  │
│  │  • Preserves native OpenAI streaming envelopes            │  │
│  │  • Exposes cache and metrics telemetry                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│       Inference processes    │                                     │
│  (One subprocess per server) │                                     │
│  • Auto start on demand      │                                     │
│  • Auto shutdown on inactivity                                     │
│  • Configurable llama.cpp GPU layers or Halogen startup settings    │
│  • Per-server process with independent API and engine ports        │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    HARDWARE RESOURCES                            │
│  • GPU (CUDA/Metal/Vulkan)                                       │
│  • CPU                                                           │
│  • Disk (models + cache)                                         │
└──────────────────────────────────────────────────────────────────┘
```

## Service Components

### Frontend Service

**Technology Stack:**
- Python 3.14+ with FastAPI
- React + TypeScript + shadcn/ui
- PostgreSQL with SQLModel ORM


**Responsibilities:**
- Serve React WebUI
- OpenAI-compatible API endpoints
- Agent registration and discovery
- Model download management
- Conversation state management
- File storage management
- Batch processing
- Audio processing (via system Whisper)
- Prompt cache metadata tracking

**Key Features:**
- Multi-agent support (manual selection)
- Agent health monitoring via WebSocket
- Centralized database for all metadata

### Agent Service

**Technology Stack:**
- Python 3.14+ with FastAPI
- llama.cpp subprocess management
- WebSocket for real-time events
- psutil for system monitoring

**Responsibilities:**
- Auto-register with Frontend on startup
- Manage llama.cpp server lifecycle
- Proxy llama.cpp HTTP API for Frontend
- GPU monitoring and reporting
- Model file management (download, delete, validate)
- Real-time status updates via WebSocket
- Minimal status WebUI for debugging

**Key Features:**
- No authentication (trusted internal network)
- Full llama.cpp management
- GPU acceleration support (CUDA/Metal/Vulkan)
- Prompt cache file storage
- Multi-server support (one per model)

## Communication Patterns

### Agent Registration Flow

```
1. Agent Service starts
   ↓
2. Agent reads Frontend URL from config
   ↓
3. Agent → Frontend: POST /api/agents/register
   {
     "agent_id": "agent-uuid",
     "host": "agent-hostname",
     "port": 8080,
     "gpu_info": {...}
   }
   ↓
4. Frontend stores Agent in database
   ↓
5. Frontend → Agent: WebSocket connection established
   ↓
6. Agent streams real-time status updates
```

### Inference Request Flow

```
1. User sends chat completion request
   ↓
2. Frontend checks if server is running for model
   ↓ (if not running)
3. Frontend → Agent: POST /servers/start
   {
     "model_id": "uuid",
     "config": { gpu_layers, context_size, ... }
   }
   ↓
4. Agent starts llama.cpp subprocess
   ↓
5. Agent → Frontend (WS): server.started
   {
     "server_id": "uuid",
     "status": "running",
     "proxy_url": "http://agent:8080/proxy/server-uuid"
   }
   ↓
6. Frontend → Agent: POST /proxy/{server_id}/cache/load
   (Load prompt cache from disk if needed)
   ↓
7. Frontend → Agent: POST /proxy/{server_id}/chat/completions
   (Inference request)
   ↓
8. Agent proxies to llama.cpp, streams SSE back to Frontend
   ↓
9. Frontend → Agent: POST /proxy/{server_id}/cache/save
   (Save cache to disk after inference)
```

### Cache Management Flow

```
Frontend decides when to:
- Load cache: Before first inference in conversation
- Save cache: After inference completes
- Delete cache: When conversation deleted

All cache operations go through Agent proxy:
- POST /proxy/{server_id}/cache/load
- POST /proxy/{server_id}/cache/save
- DELETE /proxy/{server_id}/cache/{cache_id}

Cache files stored on Agent machine:
/agent-cache/{server_id}/{cache_id}.cache
```

## Data Models

### PostgreSQL Schema

```sql
-- Agents table (new)
agents:
  - id: UUID (primary key)
  - name: str
  - host: str (hostname or IP)
  - port: int
  - status: str (online, offline, unreachable)
  - gpu_info: JSON
  - last_seen: datetime
  - websocket_connected: bool
  - created_at: datetime

-- Updated ServerInstances
server_instances:
  - id: UUID
  - agent_id: UUID (foreign key → agents)  -- NEW
  - model_id: UUID (foreign key → models)
  - port: int (port on Agent machine)
  - proxy_url: str  -- NEW
  - status: str
  - ... (other fields unchanged)

-- PromptCache (metadata only)
prompt_cache:
  - id: UUID
  - cache_key: str
  - model_id: UUID
  - agent_id: UUID  -- Which agent stores the cache file
  - content_hash: str
  - hits: int
  - size_bytes: int
  - cache_path: str  -- Path on Agent machine
  - created_at: datetime
  - expires_at: datetime
```

## Agent API Specification

### REST Endpoints

**Agent Registration (called by Agent)**
```
POST /api/agents/register
Body: {
  "agent_id": "uuid",
  "name": "agent-name",
  "host": "hostname",
  "port": 8080,
  "gpu_info": {
    "name": "NVIDIA RTX 4090",
    "vram_total": 24576000000,
    "backend": "cuda"
  }
}
Response: { "registered": true, "frontend_version": "1.0.0" }
```

**Server Management**
```
POST /api/servers/start
Body: {
  "model_id": "uuid",
  "model_path": "/models/llama-3-8b.Q4_K_M.gguf",
  "config": {
    "gpu_layers": 35,
    "context_size": 4096,
    "batch_size": 512,
    "cache_prompt": true
  }
}
Response: {
  "server_id": "uuid",
  "status": "starting",
  "proxy_url": "http://agent:8080/proxy/server-uuid"
}

POST /api/servers/{server_id}/stop
Body: { "force": false }
Response: { "status": "stopped" }

GET /api/servers
Response: {
  "servers": [
    {
      "server_id": "uuid",
      "model_id": "uuid",
      "status": "running",
      "port": 8081,
      "uptime_seconds": 300
    }
  ]
}
```

**GPU Information**
```
GET /api/gpu/info
Response: {
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA RTX 4090",
      "vram_total": 24576000000,
      "vram_used": 8589934592,
      "vram_free": 15986065408,
      "utilization": 45,
      "backend": "cuda"
    }
  ]
}
```

**Model Management**
```
GET /api/models
Response: {
  "models": [
    {
      "filename": "llama-3-8b.Q4_K_M.gguf",
      "path": "/models/llama-3-8b.Q4_K_M.gguf",
      "size_bytes": 4916677728,
      "architecture": "llama",
      "quantization": "Q4_K_M"
    }
  ]
}

POST /api/models/download
Body: {
  "source": "huggingface",
  "repo_id": "TheBloke/Llama-3-8B-Instruct-GGUF",
  "filename": "llama-3-8b-instruct.Q4_K_M.gguf"
}
Response: { "job_id": "uuid", "status": "downloading" }

DELETE /api/models/{filename}
Response: { "deleted": true }
```

**Health Check**
```
GET /api/health
Response: {
  "status": "healthy",
  "uptime_seconds": 3600,
  "running_servers": 2
}
```

### WebSocket Events

**Connection**
```
WS /api/ws/status
Headers: { "X-Agent-ID": "agent-uuid" }

Frontend maintains persistent WebSocket connection to each Agent
```

**Events (Agent → Frontend)**
```json
// Server started
{
  "event": "server.started",
  "data": {
    "server_id": "uuid",
    "model_id": "uuid",
    "port": 8081,
    "proxy_url": "http://agent:8080/proxy/server-uuid"
  }
}

// Server stopped
{
  "event": "server.stopped",
  "data": {
    "server_id": "uuid",
    "reason": "graceful" | "crash" | "inactivity"
  }
}

// Server health update
{
  "event": "server.health",
  "data": {
    "server_id": "uuid",
    "status": "healthy" | "unhealthy",
    "requests_total": 150,
    "tokens_generated": 45000
  }
}

// GPU usage update
{
  "event": "gpu.usage",
  "data": {
    "gpu_id": 0,
    "vram_used": 8589934592,
    "vram_free": 15986065408,
    "utilization": 45
  }
}

// Download progress
{
  "event": "download.progress",
  "data": {
    "job_id": "uuid",
    "progress_percent": 45.5,
    "bytes_downloaded": 2200000000,
    "total_bytes": 4916677728,
    "speed_mbps": 12.5
  }
}

// Agent status change
{
  "event": "agent.status",
  "data": {
    "status": "online" | "offline" | "reconnecting"
  }
}
```

## llama.cpp Proxy

The Agent proxies all llama.cpp HTTP API calls:

**Proxy Endpoints (on Agent)**
```
POST /proxy/{server_id}/v1/chat/completions
POST /proxy/{server_id}/v1/completions
POST /proxy/{server_id}/v1/embeddings
POST /proxy/{server_id}/cache/load
POST /proxy/{server_id}/cache/save
DELETE /proxy/{server_id}/cache/{cache_id}
```

**Proxy Behavior:**
- Agent forwards request to llama.cpp subprocess
- Streams SSE responses back to Frontend in real-time
- Handles connection pooling and retries
- Logs all proxied requests for debugging

## Deployment Architecture

### Docker Compose Setup

```yaml
services:
  # Frontend Service
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"  # WebUI
      - "8000:8000"  # API
    volumes:
      - ./data:/data
      - ./models:/models:ro  # Read-only, models synced from Agent
    environment:
      - DATABASE_URL=postgresql+psycopg://...
      - AGENT_DISCOVERY_URL=http://agent:8080
    depends_on:
      - postgres

  # Agent Service
  agent:
    build: ./agent
    ports:
      - "8080:8080"  # Agent API
    volumes:
      - ./models:/models
      - ./cache:/cache
    environment:
      - FRONTEND_URL=http://frontend:8000
      - AGENT_ID=agent-1
    devices:
      - /dev/nvidia0:/dev/nvidia0  # GPU passthrough
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  # Database
  postgres:
    image: postgres:16
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=inference
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=inference_matrix

volumes:
  postgres_data:
```

### Multi-Agent Deployment

```yaml
# Add multiple agents for different hardware
services:
  agent-gpu-nvidia:
    # ... NVIDIA GPU config

  agent-gpu-amd:
    # ... AMD GPU config
    environment:
      - AGENT_ID=agent-amd

  agent-cpu-only:
    # ... CPU-only config
    environment:
      - AGENT_ID=agent-cpu
```

## Failure Handling

### Agent Unavailable

**Detection:**
- Frontend monitors WebSocket connection
- Heartbeat every 30 seconds
- Mark Agent as "offline" after 90 seconds no contact

**Recovery:**
1. Frontend marks all Agent's servers as "orphaned"
2. Existing llama.cpp servers continue running (no management)
3. New inference requests to that Agent are queued
4. When Agent reconnects:
   - Re-synchronize server state
   - Resume queued requests
   - Update server health status

### llama.cpp Server Crash

**Detection:**
- Agent monitors subprocess health
- Health check every 10 seconds

**Recovery:**
1. Agent detects crash
2. Agent → Frontend (WS): server.stopped (reason: crash)
3. Frontend can request restart: POST /servers/start
4. Agent restarts llama.cpp with same config
5. Agent → Frontend (WS): server.started

### Frontend Unavailable

**Agent Behavior:**
1. Agent continues running all llama.cpp servers
2. Agent retries WebSocket connection every 30 seconds
3. Agent buffers events in memory (max 1000 events)
4. When Frontend reconnects:
   - Agent replays buffered events
   - Re-synchronizes state

## Security Considerations

**Internal Network Trust:**
- No authentication between Frontend and Agent
- Assumes trusted internal network
- Firewall rules should restrict access

**External Access:**
- Frontend exposes API to external clients
- Agent should NEVER be exposed externally
- All external traffic goes through Frontend

**Data Protection:**
- No secrets in logs
- HTTPS at Frontend level (Traefik)

## Monitoring & Observability

**Frontend Metrics:**
- Agent connection status (per Agent)
- Server count and status
- Request latency (end-to-end)
- Cache hit/miss rates
- Token generation rates

**Agent Metrics:**
- GPU VRAM usage
- GPU utilization
- llama.cpp process health
- Model download progress
- Cache file sizes

**Export Formats:**
- Prometheus metrics endpoint (Frontend)
- WebSocket real-time updates (Agent → Frontend)
- Structured JSON logs

## Extensibility

**Future Enhancements:**
- Automatic agent load balancing
- Agent failover and high availability
- Distributed cache storage
- Multi-tenant support
- Additional model formats
- Native TTS integration on Agent
