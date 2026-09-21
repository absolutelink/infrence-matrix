# API Endpoints Specification

Inference Matrix provides an OpenAI-compatible REST API through the Frontend Service. The Frontend Service communicates with Agent Services to manage inference.

## Base URLs

**Frontend Service:**
```
http://localhost:8000/v1
```

**Agent Service (internal):**
```
http://agent-hostname:8080/api
```

**Agent Service has no authentication** (trusted internal network).

---

## Frontend Service API

### Models

#### List Models

**GET** `/v1/models`

Returns a list of available models across all Agents.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3-8b.Q4_K_M.gguf",
      "object": "model",
      "created": 1234567890,
      "owned_by": "inference-matrix",
      "agent_id": "agent-uuid"
    }
  ]
}
```

---

### Chat Completions

#### Create Chat Completion

**POST** `/v1/chat/completions`

Creates a model response for the given chat conversation.

**Request Body:**
```json
{
  "model": "llama-3-8b.Q4_K_M.gguf",
  "agent_id": "agent-uuid",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 512,
  "prompt_cache_options": {
    "enabled": true,
    "ttl": 1800
  }
}
```

**Parameters:**
- `model` (string, required): Model ID to use
- `agent_id` (string, optional): Which Agent to use (required if model exists on multiple Agents)
- `messages` (array, required): List of messages
- `stream` (boolean, optional): Enable SSE streaming (default: false)
- `temperature` (number, optional): Sampling temperature (0-2)
- `max_tokens` (number, optional): Maximum tokens to generate
- `top_p` (number, optional): Nucleus sampling parameter
- `frequency_penalty` (number, optional): Frequency penalty (-2 to 2)
- `presence_penalty` (number, optional): Presence penalty (-2 to 2)
- `stop` (string|array, optional): Stop sequences
- `tools` (array, optional): Tool definitions for function calling
- `prompt_cache_options` (object, optional): Cache configuration
  - `enabled`: boolean
  - `ttl`: seconds

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama-3-8b.Q4_K_M.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**Streaming Response (SSE):**
```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"!"},"index":0}]}

data: [DONE]
```

**Flow:**
1. Frontend checks if server is running for model on specified Agent
2. If not running: Frontend → Agent: POST /api/servers/start
3. Frontend → Agent: POST /proxy/{server_id}/cache/load (if cache enabled)
4. Frontend → Agent: POST /proxy/{server_id}/v1/chat/completions
5. Agent proxies to llama.cpp, streams SSE back
6. Frontend → Agent: POST /proxy/{server_id}/cache/save (if cache enabled)

---

### Completions (Legacy)

#### Create Completion

**POST** `/v1/completions`

Creates a completion for the given prompt (GPT-3 style legacy endpoint).

**Request Body:**
```json
{
  "model": "llama-3-8b.Q4_K_M.gguf",
  "agent_id": "agent-uuid",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "llama-3-8b.Q4_K_M.gguf",
  "choices": [
    {
      "text": ", there lived a brave knight...",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 12,
    "total_tokens": 16
  }
}
```

---

### Embeddings

#### Create Embedding

**POST** `/v1/embeddings`

Creates embeddings for the given input text.

**Request Body:**
```json
{
  "model": "nomic-embed-text.Q4_K_M.gguf",
  "agent_id": "agent-uuid",
  "input": "The quick brown fox jumps over the lazy dog",
  "encoding_format": "float"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0123, 0.456, ...],
      "index": 0
    }
  ],
  "model": "nomic-embed-text.Q4_K_M.gguf",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9
  }
}
```

---

### Responses API

#### Create Response

**POST** `/v1/responses`

Creates a response using the new Responses API with support for tree-structured conversations.

**Request Body:**
```json
{
  "model": "llama-3-8b.Q4_K_M.gguf",
  "agent_id": "agent-uuid",
  "input": [
    {"type": "message", "role": "user", "content": "Hello!"}
  ],
  "previous_response_id": null,
  "stream": false,
  "temperature": 0.7,
  "max_output_tokens": 512,
  "store": true
}
```

**Response:**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1234567890,
  "model": "llama-3-8b.Q4_K_M.gguf",
  "output": [
    {
      "type": "message",
      "id": "msg_abc123",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Hello! How can I help you?"
        }
      ]
    }
  ],
  "status": "completed",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 8,
    "total_tokens": 18
  }
}
```

---

### Files

#### Upload File

**POST** `/v1/files`

Uploads a file for batch processing or retrieval.

**Request Body (multipart/form-data):**
- `file` (file, required): File to upload
- `purpose` (string, required): `batch`|`retrieval`|`assistants`

**Response:**
```json
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 1234567,
  "created_at": 1234567890,
  "filename": "dataset.jsonl",
  "purpose": "batch",
  "status": "uploaded"
}
```

#### List Files

**GET** `/v1/files`

Returns a list of uploaded files.

---

### Batches

#### Create Batch

**POST** `/v1/batches`

Creates a batch job for async processing.

**Request Body:**
```json
{
  "input_file_id": "file-abc123",
  "endpoint": "/v1/chat/completions",
  "completion_window": "24h"
}
```

**Response:**
```json
{
  "id": "batch_abc123",
  "object": "batch",
  "created_at": 1234567890,
  "endpoint": "/v1/chat/completions",
  "input_file_id": "file-abc123",
  "status": "validating",
  "completion_window": "24h"
}
```

---

### Audio

#### Create Transcription

**POST** `/v1/audio/transcriptions`

Transcribes audio to text using system Whisper.

**Request Body (multipart/form-data):**
- `file` (file, required): Audio file (wav, mp3, etc.)
- `model` (string, required): Whisper model ID
- `language` (string, optional): Language code
- `response_format` (string, optional): `json`|`text`|`srt`|`verbose_json`

**Response:**
```json
{
  "text": "Hello, this is the transcription of your audio file."
}
```

#### Create Translation

**POST** `/v1/audio/translations`

Translates audio to English text.

#### Create Speech

**POST** `/v1/audio/speech`

Generates speech from text (TTS).

**Request Body:**
```json
{
  "model": "tts-model",
  "input": "Hello, this is text to speech.",
  "voice": "alloy",
  "response_format": "mp3"
}
```

**Response:** Audio file (binary)

---

## Agent Service API

### Agent Registration

#### Register Agent

**POST** `/api/agents/register`

Called by Agent on startup to register with Frontend.

**Request Body:**
```json
{
  "agent_id": "agent-uuid",
  "name": "inference-agent-1",
  "host": "agent-hostname",
  "port": 8080,
  "gpu_info": {
    "name": "NVIDIA RTX 4090",
    "vram_total": 24576000000,
    "backend": "cuda"
  }
}
```

**Response:**
```json
{
  "registered": true,
  "frontend_version": "1.0.0"
}
```

---

### Server Management

#### Start Server

**POST** `/api/servers/start`

Start a llama.cpp server for a model.

**Request Body:**
```json
{
  "model_id": "uuid",
  "model_path": "/models/llama-3-8b.Q4_K_M.gguf",
  "config": {
    "gpu_layers": 35,
    "context_size": 4096,
    "batch_size": 512,
    "cache_prompt": true,
    "flash_attn": true
  }
}
```

**Response:**
```json
{
  "server_id": "server-uuid",
  "status": "starting",
  "proxy_url": "http://agent:8080/proxy/server-uuid"
}
```

#### Stop Server

**POST** `/api/servers/{server_id}/stop`

Stop a running llama.cpp server.

**Request Body:**
```json
{
  "force": false
}
```

**Response:**
```json
{
  "status": "stopped"
}
```

#### List Servers

**GET** `/api/servers`

List all running servers on this Agent.

**Response:**
```json
{
  "servers": [
    {
      "server_id": "uuid",
      "model_id": "uuid",
      "model_path": "/models/llama-3-8b.Q4_K_M.gguf",
      "status": "running",
      "port": 8081,
      "uptime_seconds": 300,
      "requests_total": 150
    }
  ]
}
```

---

### GPU Information

#### Get GPU Info

**GET** `/api/gpu/info`

Get GPU information and current usage.

**Response:**
```json
{
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA RTX 4090",
      "vram_total": 24576000000,
      "vram_used": 8589934592,
      "vram_free": 15986065408,
      "utilization": 45,
      "temperature": 65,
      "backend": "cuda"
    }
  ]
}
```

---

### Model Management

#### List Models

**GET** `/api/models`

List all model files available on this Agent.

**Response:**
```json
{
  "models": [
    {
      "filename": "llama-3-8b.Q4_K_M.gguf",
      "path": "/models/llama-3-8b.Q4_K_M.gguf",
      "size_bytes": 4916677728,
      "architecture": "llama",
      "quantization": "Q4_K_M",
      "parameter_count": 8000000000
    }
  ]
}
```

#### Download Model

**POST** `/api/models/download`

Download a model from HuggingFace or ModelScope.

**Request Body:**
```json
{
  "source": "huggingface",
  "repo_id": "TheBloke/Llama-3-8B-Instruct-GGUF",
  "filename": "llama-3-8b-instruct.Q4_K_M.gguf"
}
```

**Response:**
```json
{
  "job_id": "download-uuid",
  "status": "downloading",
  "progress_percent": 0.0
}
```

#### Delete Model

**DELETE** `/api/models/{filename}`

Delete a model file from disk.

**Response:**
```json
{
  "deleted": true
}
```

---

### Health Check

#### Get Health

**GET** `/api/health`

Check Agent health status.

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "running_servers": 2,
  "gpu_status": "normal"
}
```

---

### WebSocket

#### Status Stream

**WS** `/api/ws/status`

Persistent WebSocket connection for real-time events.

**Headers:**
```
X-Agent-ID: agent-uuid
```

**Events (Agent → Frontend):**

Server started:
```json
{
  "event": "server.started",
  "data": {
    "server_id": "uuid",
    "model_id": "uuid",
    "port": 8081,
    "proxy_url": "http://agent:8080/proxy/server-uuid"
  }
}
```

Server stopped:
```json
{
  "event": "server.stopped",
  "data": {
    "server_id": "uuid",
    "reason": "graceful"
  }
}
```

GPU usage update:
```json
{
  "event": "gpu.usage",
  "data": {
    "gpu_id": 0,
    "vram_used": 8589934592,
    "vram_free": 15986065408,
    "utilization": 45
  }
}
```

Download progress:
```json
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
```

---

## llama.cpp Proxy Endpoints (on Agent)

All llama.cpp API calls are proxied through the Agent:

### Chat Completions Proxy

**POST** `/proxy/{server_id}/v1/chat/completions`

Proxied to llama.cpp's `/v1/chat/completions` endpoint.

### Completions Proxy

**POST** `/proxy/{server_id}/v1/completions`

Proxied to llama.cpp's `/v1/completions` endpoint.

### Embeddings Proxy

**POST** `/proxy/{server_id}/v1/embeddings`

Proxied to llama.cpp's `/v1/embeddings` endpoint.

### Cache Operations

#### Load Cache

**POST** `/proxy/{server_id}/cache/load`

Load prompt cache from disk.

**Request Body:**
```json
{
  "cache_key": "conversation:uuid",
  "cache_path": "/cache/server-uuid/cache-abc123.cache"
}
```

**Response:**
```json
{
  "loaded": true,
  "tokens_cached": 256
}
```

#### Save Cache

**POST** `/proxy/{server_id}/cache/save`

Save prompt cache to disk.

**Request Body:**
```json
{
  "cache_key": "conversation:uuid",
  "cache_path": "/cache/server-uuid/cache-abc123.cache",
  "ttl_seconds": 1800
}
```

**Response:**
```json
{
  "saved": true,
  "cache_id": "cache-uuid",
  "size_bytes": 1048576
}
```

#### Delete Cache

**DELETE** `/proxy/{server_id}/cache/{cache_id}`

Delete cached prompt from disk.

**Response:**
```json
{
  "deleted": true
}
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "error": {
    "message": "Human-readable error message",
    "type": "invalid_request_error",
    "code": "model_not_found",
    "param": "model"
  }
}
```

**HTTP Status Codes:**
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (model/resource not found)
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
- `503` - Service Unavailable (model loading or Agent offline)

---

## Rate Limits

When rate limiting is enabled, the following headers are returned:

- `x-ratelimit-limit-requests` - Max requests per minute
- `x-ratelimit-remaining-requests` - Remaining requests
- `x-ratelimit-reset-requests` - Seconds until reset
- `x-ratelimit-limit-tokens` - Max tokens per minute
- `x-ratelimit-remaining-tokens` - Remaining tokens
- `x-ratelimit-reset-tokens` - Seconds until reset

---

## Request IDs

All responses include an `x-request-id` header for debugging:

```bash
curl -i http://localhost:8000/v1/models
# x-request-id: req_abc123xyz
```

Include this ID when reporting issues.
