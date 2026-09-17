# API Endpoints Specification

Inference Matrix provides an OpenAI-compatible REST API. All endpoints support optional authentication via API keys.

## Base URL

```
http://localhost:8000/v1
```

## Authentication

API keys are passed via the `Authorization` header:

```bash
Authorization: Bearer YOUR_API_KEY
```

Authentication is optional and can be enabled/disabled via configuration.

---

## Models

### List Models

**GET** `/v1/models`

Returns a list of available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.2-3b-instruct.Q4_K_M.gguf",
      "object": "model",
      "created": 1234567890,
      "owned_by": "inference-matrix"
    }
  ]
}
```

---

## Chat Completions

### Create Chat Completion

**POST** `/v1/chat/completions`

Creates a model response for the given chat conversation.

**Request Body:**
```json
{
  "model": "llama-3.2-3b-instruct.Q4_K_M.gguf",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Parameters:**
- `model` (string, required): Model ID to use
- `messages` (array, required): List of messages (system, user, assistant)
- `stream` (boolean, optional): Enable SSE streaming (default: false)
- `temperature` (number, optional): Sampling temperature (0-2)
- `max_tokens` (number, optional): Maximum tokens to generate
- `top_p` (number, optional): Nucleus sampling parameter
- `frequency_penalty` (number, optional): Frequency penalty (-2 to 2)
- `presence_penalty` (number, optional): Presence penalty (-2 to 2)
- `stop` (string|array, optional): Stop sequences
- `tools` (array, optional): Tool definitions for function calling
- `tool_choice` (string|object, optional): Tool choice strategy

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama-3.2-3b-instruct.Q4_K_M.gguf",
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

---

## Completions (Legacy)

### Create Completion

**POST** `/v1/completions`

Creates a completion for the given prompt (GPT-3 style legacy endpoint).

**Request Body:**
```json
{
  "model": "llama-3.2-3b-instruct.Q4_K_M.gguf",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Parameters:**
- `model` (string, required): Model ID to use
- `prompt` (string|array, required): Prompt text or tokens
- `stream` (boolean, optional): Enable SSE streaming
- `max_tokens` (number, optional): Maximum tokens to generate
- `temperature` (number, optional): Sampling temperature
- `top_p` (number, optional): Nucleus sampling
- `stop` (string|array, optional): Stop sequences

**Response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "llama-3.2-3b-instruct.Q4_K_M.gguf",
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

## Embeddings

### Create Embedding

**POST** `/v1/embeddings`

Creates embeddings for the given input text.

**Request Body:**
```json
{
  "model": "nomic-embed-text-v1.5.Q4_K_M.gguf",
  "input": "The quick brown fox jumps over the lazy dog",
  "encoding_format": "float"
}
```

**Parameters:**
- `model` (string, required): Embedding model ID
- `input` (string|array, required): Input text(s) to embed
- `encoding_format` (string, optional): `float` or `base64` (default: float)

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
  "model": "nomic-embed-text-v1.5.Q4_K_M.gguf",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9
  }
}
```

---

## Responses API

### Create Response

**POST** `/v1/responses`

Creates a response using the new Responses API with support for tree-structured conversations.

**Request Body:**
```json
{
  "model": "llama-3.2-3b-instruct.Q4_K_M.gguf",
  "input": [
    {"type": "message", "role": "user", "content": "Hello!"}
  ],
  "previous_response_id": null,
  "stream": false,
  "temperature": 0.7,
  "max_output_tokens": 512,
  "tools": [],
  "tool_choice": "auto",
  "reasoning": {
    "effort": "medium",
    "summary": "concise"
  }
}
```

**Parameters:**
- `model` (string, required): Model ID to use
- `input` (string|array, required): Input items (messages, function calls, etc.)
- `previous_response_id` (string, optional): ID of prior response for multi-turn
- `include` (array, optional): Additional fields to include
- `tools` (array, optional): Available tools
- `tool_choice` (string|object, optional): Tool selection
- `metadata` (object, optional): Custom metadata (16 key-value pairs)
- `temperature` (number, optional): Sampling temperature
- `top_p` (number, optional): Nucleus sampling
- `max_output_tokens` (number, optional): Max tokens to generate
- `reasoning` (object, optional): Reasoning configuration
  - `effort`: `none`|`low`|`medium`|`high`|`xhigh`
  - `summary`: `concise`|`detailed`|`auto`
- `stream` (boolean, optional): Enable SSE streaming
- `store` (boolean, optional): Store response for retrieval (default: true)
- `truncation` (string, optional): `auto`|`disabled`

**Input Item Types:**
- `message` (user/assistant/system/developer)
- `function_call`
- `function_call_output`
- `reasoning`
- `compaction`

**Response:**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1234567890,
  "model": "llama-3.2-3b-instruct.Q4_K_M.gguf",
  "input": [...],
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

**Streaming Events:**
- `response.created`
- `response.in_progress`
- `response.output_item.added`
- `response.output_item.done`
- `response.completed`

---

## Files

### List Files

**GET** `/v1/files`

Returns a list of uploaded files.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "file-abc123",
      "object": "file",
      "bytes": 1234567,
      "created_at": 1234567890,
      "filename": "dataset.jsonl",
      "purpose": "batch",
      "status": "uploaded"
    }
  ]
}
```

### Upload File

**POST** `/v1/files`

Uploads a file.

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

### Retrieve File

**GET** `/v1/files/{file_id}`

Returns information about a specific file.

**Response:** Same structure as List Files

### Delete File

**DELETE** `/v1/files/{file_id}`

Deletes a file.

**Response:**
```json
{
  "id": "file-abc123",
  "object": "file",
  "deleted": true
}
```

### Retrieve File Content

**GET** `/v1/files/{file_id}/content`

Returns the content of a file.

---

## Batches

### Create Batch

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

**Parameters:**
- `input_file_id` (string, required): File ID with batch requests
- `endpoint` (string, required): Target endpoint
- `completion_window` (string, optional): Max processing time

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

### Retrieve Batch

**GET** `/v1/batches/{batch_id}`

Returns information about a batch job.

### Cancel Batch

**POST** `/v1/batches/{batch_id}/cancel`

Cancels a batch job.

### List Batches

**GET** `/v1/batches`

Returns a list of batch jobs.

---

## Audio

### Create Transcription

**POST** `/v1/audio/transcriptions`

Transcribes audio to text.

**Request Body (multipart/form-data):**
- `file` (file, required): Audio file (wav, mp3, etc.)
- `model` (string, required): Whisper model ID
- `language` (string, optional): Language code (e.g., "en")
- `prompt` (string, optional): Transcription hint
- `response_format` (string, optional): `json`|`text`|`srt`|`verbose_json`
- `temperature` (number, optional): Sampling temperature

**Response:**
```json
{
  "text": "Hello, this is the transcription of your audio file."
}
```

### Create Translation

**POST** `/v1/audio/translations`

Translates audio to English text.

**Request Body (multipart/form-data):**
- `file` (file, required): Audio file
- `model` (string, required): Whisper model ID
- `prompt` (string, optional): Translation hint
- `response_format` (string, optional): `json`|`text`|`srt`|`verbose_json`
- `temperature` (number, optional): Sampling temperature

**Response:**
```json
{
  "text": "Hello, this is the translated text."
}
```

### Create Speech

**POST** `/v1/audio/speech`

Generates speech from text (TTS).

**Request Body:**
```json
{
  "model": "tts-model",
  "input": "Hello, this is text to speech.",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Parameters:**
- `model` (string, required): TTS model ID
- `input` (string, required): Text to synthesize
- `voice` (string, optional): Voice ID
- `response_format` (string, optional): `mp3`|`wav`|`flac`|`opus`
- `speed` (number, optional): Speech speed (0.25-4.0)

**Response:** Audio file (binary)

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
- `401` - Unauthorized (invalid/missing API key)
- `404` - Not Found (model/resource not found)
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
- `503` - Service Unavailable (model loading)

**Error Types:**
- `invalid_request_error` - Invalid parameters
- `authentication_error` - Invalid API key
- `not_found_error` - Resource not found
- `rate_limit_error` - Too many requests
- `server_error` - Internal error
- `model_error` - Model loading/execution error

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
