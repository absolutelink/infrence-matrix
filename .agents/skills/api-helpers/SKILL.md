---
name: api-helpers
description: OpenAI API compatibility helpers and utilities for endpoint implementation
---

# API Helpers Skill

Use this skill when implementing or debugging OpenAI-compatible API endpoints.

## OpenAI API Compatibility

### Request Validation

```python
from app.utils.api_helpers import validate_chat_request, ValidationError

try:
    validated = validate_chat_request(request_body)
    # validated.model, validated.messages, validated.temperature, etc.
except ValidationError as e:
    return JSONResponse(status_code=400, content=e.to_openai_error())
```

**Validates:**
- Required fields (model, messages)
- Field types and ranges
- Message structure
- Tool definitions
- Streaming parameters

### Response Formatting

```python
from app.utils.api_helpers import format_chat_response, format_error_response

# Success response
response = format_chat_response(
    model="llama-2-7b-chat.Q4_K_M.gguf",
    choices=[...],
    usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
)

# Error response
error = format_error_response(
    message="Model not found",
    error_type="not_found_error",
    code="model_not_found",
    param="model"
)
```

### SSE Streaming

```python
from app.utils.api_helpers import sse_stream

async def stream_response():
    async with sse_stream() as stream:
        # Send chunks
        await stream.send({
            "id": "chatcmpl-123",
            "choices": [{"delta": {"content": "Hello"}, "index": 0}]
        })
        
        # Send usage at end
        await stream.send({
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        })
        
        # Send DONE
        await stream.close()
```

## API Key Authentication

### Validate API Key

```python
from app.utils.api_helpers import validate_api_key

async def get_current_user(authorization: str):
    api_key = extract_api_key(authorization)
    is_valid, key_info = await validate_api_key(api_key)
    
    if not is_valid:
        raise HTTPException(401, "Invalid API key")
    
    return key_info
```

### Extract API Key

```python
from app.utils.api_helpers import extract_api_key

# From Authorization header
api_key = extract_api_key("Bearer sk-abc123...")

# From X-API-Key header
api_key = extract_api_key(headers.get("X-API-Key"))
```

### Check Permissions

```python
from app.utils.api_helpers import check_permissions

await check_permissions(
    api_key_info=key_info,
    required_permissions=["chat", "embeddings"]
)
```

## Rate Limiting

### Apply Rate Limits

```python
from app.utils.api_helpers import rate_limit, RateLimitExceeded

@rate_limit(requests_per_minute=60, tokens_per_minute=10000)
async def create_chat_completion(request):
    ...
```

### Get Rate Limit Headers

```python
from app.utils.api_helpers import get_rate_limit_headers

headers = get_rate_limit_headers(
    limit_requests=60,
    remaining_requests=45,
    reset_requests=30,
    limit_tokens=10000,
    remaining_tokens=8000,
    reset_tokens=45
)
```

## Token Counting

### Count Tokens

```python
from app.utils.api_helpers import count_tokens

# Count message tokens
num_tokens = count_tokens(
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    model="llama-2-7b-chat"
)

# Count single text
num_tokens = count_tokens(text="Hello world")
```

### Estimate Context Usage

```python
from app.utils.api_helpers import estimate_context_usage

usage = estimate_context_usage(
    messages=messages,
    max_tokens=512,
    context_length=4096
)
# usage.total_tokens, usage.available, usage.will_fit
```

## Tool/Function Calling

### Parse Tool Calls

```python
from app.utils.api_helpers import parse_tool_calls

tool_calls = parse_tool_calls(
    response_text=llm_output,
    available_tools=[tool1, tool2]
)
```

### Format Tool Response

```python
from app.utils.api_helpers import format_tool_response

response = format_tool_response(
    tool_call_id="call_123",
    output={"result": "success"}
)
```

## Error Handling

### OpenAI Error Format

```python
from app.utils.api_helpers import OpenAIError

# Create error
error = OpenAIError(
    message="Invalid model specified",
    error_type="invalid_request_error",
    code="model_not_found",
    param="model"
)

# Convert to response
response = error.to_response(status_code=400)

# Raise as exception
raise error.to_http_exception()
```

### Common Errors

```python
# Model not found
raise model_not_found_error(model_id)

# Invalid request
raise invalid_request_error("messages is required", param="messages")

# Authentication error
raise authentication_error("Invalid API key")

# Rate limit
raise rate_limit_error(retry_after=60)

# Server error
raise server_error("Model loading failed")
```

## Request ID Tracking

### Generate Request ID

```python
from app.utils.api_helpers import generate_request_id

request_id = generate_request_id()  # req_abc123xyz
```

### Add to Response

```python
from app.utils.api_helpers import add_request_id

response = add_request_id(
    response_object,
    request_id="req_abc123"
)
```

### Log Request ID

```python
from app.utils.api_helpers import log_request

log_request(
    request_id="req_abc123",
    endpoint="/v1/chat/completions",
    model="llama-2-7b",
    user_id="user_123"
)
```

## Streaming Utilities

### Chunk Formatter

```python
from app.utils.api_helpers import ChatCompletionChunk

chunk = ChatCompletionChunk(
    id="chatcmpl-123",
    model="llama-2-7b",
    choices=[{
        "index": 0,
        "delta": {"content": "Hello"},
        "finish_reason": None
    }]
)

json_str = chunk.to_json()
```

### Usage Aggregator

```python
from app.utils.api_helpers import UsageAggregator

aggregator = UsageAggregator()
aggregator.add_prompt_tokens(10)
aggregator.add_completion_tokens(20)

usage = aggregator.get_usage()
# {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
```

## Model Resolution

### Resolve Model ID

```python
from app.utils.api_helpers import resolve_model

model_info = await resolve_model("llama-2-7b-chat.Q4_K_M.gguf")
# Returns: model_id, path, config, is_loaded

# Check if loaded
if not model_info.is_loaded:
    await load_model(model_info.id)
```

### List Available Models

```python
from app.utils.api_helpers import list_models

models = await list_models()
# Returns OpenAI-compatible model list format
```

## Best Practices

1. **Always validate requests** before processing
2. **Use OpenAI error format** for all errors
3. **Include request IDs** in all responses
4. **Log all API calls** with timing
5. **Rate limit by IP and API key**
6. **Stream large responses** to reduce memory
7. **Count tokens accurately** for usage tracking

## Configuration

API config in `/etc/inference-matrix/api.yaml`:

```yaml
api:
  base_url: /v1
  max_request_size_mb: 10
  timeout_seconds: 300
  
authentication:
  enabled: false
  header: Authorization
  key_prefix: sk-
  
rate_limits:
  default_requests_per_minute: 60
  default_tokens_per_minute: 10000
  
logging:
  include_request_body: false
  include_response_body: false
  log_request_ids: true
```
