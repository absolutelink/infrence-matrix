---
name: api-helpers
description: OpenAI-compatible endpoint implementation patterns for the Inference Matrix admin (litellm-driven /v1 API, SSE streaming, error mapping)
---

# API Helpers — Admin `/v1` Patterns

Use this skill when implementing or debugging the public OpenAI-compatible
API in `admin/backend/app/api/v1/`. The admin drives **litellm** against a
scheduler-admitted provider instance and owns the client-facing contract.
Architecture: `ARCHITECTURE.md` §7. Fidelity basis:
`spike/litellm-fidelity/FINDINGS.md`.

## Core principles

1. **litellm does the provider talking; the admin owns the contract.**
   Never hand-roll upstream HTTP to a backend. Call
   `litellm.aresponses` / `litellm.acompletion` with
   `custom_llm_provider="openai"` and `api_base` pointed at the provider
   instance port.
2. **The admin owns the `resp_<uuid>`.** Mint it yourself; overwrite
   `response.id` on every lifecycle frame via `SSEEmitter`. Store litellm's
   wrapped id in `ResponseRecord.parameters` for affinity — never surface it
   to the client.
3. **The admin owns the conversation chain.** Reconstruct full `input`
   from Postgres. Do **not** pass `previous_response_id` to litellm.
4. **Register the alias before calling litellm**, or native streaming isn't
   selected and you get a confusing `APIError`.
5. **Admission is the scheduler's job**, not the route's. Acquire before
   streaming; release in a cancellation-safe `finally`.

## Endpoint skeleton (`POST /v1/responses`)

```python
@router.post("/v1/responses")
async def create_response(request: Request) -> Any:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")

    alias = body.get("model")
    input_value = body.get("input")
    if not alias or input_value is None:
        raise HTTPException(400, "'model' and 'input' are required")

    # 1. Resolve the enabled ProviderDefinition by alias (404 if missing/disabled).
    # 2. Load prior ResponseRecord for previous_response_id (404 if not found).
    # 3. Mint the admin-owned client id.
    client_response_id = f"resp_{uuid.uuid4().hex}"

    # 4. Register litellm native streaming for this alias (idempotent).
    ensure_registered(alias)

    # 5. Reconstruct litellm input from the DB chain.
    litellm_input = build_litellm_input(body, previous)

    # 6. Admit via the scheduler.
    scheduler = get_scheduler(request)
    try:
        admission = await scheduler.acquire(alias, client_response_id)
    except NoProviderAvailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(504, str(exc)) from exc

    # 7. Forward only whitelisted passthrough fields.
    passthrough = {k: body[k] for k in PASSTHROUGH_FIELDS if k in body}
    ...
```

## Passthrough fields

Forward only a known whitelist to litellm; never forward arbitrary client
JSON. Current set (`app/api/v1/responses.py`):

`instructions, tools, tool_choice, temperature, top_p, max_output_tokens,
reasoning, text, truncation, parallel_tool_calls, metadata, user, store,
background, include`

## Calling litellm

```python
stream = await litellm.aresponses(
    model=alias,                         # bare alias; provider is "openai"
    custom_llm_provider="openai",
    api_base=f"{admission.base_url}/v1", # base_url has NO /v1; append here
    stream=True,
    input=litellm_input,
    **passthrough,
)
```

- `admission.base_url` is `http://{machine.reachable_address()}:{port}`
  (no `/v1`). The caller appends `/v1`.
- Chat completions use `litellm.acompletion` the same way (`mode` differs
  in the alias registration).

## Alias registration (REQUIRED)

```python
from app.services.alias_registry import ensure_registered

ensure_registered(alias)   # idempotent, process-cached
```

Registers with litellm as:

```python
litellm.register_model({alias: {
    "supports_native_streaming": True,
    "litellm_provider": "openai",
    "mode": "responses",          # "chat" for completions
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
}})
```

Provider-definition create/update flows should also call this so aliases are
warm before first use.

## SSE emitter

```python
from app.services.sse import SSEEmitter, to_dict

emitter = SSEEmitter(client_response_id)
async for event in stream:
    yield emitter.frame(event)      # replaces id on lifecycle frames,
                                   # reassigns sequence_number, passes the rest
yield emitter.done()                # "data: [DONE]\n\n"
```

- Lifecycle frames whose `response.id` is rewritten: `response.created`,
  `response.in_progress`, `response.completed`, `response.failed`,
  `response.incomplete`.
- `sequence_number` is reassigned monotonically (0,1,2,...) on every frame.
- Non-canonical events (e.g. `response.reasoning_text.delta`) pass through
  untouched — native streaming preserves them.
- `event_type_of()` unwraps litellm's `ResponsesAPIStreamEvents` enum
  members to their dotted string so the `event:` line is spec-clean.
- `to_dict()` tolerantly converts litellm pydantic-object or dict events.

## Error mapping

litellm does not yield a `response.failed` event — it raises
`MidStreamFallbackError`. Map exceptions to the spec terminal frames:

```python
try:
    async for event in stream:
        ...
except Exception as exc:   # MidStreamFallbackError and friends
    error = map_exception_to_error(exc)
    for frame in emitter.failed(error):
        yield frame
    yield emitter.done()
```

`map_exception_to_error`:
- `litellm.exceptions.NotFoundError` →
  `{"type": "invalid_request_error", "code": "model_not_found"}`
- everything else (MidStreamFallbackError, APIError, connection errors) →
  `{"type": "server_error", "code": "upstream_failed"}`

`SSEEmitter.failed(error)` returns `[response.failed frame, error frame]` —
`yield from` it.

## Cancellation-safe cleanup (CRITICAL)

The scheduler slot must be released even if the client disconnects
mid-stream. Wrap teardown in `asyncio.shield` and use `finally`:

```python
try:
    async for event in stream:
        yield emitter.frame(event)
    yield emitter.done()
finally:
    await asyncio.shield(_teardown())   # close upstream, release slot, persist once
```

`InferenceScheduler.release()` is idempotent and shielded internally. The
provider frees its own slot on TCP close, so a dead admin never wedges a
backend.

## Persistence

On completion (stream or non-stream), persist exactly once:

- `ResponseRecord`: `response_id`=client id, `previous_response_id`,
  `input_items` (the FULL reconstructed input for this turn),
  `output_items`, token counts, `store`, `status`, and litellm's wrapped
  id in `parameters`.
- `TokenUsageSample`: prompt/cached/completion tokens + rates.

Because each record stores the full conversation up to that turn, a
continuation reconstructs in O(1) from its immediate predecessor — no
chain walk.

## Request ID

The client response id doubles as the scheduler `request_id`
(`request_id = client_response_id`). Keep them equal so acquire/release and
the DB row line up.

## 501 stubs (unimplemented endpoints)

For endpoints not yet in the core path, return the OpenAI error envelope:

```python
@router.post("/v1/embeddings")
async def embeddings() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"error": {
            "message": "embeddings are not implemented in this deployment",
            "type": "not_supported_error",
            "code": "endpoint_not_implemented",
            "param": None,
        }},
    )
```

Do **not** restore old implementations from `legacy/` — see Accepted
Regressions in `IMPLEMENTATION_STATUS.md`.

## Adding a new `/v1` endpoint checklist

1. Resolve the enabled `ProviderDefinition` by alias (404 otherwise).
2. `ensure_registered(alias)` with the right litellm `mode`.
3. `scheduler.acquire(...)` → 503/504 before the stream starts.
4. Call litellm with `api_base = admission.base_url + "/v1"`.
5. Frame the response with the admin-owned id + sequence numbers.
6. Map litellm exceptions to spec terminal frames.
7. Release + persist in a cancellation-safe `finally`.
8. `bash scripts/generate-client.sh` if the schema changed.
9. Run the OpenResponses conformance suite (see integration-testing skill).

## Anti-patterns

- ❌ Passing `previous_response_id` to litellm (admin owns the chain).
- ❌ Surfacing litellm's wrapped response id to the client.
- ❌ Calling a provider backend directly instead of through litellm.
- ❌ Forwarding un-whitelisted client fields to litellm.
- ❌ Releasing the scheduler slot only on normal completion (must be
  `finally` + shield).
- ❌ Reimplementing provider-specific translation in the admin — that lives
  in `provider/<type>` (see `provider/README.md`).
