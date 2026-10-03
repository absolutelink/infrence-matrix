# Phase 0 — litellm Responses API Fidelity Spike

Spike code: `spike/litellm-fidelity/` (litellm 1.103.2, Python 3.14).

Purpose: decide whether `litellm.aresponses(stream=True)` preserves enough of the
OpenResponses-spec event stream for the admin to re-emit a spec-perfect SSE to the
client, or whether we must keep a broker-side emitter and use litellm only for
chat/completions.

## How it was run

1. `toy_server.py` — FastAPI SSE server emitting the full 26-event set we care
   about: lifecycle (`created`/`in_progress`/`completed`), reasoning summary
   part/text added+delta+done, dual-name `response.reasoning_text.delta/done`,
   `output_item.added/done`, `content_part.added/done`, `output_text.delta/done`,
   `output_text.annotation.added`, `refusal.delta/done`,
   `function_call_arguments.delta/done`, terminal `completed` with populated
   `output[]` and full `usage` details, plus `sequence_number` on every event.
2. `spike_fidelity.py` / `spike2.py` — point `litellm.aresponses` at the toy
   server with `custom_llm_provider="openai"` + `api_base`, collect every event
   that survives, diff against what was emitted.
3. `spike5.py` — populated `output[]` (reasoning + message + function_call) on
   the terminal `response.completed`.
4. `spike6.py` — terminal `response.failed` path.

## Critical gotcha discovered first

Without registering the model, the spike received **0 events** and raised
`APIError: OpenAIException`. Root cause:

`OpenAIResponsesAPIConfig.should_fake_stream()` calls
`litellm.supports_native_streaming(model, custom_llm_provider)`, which returns
`False` when the model is unknown to litellm's model table. `fake_stream=True`
makes litellm feed the SSE body into `transform_response_api_response()` as if it
were a single non-streaming JSON payload → `JSONDecodeError` → mapped to
`APIError`.

**Resolution required for our provider aliases**: register each model so native
streaming is selected, e.g.

```python
litellm.register_model({
    alias: {
        "supports_native_streaming": True,
        "litellm_provider": "openai",
        "mode": "responses",
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    }
})
```

Alternative: ship a litellm `model_cost` JSON override in the admin image so
`/v1/models` sync registers aliases automatically. This must be part of the
provider-definition create/update flow, otherwise cold aliases break streaming.

## Results with native streaming enabled

`emitted=26 received=26`, `exact=23 modified=3 dropped=0`

| Aspect | Verdict |
|---|---|
| Event count / ordering | **Preserved exactly**, no drops, no reordering |
| `sequence_number` | **Preserved** verbatim when upstream sends it |
| Unknown / non-canonical event types (e.g. `response.reasoning_text.delta`) | **Preserved** — mapped to `GenericEvent` with `extra="allow"`, full payload intact |
| `output_text.delta`, `function_call_arguments.delta`, `refusal.delta`, `content_part.added`, `output_item.added/done`, annotations | **Preserved exactly** |
| Populated `output[]` on `response.completed` (reasoning + message + function_call) | **Preserved** — all 3 items with fields intact |
| `usage` incl. `input_tokens_details.cached_tokens` / `output_tokens_details.reasoning_tokens` | **Preserved** |
| `response` object on `created`/`in_progress`/`completed` | **MODIFIED — id is rewritten** (see below) |
| Terminal `response.failed` | **Raises** `MidStreamFallbackError` wrapping `InternalServerError` instead of yielding the failed event |

### 1. Response id wrapping (must handle, not a blocker)

litellm base64-wraps the upstream `response.id` in every lifecycle event:

```
sent: resp_toy123
got:  resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfdG95MTIz
      decodes to: litellm:custom_llm_provider:openai;model_id:None;response_id:resp_toy123
```

This is deliberate (multi-deployment routing affinity). For us:

- **The admin owns the client-facing `resp_` id.** Do not expose litellm's
  wrapped id. Mint our own `resp_<uuid>` for the client `ResponseRecord`, and
  store litellm's wrapped id in `ResponseRecord.parameters` (or a dedicated
  column) so `previous_response_id` from the client resolves to our DB row and
  we can hand the correct upstream/affinity id back to litellm on the next turn.
- Decode helper exists: `litellm.responses.utils.ResponsesAPIRequestUtils._decode_responses_api_response_id`
  plus `decode_previous_response_id_to_original_previous_response_id` if we ever
  want to unwrap rather than store.

### 2. `response.failed` becomes an exception, not a stream event

litellm intercepts `response.failed` mid-stream and raises
`MidStreamFallbackError` (wrapping `InternalServerError`) rather than yielding
the failed event. The admin emitter must catch litellm exceptions and map them to
the spec `response.failed` / `error` SSE frames itself. This is actually fine for
our design (admin owns the emitter and the error taxonomy) but means we cannot
blindly forward litellm events as the terminal frame.

### 3. Pydantic serializer warnings

`model_dump()` emits `PydanticSerializationUnexpectedValue` warnings for
`content_part.done` `part` unions and `usage` when the upstream payload is a
plain dict. Harmless (values still correct) but noisy; suppress or log at debug.

## Decision (Phase 0 gate)

**litellm is suitable for `/v1/responses` streaming.** Native streaming preserves
every event we need, including non-canonical dual-name reasoning events,
`sequence_number`, populated `output[]`, and full usage details.

Required design constraints carried into Phase 6:

1. **Admin owns the emitter and the client-facing `resp_` id.** Consume litellm
   events → re-frame with our own `SSEmitter` (monotonic sequence numbers,
   `event:` + `data:` framing, `[DONE]` terminator), replacing the upstream
   `response.id` with our minted id on lifecycle frames.
2. **Register every provider alias with litellm** (`supports_native_streaming:
   True`, `mode: "responses"`, `litellm_provider: "openai"`) as part of the
   provider-definition create/update/sync flow. Without this, streaming breaks
   with a confusing `APIError`.
3. **Map litellm exceptions to spec terminal frames.** `response.failed` arrives
   as `MidStreamFallbackError`; boot failures, provider 404s, timeouts likewise.
   Admin's error taxonomy must convert these into `response.failed` / `error` SSE
   frames rather than relying on a yielded event.
4. **Store litellm's wrapped response id** for turn affinity/continuation, and
   decide explicitly whether we pass the raw or wrapped id back to litellm on
   `previous_response_id` turns (recommend: keep our own chain in DB and pass
   full reconstructed `input` to litellm, ignoring litellm-side session state).
5. **Provider-side translation layer is still required** (unchanged from plan):
   the provider normalizes its backend into spec-clean OpenAI/OpenResponses on
   the instance port so admin's litellm call sees a clean `openai`-compatible
   provider. litellm does not fix upstream non-compliance; it faithfully
   transports whatever the provider emits.

No fallback to "chat-only litellm" is needed.

## Reproducing

```bash
cd spike/litellm-fidelity
uv run python spike2.py   # full 26-event fidelity diff
uv run python spike5.py   # populated output[] check
uv run python spike6.py   # response.failed path check
```
