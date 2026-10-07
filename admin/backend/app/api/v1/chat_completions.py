"""Public OpenAI Chat Completions endpoint: ``POST /v1/chat/completions`` (Phase 7).

Mirrors the Phase 6 ``/v1/responses`` route (``app/api/v1/responses.py``)
against ``litellm.acompletion`` instead of ``aresponses``: same scheduler
admission, same provider port, same cancellation-safe release + persist
discipline. Chat-spec differences:

- The admin owns the client-facing id here too: every chunk's ``id`` is
  replaced with a minted ``chatcmpl-<uuid>`` (litellm passes the raw
  upstream chat id through for chat — unlike the base64-wrapped responses
  id — but we never surface it; it is captured for persistence).
- Chat SSE is data-only — no ``event:`` lines — so each chunk is framed as
  ``data: {chunk}\\n\\n`` and the stream ends with ``data: [DONE]\\n\\n``.
- Failures raised by the awaited ``litellm.acompletion`` call (before any
  SSE byte) are a real HTTP 502 JSON, like the non-stream path. Once the
  stream has started, mid-stream failures use chat-style error framing:
  ``data: {"error": {...}}\\n\\n`` + ``[DONE]`` (not ``response.failed``).
- Persistence reuses the Phase 6 ``persist_turn`` path with
  ``parameters.api_format = "chat_completions"``: the request
  ``messages`` are stored as ``input_items`` and the aggregated assistant
  message(s) as ``output_items``.
- ``stream`` defaults to **false** per the OpenAI spec.
- The admin always requests ``stream_options.include_usage`` on the
  streaming path so the upstream reports a final usage chunk and we can
  persist token counts.

Native chat streaming requires the alias registered with ``mode: "chat"``
— see ``app/services/alias_registry.py`` for the litellm source evidence
(a ``mode: "responses"`` entry makes ``acompletion`` bridge the request
to ``/v1/responses`` instead of the chat endpoint).

Auth model: trusted LAN — this public route is unauthenticated.
"""

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session, select

from app.api.v1.responses import (
    _as_item_list,
    get_scheduler,
    map_exception_to_error,
    persist_turn_logged,
)
from app.core.db import engine
from app.models import ProviderDefinition
from app.services.alias_registry import ensure_registered
from app.services.scheduler import (
    Admission,
    InferenceScheduler,
    NoProviderAvailable,
    QueueTimeout,
)
from app.services.sse import to_dict

logger = logging.getLogger("admin.v1.chat_completions")

router = APIRouter(tags=["chat"])

# Request fields forwarded straight to litellm.acompletion when present.
CHAT_PASSTHROUGH_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "tools",
    "tool_choice",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "n",
    "user",
    "response_format",
    "logprobs",
    "top_logprobs",
)


class _ChatChunkAggregator:
    """Fold chat.completion.chunk deltas into assistant message(s).

    Tracks per-choice content, tool-call deltas (merged by tool-call
    index), and whether any chunk carried a ``finish_reason`` (the chat
    terminal signal used for persistence status).
    """

    def __init__(self) -> None:
        self._choices: dict[int, dict[str, Any]] = {}
        self.finished = False

    def consume(self, chunk: dict[str, Any]) -> None:
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = int(choice.get("index", 0))
            slot = self._choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )
            message: dict[str, Any] = slot["message"]
            delta = choice.get("delta")
            if isinstance(delta, dict):
                if delta.get("role"):
                    message["role"] = delta["role"]
                content = delta.get("content")
                if content:
                    message["content"] = (message.get("content") or "") + str(content)
                for tool_delta in delta.get("tool_calls") or []:
                    self._merge_tool_call(message, tool_delta)
            finish = choice.get("finish_reason")
            if finish:
                slot["finish_reason"] = finish
                self.finished = True

    @staticmethod
    def _merge_tool_call(message: dict[str, Any], tool_delta: Any) -> None:
        if not isinstance(tool_delta, dict):
            return
        calls: list[dict[str, Any]] = message.setdefault("tool_calls", [])
        index = int(tool_delta.get("index", len(calls) - 1 if calls else 0))
        while len(calls) <= index:
            calls.append(
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )
        call = calls[index]
        if tool_delta.get("id"):
            call["id"] = tool_delta["id"]
        if tool_delta.get("type"):
            call["type"] = tool_delta["type"]
        fn_delta = tool_delta.get("function") or {}
        if fn_delta.get("name"):
            call["function"]["name"] = (call["function"].get("name") or "") + fn_delta[
                "name"
            ]
        if fn_delta.get("arguments"):
            call["function"]["arguments"] = (
                call["function"].get("arguments") or ""
            ) + str(fn_delta["arguments"])

    def output_items(self) -> list[dict[str, Any]]:
        return [self._choices[i]["message"] for i in sorted(self._choices)]


def _normalize_chat_usage(usage: Any) -> dict[str, Any] | None:
    """Map chat-spec usage onto the responses-shaped keys ``persist_turn``
    reads (``input_tokens_details`` for cached tokens), keeping the
    prompt_tokens/completion_tokens aliases it already tolerates."""
    if not usage:
        return None
    u = to_dict(usage)
    if "input_tokens_details" not in u and u.get("prompt_tokens_details"):
        u["input_tokens_details"] = u["prompt_tokens_details"]
    return u


def _chat_error_body(error: dict[str, Any]) -> str:
    return f"data: {json.dumps({'error': error})}\n\n"


@router.post("/v1/chat/completions")
async def create_chat_completion(request: Request) -> Any:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="request body must be a JSON object"
        )

    alias = body.get("model")
    messages = body.get("messages")
    if not alias:
        raise HTTPException(status_code=400, detail="'model' is required")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(
            status_code=400, detail="'messages' must be a non-empty list"
        )

    with Session(engine) as session:
        definition = session.exec(
            select(ProviderDefinition).where(ProviderDefinition.alias == alias)
        ).first()
        if definition is None:
            raise HTTPException(
                status_code=404, detail=f"unknown model alias '{alias}'"
            )
        if not definition.enabled:
            raise HTTPException(
                status_code=404, detail=f"model alias '{alias}' is disabled"
            )
        definition_id = definition.id
        base_params: dict[str, Any] = {
            "model": alias,
            "provider_type": definition.provider_type,
            "api_format": "chat_completions",
        }

    stream = bool(body.get("stream", False))
    store = bool(body.get("store", True))
    client_completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    request_id = client_completion_id

    ensure_registered(alias)

    scheduler = get_scheduler(request)
    try:
        admission = await scheduler.acquire(alias, request_id)
    except NoProviderAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    passthrough = {k: body[k] for k in CHAT_PASSTHROUGH_FIELDS if k in body}
    # Chat SSE has no usage by default; force include_usage so the stream
    # carries a final usage chunk we can persist (merged with any
    # client-provided stream_options).
    client_stream_options = body.get("stream_options")
    stream_options = (
        dict(client_stream_options) if isinstance(client_stream_options, dict) else {}
    )
    stream_options["include_usage"] = True
    input_items_for_record = _as_item_list(messages)

    if stream:
        # Await the call itself BEFORE committing the SSE response:
        # a call-time raise (upstream 4xx/5xx after litellm's retries,
        # connection refused) maps to a real HTTP 502 like the non-stream
        # path. Once headers are committed the only honest signal left is
        # an aborted stream — which browsers report as a bare network
        # error. Failures raised while iterating (after the first await
        # of the committed stream) keep the in-stream error framing.
        try:
            upstream = await litellm.acompletion(
                model=alias,
                custom_llm_provider="openai",
                # The openai chat path demands a non-empty key even for
                # unauthenticated local upstreams; the provider ignores it.
                api_key="unused",
                api_base=f"{admission.base_url}/v1",
                stream=True,
                messages=messages,
                # Forced include_usage (computed above) so token counts
                # persist on the streaming path.
                stream_options=stream_options,
                # Belt-and-suspenders: never let litellm's gpt-5 conditional
                # bridge reroute this chat call to /v1/responses.
                _skip_responses_api_bridge=True,
                **passthrough,
            )
        except asyncio.CancelledError:
            # Client vanished during the await: free the slot (idempotent)
            # before unwinding, shielded so the cancel cannot skip it.
            with contextlib.suppress(Exception):
                await asyncio.shield(scheduler.release(alias, request_id))
            raise
        except Exception as exc:  # noqa: BLE001 - call-time litellm failure
            error = map_exception_to_error(exc)
            logger.warning(
                "litellm chat call failed for %s before streaming: %s",
                client_completion_id,
                exc,
            )
            # Shielded like the CancelledError branch above: a cancel
            # landing here must not skip the release (a leaked active
            # slot would also block the reaper's zero-slots guard).
            with contextlib.suppress(Exception):
                await asyncio.shield(scheduler.release(alias, request_id))
            persist_turn_logged(
                client_response_id=client_completion_id,
                previous_response_id=None,
                input_items=input_items_for_record,
                output_items=[],
                definition_id=definition_id,
                instance_id=uuid.UUID(admission.instance_id),
                parameters=_record_parameters(
                    base_params,
                    admission,
                    stream=True,
                    litellm_id=None,
                    passthrough=passthrough,
                ),
                status="failed",
                usage=None,
                error=error,
                store=store,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": error["type"],
                        "code": error["code"],
                        "message": error["message"],
                    }
                },
            )
        return StreamingResponse(
            _stream_chat(
                scheduler=scheduler,
                alias=alias,
                request_id=request_id,
                client_completion_id=client_completion_id,
                input_items_for_record=input_items_for_record,
                definition_id=definition_id,
                admission=admission,
                upstream=upstream,
                passthrough=passthrough,
                base_params=base_params,
                store=store,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return await _non_stream_chat(
        scheduler=scheduler,
        alias=alias,
        request_id=request_id,
        client_completion_id=client_completion_id,
        input_items_for_record=input_items_for_record,
        definition_id=definition_id,
        admission=admission,
        messages=messages,
        passthrough=passthrough,
        base_params=base_params,
        store=store,
    )


def _record_parameters(
    base_params: dict[str, Any],
    admission: Admission,
    *,
    stream: bool,
    litellm_id: str | None,
    passthrough: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base_params,
        "litellm_id": litellm_id,
        "api_base": admission.base_url,
        "stream": stream,
        "request_parameters": passthrough,
    }


async def _stream_chat(
    *,
    scheduler: InferenceScheduler,
    alias: str,
    request_id: str,
    client_completion_id: str,
    input_items_for_record: list[dict[str, Any]],
    definition_id: uuid.UUID,
    admission: Admission,
    upstream: Any,
    passthrough: dict[str, Any],
    base_params: dict[str, Any],
    store: bool = True,
) -> AsyncIterator[str]:
    """Relay an already-created litellm chat stream as SSE.

    The awaited ``litellm.acompletion`` call happens in the route (a
    call-time failure is a real HTTP 502 there); by the time this runs
    the response headers are committed, so every failure while
    iterating must surface as the chat-style error frame + [DONE],
    never as an aborted chunked body.
    """
    aggregator = _ChatChunkAggregator()
    terminal_usage: dict[str, Any] | None = None
    litellm_id: str | None = None
    failed_error: dict[str, Any] | None = None

    try:
        try:
            async for chunk in upstream:
                data = to_dict(chunk)
                if litellm_id is None:
                    litellm_id = data.get("id")
                aggregator.consume(data)
                if data.get("usage"):
                    terminal_usage = _normalize_chat_usage(data["usage"])
                # The admin owns the client-facing id AND model name on
                # every chunk (provider may echo the raw backend path).
                data["id"] = client_completion_id
                data["model"] = alias
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001 - mid-stream litellm failures
            failed_error = map_exception_to_error(exc)
            logger.warning(
                "litellm chat stream failed for %s: %s", client_completion_id, exc
            )
            yield _chat_error_body(failed_error)
            yield "data: [DONE]\n\n"
    finally:
        # Single shielded cleanup so a cancelled aclose can never skip the
        # slot release or persistence on the client-disconnect path
        # (same pattern as Phase 6 _stream_response).
        async def _cleanup() -> None:
            if upstream is not None:
                with contextlib.suppress(Exception):
                    await upstream.aclose()
            with contextlib.suppress(Exception):
                await scheduler.release(alias, request_id)

        with contextlib.suppress(Exception):
            await asyncio.shield(_cleanup())
        completed = aggregator.finished and failed_error is None
        if not completed and failed_error is None:
            failed_error = {
                "type": "server_error",
                "code": "client_disconnected",
                "message": "client disconnected before the completion finished",
            }
        persist_turn_logged(
            client_response_id=client_completion_id,
            previous_response_id=None,
            input_items=input_items_for_record,
            output_items=aggregator.output_items() if completed else [],
            definition_id=definition_id,
            instance_id=uuid.UUID(admission.instance_id),
            parameters=_record_parameters(
                base_params,
                admission,
                stream=True,
                litellm_id=litellm_id,
                passthrough=passthrough,
            ),
            status="completed" if completed else "failed",
            usage=terminal_usage if completed else None,
            error=None if completed else failed_error,
            store=store,
        )


async def _non_stream_chat(
    *,
    scheduler: InferenceScheduler,
    alias: str,
    request_id: str,
    client_completion_id: str,
    input_items_for_record: list[dict[str, Any]],
    definition_id: uuid.UUID,
    admission: Admission,
    messages: list[dict[str, Any]],
    passthrough: dict[str, Any],
    base_params: dict[str, Any],
    store: bool = True,
) -> JSONResponse:
    try:
        try:
            result = await litellm.acompletion(
                model=alias,
                custom_llm_provider="openai",
                api_key="unused",
                api_base=f"{admission.base_url}/v1",
                stream=False,
                messages=messages,
                _skip_responses_api_bridge=True,
                **passthrough,
            )
        except Exception as exc:  # noqa: BLE001
            error = map_exception_to_error(exc)
            persist_turn_logged(
                client_response_id=client_completion_id,
                previous_response_id=None,
                input_items=input_items_for_record,
                output_items=[],
                definition_id=definition_id,
                instance_id=uuid.UUID(admission.instance_id),
                parameters=_record_parameters(
                    base_params,
                    admission,
                    stream=False,
                    litellm_id=None,
                    passthrough=passthrough,
                ),
                status="failed",
                usage=None,
                error=error,
                store=store,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": error["type"],
                        "code": error["code"],
                        "message": error["message"],
                    }
                },
            )

        data = to_dict(result)
        litellm_id = data.get("id")
        # The admin owns the client-facing id AND the model name: the
        # provider echoes its raw backend path (e.g. a GGUF path) as the
        # model, which must not leak to clients using a different alias.
        data["id"] = client_completion_id
        data["model"] = alias
        usage = _normalize_chat_usage(data.get("usage"))
        output_items: list[dict[str, Any]] = []
        for choice in data.get("choices") or []:
            if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                output_items.append(choice["message"])

        persist_turn_logged(
            client_response_id=client_completion_id,
            previous_response_id=None,
            input_items=input_items_for_record,
            output_items=output_items,
            definition_id=definition_id,
            instance_id=uuid.UUID(admission.instance_id),
            parameters=_record_parameters(
                base_params,
                admission,
                stream=False,
                litellm_id=litellm_id,
                passthrough=passthrough,
            ),
            status="completed",
            usage=usage,
            error=None,
            store=store,
        )
        return JSONResponse(content=data)
    finally:
        # Guaranteed slot release on EVERY exit path — including
        # asyncio.CancelledError (client disconnect during the await), which
        # `except Exception` does not catch. Release is idempotent.
        with contextlib.suppress(Exception):
            await asyncio.shield(scheduler.release(alias, request_id))
