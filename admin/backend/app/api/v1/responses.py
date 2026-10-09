"""Public OpenResponses endpoint: ``POST /v1/responses`` (Phase 6).

Drives litellm against a scheduler-admitted provider instance, re-frames
the SSE stream as the OpenResponses spec with the admin-owned
``resp_<uuid>`` id (see ``spike/litellm-fidelity/FINDINGS.md``), and
persists the turn as a ``ResponseRecord`` + ``TokenUsageSample``.

Flow:

1. Parse the body: ``model`` (alias) and ``input`` required; optional
   ``previous_response_id``, ``stream`` (default false, per spec),
   ``store`` (default true), and passthrough params (instructions,
   tools, ...).
2. Resolve the enabled ``ProviderDefinition`` by alias -> else 404.
3. Mint ``client_response_id = "resp_" + uuid4().hex`` (the admin owns
   the client-facing id; litellm's wrapped id is stored in parameters).
4. Build the litellm ``input``: on ``previous_response_id``, load our
   prior ``ResponseRecord`` and prepend its ``input_items`` +
   ``output_items`` to the new input. We own the chain; litellm-side
   session state is never relied on and ``previous_response_id`` is NOT
   passed to litellm.
5. ``ensure_registered(alias)`` so litellm selects native streaming.
6. ``scheduler.acquire(alias, request_id)`` -> ``Admission`` with the
   instance base URL. **Error contract differs by transport (Phase 6):**

   * Non-stream: ``NoProviderAvailable`` -> 503, ``QueueCleared`` -> 503
     (operator cleared the queue), ``QueueTimeout`` -> 504
     (JSON, since the stream never starts).
   * Stream: the response starts immediately with SSE keepalive comments
     while ``acquire`` blocks (cold boot + FIFO wait can take minutes and
     would otherwise trip proxy idle timeouts). The scheduler errors
     arrive *inside* the stream and are synthesized as the spec
     ``response.failed`` + ``error`` frames (FINDINGS.md §2: the admin
     owns terminal frames) plus a failed ``ResponseRecord`` — no HTTP
     status is possible once the stream has begun.

   The fast pre-stream checks (unknown/disabled alias -> 404, missing
   ``model``/``input`` -> 400) stay HTTP for both transports.
7. Streaming: iterate litellm events through the
    ``SSEEmitter``; on litellm exceptions (``MidStreamFallbackError`` and
    friends) synthesize the spec ``response.failed`` frame; in a
    cancellation-safe ``finally`` close the upstream stream, release the
    scheduler slot, and persist once.
8. Non-stream: accumulate the terminal response, replace its id, persist,
    release, return JSON.

Auth model: trusted LAN — this public route is unauthenticated (see
``app/core/config.py``).
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session, select

from app.core.db import engine
from app.models import (
    ProviderDefinition,
    ResponseRecord,
    TokenUsageSample,
)
from app.services.alias_registry import ensure_registered
from app.services.scheduler import (
    Admission,
    InferenceScheduler,
    NoProviderAvailable,
    QueueCleared,
    QueueTimeout,
    SchedulerError,
)
from app.services.sse import SSEEmitter, to_dict

logger = logging.getLogger("admin.v1.responses")

router = APIRouter(tags=["responses"])

# SSE keepalive cadence during the cold-boot / queue-wait phase. A comment
# line (``: keep-alive``) is ignored by spec-compliant SSE clients but
# resets proxy (Traefik) idle timers so a minutes-long ``scheduler.acquire``
# (model download + load) does not drop the connection before any event.
KEEPALIVE_INTERVAL_SECONDS = 10.0
KEEPALIVE_COMMENT = ": keep-alive\n\n"


# Request fields forwarded straight to litellm.aresponses when present.
PASSTHROUGH_FIELDS = (
    "instructions",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_output_tokens",
    "reasoning",
    "text",
    "truncation",
    "parallel_tool_calls",
    "metadata",
    "user",
    "store",
    "background",
    "include",
)


def get_scheduler(request: Request) -> InferenceScheduler:
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="scheduler not initialized")
    return scheduler  # type: ignore[no-any-return]


def map_exception_to_error(exc: Exception) -> dict[str, Any]:
    """Map a litellm/provider exception to the spec error object."""
    if isinstance(exc, litellm.exceptions.NotFoundError):
        return {
            "type": "invalid_request_error",
            "code": "model_not_found",
            "message": str(exc),
        }
    # MidStreamFallbackError wraps an InternalServerError produced by a
    # provider-side response.failed; both are upstream failures, as are
    # APIError/connection errors.
    return {
        "type": "server_error",
        "code": "upstream_failed",
        "message": str(exc),
    }


def _as_item_list(value: Any) -> list[dict[str, Any]]:
    """Normalize spec input/output payloads into a list of dicts."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v if isinstance(v, dict) else to_dict(v) for v in value]
    return [{"content": value}]


def _clean_output_items(items: list[Any]) -> list[Any]:
    """Drop spec-optional keys that litellm re-injects as ``None``.

    ``to_dict(exclude_none=False)`` adds litellm model defaults such as
    ``phase: null`` (assistant message items) and ``logprobs: null``
    (output text parts). The spec's Zod schema declares those fields
    optional but *non-nullable*, so a present ``null`` fails validation,
    while an absent one passes. Required-but-nullable fields
    (``completed_at``, ``error``, ``usage``, ...) are left untouched —
    stripping those would break the schema instead.
    """
    # key -> only these are safe to remove when None.
    optional_non_nullable = ("phase", "logprobs")

    def strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: strip(v)
                for k, v in obj.items()
                if not (v is None and k in optional_non_nullable)
            }
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    return [strip(item) for item in items]


def _is_int(value: Any) -> bool:
    """True for real ints only — bool is an int subclass."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _as_int(value: Any, default: int = 0) -> int:
    return value if _is_int(value) else default


# Schema-type gates for the required non-nullable response fields: a value
# (provider- OR request-sourced) that fails its check is replaced by the
# next tier down, so wrong-typed upstream/client values can never reach
# the client (e.g. llama.cpp sending ``tools: {}`` or a client echoing
# ``temperature: "0.7"``).
_FIELD_CHECKS: dict[str, Any] = {
    "tools": lambda v: isinstance(v, list),
    "tool_choice": lambda v: isinstance(v, (str, dict)),
    "truncation": lambda v: v in ("auto", "disabled"),
    "parallel_tool_calls": _is_bool,
    "text": lambda v: isinstance(v, dict),
    "top_p": _is_num,
    "temperature": _is_num,
    "presence_penalty": _is_num,
    "frequency_penalty": _is_num,
    "top_logprobs": _is_int,
    "background": _is_bool,
    "service_tier": lambda v: isinstance(v, str),
}


def _normalize_response(
    data: dict[str, Any],
    *,
    store: bool,
    requested: dict[str, Any] | None = None,
    default_status: str = "completed",
) -> None:
    """Coerce a provider terminal response into the spec ResponseResource.

    Backends (e.g. llama.cpp) omit or null out required fields of the
    terminal response. The admin owns the client-facing contract, so fill
    values in place with precedence provider value -> value the client
    requested (the spec requires the response to echo the parameters
    actually used) -> spec default, with each tier type-gated
    (``_FIELD_CHECKS``). Required nullable fields are made present;
    ``store`` is always the admin's own (request-derived) value;
    ``status``/``model``/``object`` are defaulted when missing or
    wrong-typed (``default_status`` derives from the terminal frame
    type). Mutates ``data`` only — nested provider objects are copied,
    never mutated (litellm may share them with its own accumulated
    state).
    """
    requested = requested or {}
    now = int(datetime.now(UTC).timestamp())
    if not isinstance(data.get("model"), str):
        data["model"] = requested.get("model") or "unknown"
    if not isinstance(data.get("status"), str):
        data["status"] = default_status
    if data.get("object") != "response":
        data["object"] = "response"

    if not _is_int(data.get("created_at")):
        data["created_at"] = now
    if data.get("status") == "completed" and not _is_int(data.get("completed_at")):
        data["completed_at"] = now
    else:
        data.setdefault("completed_at", None)

    spec_defaults: dict[str, Any] = {
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
        "parallel_tool_calls": True,
        "text": {"format": {"type": "text"}},
        "top_p": 1,
        "temperature": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_logprobs": 0,
        "background": False,
        "service_tier": "default",
    }
    for key, fallback in spec_defaults.items():
        check = _FIELD_CHECKS[key]
        value = data.get(key)
        if not check(value):
            value = requested.get(key)
        if not check(value):
            value = fallback
        data[key] = value
    data["store"] = store
    for key in (
        "incomplete_details",
        "error",
        "previous_response_id",
        "instructions",
        "reasoning",
        "max_output_tokens",
        "max_tool_calls",
        "safety_identifier",
        "prompt_cache_key",
    ):
        data.setdefault(key, None)
    data.setdefault("metadata", None)

    usage = data.get("usage")
    if isinstance(usage, dict):
        usage = dict(usage)
        if not _is_int(usage.get("input_tokens")):
            usage["input_tokens"] = 0
        if not _is_int(usage.get("output_tokens")):
            usage["output_tokens"] = 0
        in_details = usage.get("input_tokens_details")
        if not isinstance(in_details, dict):
            in_details = {}
        usage["input_tokens_details"] = {
            **{k: v for k, v in in_details.items() if v is not None},
            "cached_tokens": _as_int(in_details.get("cached_tokens")),
        }
        out_details = usage.get("output_tokens_details")
        if not isinstance(out_details, dict):
            out_details = {}
        usage["output_tokens_details"] = {
            **{k: v for k, v in out_details.items() if v is not None},
            "reasoning_tokens": _as_int(out_details.get("reasoning_tokens")),
        }
        if not _is_int(usage.get("total_tokens")):
            usage["total_tokens"] = _as_int(usage.get("input_tokens")) + _as_int(
                usage.get("output_tokens")
            )
        data["usage"] = usage
    else:
        data["usage"] = None


def _usage_tokens(usage: dict[str, Any]) -> tuple[int, int, int, int]:
    """(input, output, total, cached) from a spec usage dict, tolerating
    legacy prompt_tokens/completion_tokens aliases."""
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(
        usage.get("output_tokens") or usage.get("completion_tokens") or 0
    )
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    details = usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    return input_tokens, output_tokens, total_tokens, cached


def build_litellm_input(body: dict[str, Any], previous: ResponseRecord | None) -> Any:
    """Reconstruct the litellm input for a (possibly continued) turn.

    Always returns a list of input items: a bare string input is wrapped
    as a user message so the stored ``input_items`` and the litellm
    payload share one shape. On continuation, the prior record's
    ``input_items`` (the full conversation up to that turn) plus its
    ``output_items`` are prepended to this turn's new input.
    """
    new_input = body.get("input")
    if isinstance(new_input, str):
        new_input = [{"role": "user", "content": new_input}]
    if previous is None:
        return new_input
    combined: list[dict[str, Any]] = list(previous.input_items)
    combined.extend(previous.output_items)
    if isinstance(new_input, list):
        combined.extend(new_input)
    return combined


def persist_turn(
    *,
    client_response_id: str,
    previous_response_id: str | None,
    input_items: list[dict[str, Any]],
    output_items: list[dict[str, Any]],
    definition_id: uuid.UUID,
    instance_id: uuid.UUID | None,
    parameters: dict[str, Any],
    status: str,
    usage: dict[str, Any] | None,
    error: dict[str, Any] | None,
    store: bool,
) -> None:
    """Insert the ResponseRecord (+ TokenUsageSample) once, at terminal."""
    input_tokens = output_tokens = total_tokens = cached = 0
    prompt_ms = predicted_ms = 0.0
    prompt_tps = predicted_tps = 0.0
    if usage:
        input_tokens, output_tokens, total_tokens, cached = _usage_tokens(usage)
        timing = usage.get("completion_tokens_details") or {}
        prompt_ms = float(timing.get("prompt_time") or 0.0) * 1000.0
        predicted_ms = float(timing.get("prediction_time") or 0.0) * 1000.0
        prompt_tps = float(timing.get("prompt_per_second") or 0.0)
        predicted_tps = float(timing.get("predicted_per_second") or 0.0)

    record = ResponseRecord(
        response_id=client_response_id,
        previous_response_id=previous_response_id,
        input_items=input_items,
        output_items=output_items,
        provider_definition_id=definition_id,
        provider_instance_id=instance_id,
        parameters=parameters,
        status=status,
        store=store,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        completed_at=datetime.now(UTC),
    )
    if error:
        record.error_code = str(error.get("code"))
        record.error_message = error.get("message")

    sample = TokenUsageSample(
        provider_instance_id=instance_id,
        provider_definition_id=definition_id,
        prompt_tokens=input_tokens,
        cached_tokens=cached,
        completion_tokens=output_tokens,
        prompt_ms=prompt_ms,
        predicted_ms=predicted_ms,
        prompt_per_second=prompt_tps,
        predicted_per_second=predicted_tps,
    )
    with Session(engine) as session:
        session.add(record)
        session.add(sample)
        session.commit()


def persist_turn_logged(**kwargs: Any) -> None:
    """``persist_turn`` for the response paths, where a DB failure must
    never break the terminal handling — but silently swallowing it loses
    the turn record unnoticed. Log loudly instead."""
    try:
        persist_turn(**kwargs)
    except Exception:
        logger.exception(
            "failed to persist response turn %s",
            kwargs.get("client_response_id"),
        )


@router.post("/v1/responses")
async def create_response(request: Request) -> Any:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="request body must be a JSON object"
        )

    alias = body.get("model")
    input_value = body.get("input")
    if not alias or input_value is None:
        raise HTTPException(status_code=400, detail="'model' and 'input' are required")

    previous_response_id = body.get("previous_response_id")
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
        if definition.modality != "llm":
            raise HTTPException(
                status_code=404,
                detail=f"model alias '{alias}' is not a chat model",
            )
        definition_id = definition.id
        base_params: dict[str, Any] = {
            "model": alias,
            "provider_type": definition.provider_type,
        }

        previous: ResponseRecord | None = None
        if previous_response_id:
            previous = session.exec(
                select(ResponseRecord).where(
                    ResponseRecord.response_id == previous_response_id
                )
            ).first()
            if previous is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"previous_response_id '{previous_response_id}' not found",
                )

    stream = bool(body.get("stream", False))
    store = bool(body.get("store", True))
    client_response_id = f"resp_{uuid.uuid4().hex}"
    request_id = client_response_id

    ensure_registered(alias)
    litellm_input = build_litellm_input(body, previous)

    scheduler = get_scheduler(request)
    passthrough = {k: body[k] for k in PASSTHROUGH_FIELDS if k in body}
    # Record the accumulated input we hand to litellm as this turn's
    # input_items. Because build_litellm_input prepends the prior record's
    # input_items + output_items, each record stores the FULL conversation
    # up to that turn, so a continuation reconstructs in O(1) from just
    # its immediate predecessor (no need to walk the whole chain).
    input_items_for_record = _as_item_list(litellm_input)

    if stream:
        # Keep the *cheap, synchronous* zero-candidate check as HTTP: a
        # stream that can never carry a response should be a 503, not a
        # keepalive-only SSE. Everything slow (boot + FIFO wait) happens
        # inside the stream, where scheduler errors surface as spec
        # response.failed frames instead (see module docstring).
        if not scheduler.has_candidates(alias):
            raise HTTPException(
                status_code=503,
                detail=f"no connected provider instance for alias '{alias}'",
            )
        return StreamingResponse(
            _stream_response(
                scheduler=scheduler,
                alias=alias,
                request_id=request_id,
                client_response_id=client_response_id,
                previous_response_id=previous_response_id,
                input_items_for_record=input_items_for_record,
                definition_id=definition_id,
                litellm_input=litellm_input,
                passthrough=passthrough,
                base_params=base_params,
                store=store,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        admission = await scheduler.acquire(alias, request_id)
    except NoProviderAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueCleared as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    return await _non_stream_response(
        scheduler=scheduler,
        alias=alias,
        request_id=request_id,
        client_response_id=client_response_id,
        previous_response_id=previous_response_id,
        input_items_for_record=input_items_for_record,
        definition_id=definition_id,
        admission=admission,
        litellm_input=litellm_input,
        passthrough=passthrough,
        base_params=base_params,
        store=store,
    )


async def _stream_response(
    *,
    scheduler: InferenceScheduler,
    alias: str,
    request_id: str,
    client_response_id: str,
    previous_response_id: str | None,
    input_items_for_record: list[dict[str, Any]],
    definition_id: uuid.UUID,
    litellm_input: Any,
    passthrough: dict[str, Any],
    base_params: dict[str, Any],
    store: bool,
) -> AsyncIterator[str]:
    emitter = SSEEmitter(client_response_id)
    terminal_output: list[dict[str, Any]] = []
    terminal_usage: dict[str, Any] | None = None
    litellm_wrapped_id: str | None = None
    failed_error: dict[str, Any] | None = None
    terminal_status: str | None = None
    admission: Admission | None = None

    # Cold-boot keepalive: scheduler.acquire can block for minutes
    # (download + load + FIFO wait). Run it as a task and emit SSE comment
    # lines every KEEPALIVE_INTERVAL_SECONDS until it settles so proxies
    # (Traefik) keep the connection open. ``shield`` keeps the acquire
    # alive across the wait_for timeouts; the loop cancels it explicitly
    # if the client disconnects mid-admission.
    acquire_task = asyncio.create_task(scheduler.acquire(alias, request_id))
    stream = None
    try:
        while True:
            try:
                admission = await asyncio.wait_for(
                    asyncio.shield(acquire_task), KEEPALIVE_INTERVAL_SECONDS
                )
                break
            except TimeoutError:
                yield KEEPALIVE_COMMENT

        try:
            # The awaited call itself can raise (connection refused, bad
            # request, litellm APIError before the first event): keep it
            # inside the try so a call-time failure emits the same
            # terminal response.failed + error + [DONE] frames as a
            # mid-stream failure instead of truncating the SSE stream.
            stream = await litellm.aresponses(
                model=alias,
                custom_llm_provider="openai",
                api_base=f"{admission.base_url}/v1",
                stream=True,
                input=litellm_input,
                **passthrough,
            )
            async for event in stream:
                # Shallow copy: to_dict passes plain dicts through, and we
                # may rewrite keys (id capture, terminal normalization) —
                # never mutate litellm's own event objects.
                data = dict(to_dict(event))
                response = data.get("response")
                if isinstance(response, dict):
                    # Capture litellm's affinity-wrapped id before the
                    # emitter replaces it with ours.
                    if litellm_wrapped_id is None:
                        litellm_wrapped_id = response.get("id")
                    if data.get("type") in (
                        "response.completed",
                        "response.incomplete",
                    ):
                        # Work on a copy: litellm may hand us its own
                        # accumulated response dict (to_dict passes plain
                        # dicts through) — never mutate upstream state.
                        response = dict(response)
                        data["response"] = response
                        terminal_output = _clean_output_items(
                            _as_item_list(response.get("output"))
                        )
                        # Emit the cleaned output too: the client validates
                        # the terminal frame against the spec schema.
                        response["output"] = terminal_output
                        _normalize_response(
                            response,
                            store=store,
                            requested={**passthrough, "model": alias},
                            default_status=(
                                "completed"
                                if data["type"] == "response.completed"
                                else "incomplete"
                            ),
                        )
                        terminal_usage = (
                            to_dict(response["usage"])
                            if response.get("usage")
                            else None
                        )
                        terminal_status = (
                            "completed"
                            if data["type"] == "response.completed"
                            else "incomplete"
                        )
                yield emitter.frame(data)
            yield emitter.done()
        except Exception as exc:  # noqa: BLE001 - MidStreamFallbackError et al.
            failed_error = map_exception_to_error(exc)
            logger.warning("litellm stream failed for %s: %s", client_response_id, exc)
            for frame in emitter.failed(failed_error):
                yield frame
            yield emitter.done()
    except SchedulerError as exc:
        # Admission failed after the SSE response started: NoProvider-
        # Available / QueueTimeout / QueueCleared surface as the spec terminal
        # failed frames (the admin synthesizes terminal frames — FINDINGS §2);
        # an HTTP status is no longer possible.
        failed_error = {
            "type": "server_error",
            "code": (
                "queue_timeout"
                if isinstance(exc, QueueTimeout)
                else "queue_cleared"
                if isinstance(exc, QueueCleared)
                else "no_provider"
            ),
            "message": str(exc),
        }
        logger.warning("admission failed in-stream for %s: %s", client_response_id, exc)
        for frame in emitter.failed(failed_error):
            yield frame
        yield emitter.done()
    finally:
        # Single shielded cleanup so a cancelled aclose can never skip the
        # acquire unwind, slot release, or persistence on the client-
        # disconnect path (including during the keepalive phase, before
        # any Admission object exists).
        async def _cleanup() -> None:
            # If the client vanished mid-admission, cancel the acquire and
            # let its own shielded waiter-drop finish before we release —
            # so a slot recorded moments before cancellation is never
            # orphaned.
            if not acquire_task.done():
                acquire_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await acquire_task
            # Close the upstream stream on every exit path (normal end,
            # error, or client disconnect) so the provider releases its
            # own slot.
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.aclose()
            with contextlib.suppress(Exception):
                await scheduler.release(alias, request_id)

        with contextlib.suppress(Exception):
            await asyncio.shield(_cleanup())
        # Persist once at terminal. Synchronous DB work: runs to completion
        # without needing to be awaited, so cancellation cannot interrupt it.
        persist_turn_logged(
            client_response_id=client_response_id,
            previous_response_id=previous_response_id,
            input_items=input_items_for_record,
            output_items=terminal_output,
            definition_id=definition_id,
            instance_id=uuid.UUID(admission.instance_id) if admission else None,
            parameters={
                **base_params,
                "litellm_response_id": litellm_wrapped_id,
                "api_base": admission.base_url if admission else None,
                "stream": True,
            },
            status=terminal_status or "failed",
            usage=terminal_usage if terminal_status else None,
            error=failed_error if terminal_status is None else None,
            store=store,
        )


async def _non_stream_response(
    *,
    scheduler: InferenceScheduler,
    alias: str,
    request_id: str,
    client_response_id: str,
    previous_response_id: str | None,
    input_items_for_record: list[dict[str, Any]],
    definition_id: uuid.UUID,
    admission: Admission,
    litellm_input: Any,
    passthrough: dict[str, Any],
    base_params: dict[str, Any],
    store: bool,
) -> JSONResponse:
    try:
        try:
            result = await litellm.aresponses(
                model=alias,
                custom_llm_provider="openai",
                api_base=f"{admission.base_url}/v1",
                stream=False,
                input=litellm_input,
                **passthrough,
            )
        except Exception as exc:  # noqa: BLE001
            error = map_exception_to_error(exc)
            persist_turn_logged(
                client_response_id=client_response_id,
                previous_response_id=previous_response_id,
                input_items=input_items_for_record,
                output_items=[],
                definition_id=definition_id,
                instance_id=uuid.UUID(admission.instance_id),
                parameters={
                    **base_params,
                    "api_base": admission.base_url,
                    "stream": False,
                },
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

        # Copy: to_dict passes plain dicts through — don't mutate litellm's
        # accumulated response object.
        data = dict(to_dict(result))
        litellm_wrapped_id = data.get("id")
        # The admin owns the client-facing id.
        data["id"] = client_response_id
        output_items = _clean_output_items(_as_item_list(data.get("output")))
        data["output"] = output_items
        _normalize_response(
            data, store=store, requested={**passthrough, "model": alias}
        )
        usage = data.get("usage")

        persist_turn_logged(
            client_response_id=client_response_id,
            previous_response_id=previous_response_id,
            input_items=input_items_for_record,
            output_items=output_items,
            definition_id=definition_id,
            instance_id=uuid.UUID(admission.instance_id),
            parameters={
                **base_params,
                "litellm_response_id": litellm_wrapped_id,
                "api_base": admission.base_url,
                "stream": False,
            },
            status=data.get("status", "completed"),
            usage=to_dict(usage) if usage else None,
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
