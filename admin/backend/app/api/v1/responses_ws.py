"""Responses-over-WebSocket transport: ``WS /v1/responses`` (Phase 21 S2).

Implements the OpenResponses §WebSocket Transport. The admin terminates the
client socket; each turn runs the *identical* transport-agnostic pipeline the
SSE route uses (:func:`app.api.v1.responses.stream_turn_events`) against a
scheduler-admitted provider instance and re-emits the normalized event dicts
as WebSocket JSON frames. Providers are unchanged (locked decision 0).

Contract (see IMPLEMENTATION_STATUS.md Phase 21 locked decisions):

* One in-flight turn per connection, processed sequentially (decision 2). A
  ``response.create`` message is read, run to its terminal event, then the
  next message is read; a dedicated reader task owns every ``receive`` so a
  mid-turn client disconnect is observed (cancelling the turn and releasing
  the slot) and any stray mid-turn message is buffered, never swallowed. The
  reader bounds queued inbound frames (inbox + pending) at
  ``WS_INBOX_MAX``; a client that floods a slow turn past that is treated as
  protocol abuse — the connection is closed with a 400 ``invalid_request``
  ("too many queued messages") envelope rather than growing memory unbounded.
* Request body is ``{"type": "response.create", ...create body}``.
  ``stream``/``stream_options``/``background`` are rejected; bad JSON, a
  missing ``model``/``input``, an unknown/disabled/non-llm alias, no
  connected provider, and an unresolvable ``previous_response_id`` all become
  ``error`` envelopes and leave the socket open (decision 2).
* Continuation state is cached per connection in a small bounded LRU
  (``resp_id -> _TurnState``) so ``store=false`` chains work within a socket;
  the Postgres fallback filters ``store=true`` so a store-false id is not
  resurrected across sockets (decision 3). A failed continuation evicts the
  referenced id (decision 3/4).
* WS-only ``function_call_output.call_id`` validation against the previous
  turn's accumulated items (decision 4).
* No ``[DONE]`` terminator ever (decision 6); the terminal event ends a turn.
* Cold boot: the first ``KEEPALIVE`` synthesizes ``response.created`` +
  ``response.in_progress`` (the SSE keepalive comment has no WS equivalent);
  a provider's own duplicates of those two are dropped so the lifecycle is
  emitted exactly once and the sequence stays monotonic (decision 7).
* 60-minute connection limit enforced per-message and by a background timer
  racing the read loop (decision 8).

Auth model: trusted LAN — unauthenticated, like the HTTP ``/v1`` routes.
"""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, WebSocket
from sqlmodel import Session, select

from app.api.v1.responses import (
    KEEPALIVE,
    PASSTHROUGH_FIELDS,
    ChainSource,
    _as_item_list,
    _normalize_response,
    build_litellm_input,
    stream_turn_events,
)
from app.core.db import engine
from app.models import ProviderDefinition, ResponseRecord
from app.services.alias_registry import ensure_registered
from app.services.scheduler import InferenceScheduler
from app.services.sse import SSEEmitter

logger = logging.getLogger("admin.v1.responses_ws")

router = APIRouter(tags=["responses"])

# Decision 8: hard cap on a single client WebSocket connection's lifetime.
WS_CONNECTION_LIMIT_SECONDS = 3600.0

# Decision 3: how many recent turns a single connection keeps in its
# continuation cache. Compliance scenarios only ever continue from the last
# successful turn, and store=true history hydrates from Postgres, so a small
# LRU bounds memory (each entry holds the FULL accumulated input) instead of
# growing O(N^2) over a long-lived connection.
WS_CACHE_MAX_TURNS = 8

# Backpressure bound: the reader stops accepting inbound frames once the sum of
# queued (inbox + pending) messages reaches this, treating a client that floods
# a slow turn with messages as protocol abuse — the connection is closed with a
# 400 ``invalid_request`` envelope rather than growing memory without limit.
WS_INBOX_MAX = 64

# Lifecycle event types that must appear at most once per turn (decision 7).
_LIFECYCLE_ONCE = ("response.created", "response.in_progress")

# Terminal event types that end a turn (decision 6).
_TERMINAL_TYPES = ("response.completed", "response.incomplete", "response.failed")

# Sentinel returned by the message reader when the connection limit is hit.
_LIMIT_HIT = object()

# Sentinel returned by the message reader when the client overflowed the inbox.
_ABUSE = object()


def _now() -> float:
    """Monotonic clock indirection so tests can drive the connection limit."""
    return time.monotonic()


@dataclass
class _TurnState:
    """Per-connection continuation state for one completed turn.

    Structurally satisfies :class:`ChainSource` for
    :func:`build_litellm_input` (which reads ``input_items``/``output_items``)
    and carries the fields needed to re-resolve a continuation without a DB
    round-trip.
    """

    input_items: list[dict[str, Any]]
    output_items: list[dict[str, Any]]
    definition_id: uuid.UUID
    alias: str
    provider_type: str
    store: bool


class _TurnCache:
    """Bounded LRU of ``resp_id -> _TurnState`` for one connection."""

    def __init__(self, maxsize: int = WS_CACHE_MAX_TURNS) -> None:
        self._data: OrderedDict[str, _TurnState] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> _TurnState | None:
        state = self._data.get(key)
        if state is not None:
            self._data.move_to_end(key)
        return state

    def put(self, key: str, state: _TurnState) -> None:
        self._data[key] = state
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def evict(self, key: str) -> None:
        self._data.pop(key, None)


def _error_envelope(
    status: int,
    code: str,
    message: str,
    *,
    param: str | None = None,
    err_type: str | None = None,
) -> dict[str, Any]:
    """Build a WS ``error`` envelope (``webSocketErrorEventSchema`` shape).

    ``status`` is the HTTP-style code; ``error.code`` is what the compliance
    client reads to end a turn with that code.
    """
    error: dict[str, Any] = {"code": code, "message": message, "param": param}
    if err_type is not None:
        error["type"] = err_type
    return {"type": "error", "status": status, "error": error}


def _valid_call_ids(previous: ChainSource) -> set[str]:
    """Collect ``call_id``s from the previous turn's accumulated ``function_call``
    items (input + output), used for the WS-only call_id validation."""
    call_ids: set[str] = set()
    for item in list(previous.input_items) + list(previous.output_items):
        if isinstance(item, dict) and item.get("type") == "function_call":
            cid = item.get("call_id")
            if isinstance(cid, str):
                call_ids.add(cid)
    return call_ids


def _unknown_call_ids(new_input: Any, previous: ChainSource) -> list[str]:
    """Return call_ids on new ``function_call_output`` items that do not match
    a ``function_call`` in ``previous`` (decision 4)."""
    valid = _valid_call_ids(previous)
    unknown: list[str] = []
    for item in _as_item_list(new_input):
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            cid = item.get("call_id")
            if cid not in valid:
                unknown.append(str(cid))
    return unknown


def _resolve_alias(
    session: Session, alias: str
) -> tuple[uuid.UUID, str] | dict[str, Any]:
    """Resolve an enabled llm alias to ``(definition_id, provider_type)`` or
    return an error envelope (mirrors the HTTP 404s as WS envelopes)."""
    definition = session.exec(
        select(ProviderDefinition).where(ProviderDefinition.alias == alias)
    ).first()
    if definition is None:
        return _error_envelope(404, "model_not_found", f"unknown model alias '{alias}'")
    if not definition.enabled:
        return _error_envelope(
            404, "model_not_found", f"model alias '{alias}' is disabled"
        )
    if definition.modality != "llm":
        return _error_envelope(
            404, "model_not_found", f"model alias '{alias}' is not a chat model"
        )
    return definition.id, definition.provider_type


def _resolve_previous(
    session: Session,
    cache: _TurnCache,
    previous_response_id: str,
) -> ChainSource | dict[str, Any]:
    """Resolve a continuation's previous state: connection cache first, then a
    ``store=true`` Postgres record; miss -> ``previous_response_not_found``
    envelope (decision 3)."""
    cached = cache.get(previous_response_id)
    if cached is not None:
        return cached
    record = session.exec(
        select(ResponseRecord).where(
            ResponseRecord.response_id == previous_response_id,
            ResponseRecord.store == True,  # noqa: E712
        )
    ).first()
    if record is not None:
        return record
    return _error_envelope(
        404,
        "previous_response_not_found",
        f"Previous response with id '{previous_response_id}' not found.",
        param="previous_response_id",
    )


async def _send(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    """Send one JSON frame; return False if the socket is already gone."""
    try:
        await websocket.send_json(payload)
        return True
    except Exception:  # noqa: BLE001 - client vanished mid-turn
        return False


async def _run_turn(
    websocket: WebSocket,
    scheduler: InferenceScheduler,
    cache: _TurnCache,
    *,
    body: dict[str, Any],
    alias: str,
    provider_type: str,
    definition_id: uuid.UUID,
    previous: ChainSource | None,
    previous_response_id: str | None,
    store: bool,
) -> None:
    """Execute one turn and forward its event dicts as JSON frames.

    Handles cold-boot lifecycle synthesis + provider-duplicate drop
    (decision 7), terminal capture, and the post-turn cache update / eviction
    (decisions 3/4). Runs to the terminal event; the caller races this against a
    client disconnect and cancels it (which drives ``stream_turn_events``'s
    cancellation-safe ``finally`` to release the slot).
    """
    client_response_id = f"resp_{uuid.uuid4().hex}"
    ensure_registered(alias)
    litellm_input = build_litellm_input(body, previous)
    input_items_for_record = _as_item_list(litellm_input)
    passthrough = {k: body[k] for k in PASSTHROUGH_FIELDS if k in body}
    base_params = {"model": alias, "provider_type": provider_type}
    requested = {**passthrough, "model": alias}

    emitter = SSEEmitter(client_response_id)
    runner = stream_turn_events(
        emitter=emitter,
        scheduler=scheduler,
        alias=alias,
        request_id=client_response_id,
        client_response_id=client_response_id,
        previous_response_id=previous_response_id,
        input_items_for_record=input_items_for_record,
        definition_id=definition_id,
        litellm_input=litellm_input,
        passthrough=passthrough,
        base_params=base_params,
        store=store,
    )

    synthesized = False
    sent_lifecycle: set[str] = set()
    terminal_status: str | None = None
    terminal_output: list[dict[str, Any]] = []

    async def _emit(event: dict[str, Any]) -> bool:
        nonlocal terminal_status, terminal_output
        etype = event.get("type")
        if etype in _LIFECYCLE_ONCE:
            if etype in sent_lifecycle:
                return True  # drop provider's duplicate lifecycle frame
            sent_lifecycle.add(etype)
        if etype in _TERMINAL_TYPES:
            terminal_status = etype.split(".", 1)[1]
            response = event.get("response")
            if isinstance(response, dict):
                terminal_output = _as_item_list(response.get("output"))
        return await _send(websocket, event)

    try:
        async for item in runner:
            if item is KEEPALIVE:
                # Decision 7: synthesize the lifecycle pair exactly once, on
                # the first keepalive; later keepalives are silent (no SSE
                # comment equivalent on WS).
                if synthesized:
                    continue
                synthesized = True
                for etype, status in (
                    ("response.created", "created"),
                    ("response.in_progress", "in_progress"),
                ):
                    skeleton: dict[str, Any] = {"status": status, "output": []}
                    _normalize_response(
                        skeleton,
                        store=store,
                        requested=requested,
                        default_status=status,
                    )
                    if not await _emit(
                        emitter.event_data({"type": etype, "response": skeleton})
                    ):
                        return
                continue
            if not await _emit(item):
                return
    finally:
        # Propagate an early exit (client gone / return above / cancellation)
        # into the runner so its cancellation-safe finally unwinds the acquire
        # + releases the slot (S1 contract). Shielded so a task cancellation
        # cannot skip it; no-op on normal exhaustion.
        with contextlib.suppress(BaseException):
            await asyncio.shield(runner.aclose())

    # Post-turn cache maintenance (decisions 3/4).
    if previous_response_id and terminal_status == "failed":
        cache.evict(previous_response_id)
    if terminal_status in ("completed", "incomplete"):
        cache.put(
            client_response_id,
            _TurnState(
                input_items=input_items_for_record,
                output_items=terminal_output,
                definition_id=definition_id,
                alias=alias,
                provider_type=provider_type,
                store=store,
            ),
        )


async def _prepare_turn(
    websocket: WebSocket,
    scheduler: InferenceScheduler,
    cache: _TurnCache,
    message: dict[str, Any],
) -> dict[str, Any] | None:
    """Parse + validate one inbound frame. On a client-facing error, send the
    error envelope and return ``None`` (socket stays open). On success return
    the kwargs for :func:`_run_turn`."""
    raw = message.get("text")
    if raw is None:
        raw = (message.get("bytes") or b"").decode("utf-8", "replace")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError, UnicodeDecodeError:
        await _send(
            websocket,
            _error_envelope(400, "invalid_request", "Request must be a JSON object."),
        )
        return None
    if not isinstance(body, dict):
        await _send(
            websocket,
            _error_envelope(400, "invalid_request", "Request must be a JSON object."),
        )
        return None
    if body.get("type") != "response.create":
        await _send(
            websocket,
            _error_envelope(
                400,
                "invalid_request",
                "Only 'response.create' events are supported over WebSocket.",
                param="type",
            ),
        )
        return None
    for forbidden in ("stream", "stream_options", "background"):
        if forbidden in body:
            await _send(
                websocket,
                _error_envelope(
                    400,
                    "invalid_request",
                    f"'{forbidden}' is not allowed over WebSocket.",
                    param=forbidden,
                ),
            )
            return None

    alias = body.get("model")
    input_value = body.get("input")
    previous_response_id = body.get("previous_response_id")
    if not alias or (input_value is None and not previous_response_id):
        await _send(
            websocket,
            _error_envelope(
                400,
                "invalid_request",
                "'model' and ('input' or 'previous_response_id') are required.",
            ),
        )
        return None

    store = bool(body.get("store", True))
    with Session(engine) as session:
        resolved = _resolve_alias(session, alias)
        if isinstance(resolved, dict):
            await _send(websocket, resolved)
            return None
        definition_id, provider_type = resolved

        previous: ChainSource | None = None
        if previous_response_id:
            resolved_prev = _resolve_previous(session, cache, previous_response_id)
            if isinstance(resolved_prev, dict):
                await _send(websocket, resolved_prev)
                return None
            previous = resolved_prev
            unknown = _unknown_call_ids(input_value, previous)
            if unknown:
                # Decision 4: bad call_id poisons the referenced id.
                cache.evict(previous_response_id)
                await _send(
                    websocket,
                    _error_envelope(
                        400,
                        "invalid_request",
                        "function_call_output call_id(s) do not match a "
                        f"function_call in the previous response: {unknown}.",
                        param="input",
                    ),
                )
                return None

    # Mirror the SSE path's cheap zero-candidate pre-check (decision 0): a turn
    # that can never be admitted is an error envelope, not a cold-boot stall.
    if not scheduler.has_candidates(alias):
        await _send(
            websocket,
            _error_envelope(
                503,
                "no_provider",
                f"no connected provider instance for alias '{alias}'",
            ),
        )
        return None

    return {
        "body": body,
        "alias": alias,
        "provider_type": provider_type,
        "definition_id": definition_id,
        "previous": previous,
        "previous_response_id": previous_response_id,
        "store": store,
    }


async def _run_turn_raced(
    websocket: WebSocket,
    scheduler: InferenceScheduler,
    cache: _TurnCache,
    params: dict[str, Any],
    inbox: asyncio.Queue,
    pending: deque,
) -> bool:
    """Run one turn as a task while racing it against the message inbox so a
    client disconnect mid-turn cancels it (releasing the slot via the runner's
    shielded ``finally``). Returns ``True`` if the client disconnected (the
    caller must stop), ``False`` if the turn finished. Any stray message that
    arrives mid-turn is pushed onto ``pending`` for the main loop (never
    swallowed)."""
    turn_task = asyncio.create_task(_run_turn(websocket, scheduler, cache, **params))
    try:
        while True:
            get_task = asyncio.create_task(inbox.get())
            done, _ = await asyncio.wait(
                {turn_task, get_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if get_task in done:
                message = get_task.result()
                if message.get("type") == "websocket.disconnect":
                    if turn_task in done:
                        # Requeue unconditionally FIRST so the sole disconnect
                        # is never lost if the turn raised; then surface the
                        # error (the main loop re-reads the disconnect next).
                        await inbox.put(message)
                        turn_task.result()
                        return False
                    # Mid-turn disconnect: cancel the turn (its shielded finally
                    # releases the slot) and drop any stray buffered messages so
                    # they cannot run turns against a dead socket.
                    turn_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await turn_task
                    pending.clear()
                    return True
                # Stray message while the turn is still running: keep it.
                pending.append(message)
                if turn_task in done:
                    turn_task.result()
                    return False
                continue
            # Turn finished first; drop the unused inbox read (no message lost:
            # a cancelled Queue.get never dequeues).
            get_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await get_task
            turn_task.result()
            return False
    finally:
        if not turn_task.done():
            turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn_task


@router.websocket("/v1/responses")
async def responses_websocket(websocket: WebSocket) -> None:
    """Sequential Responses-over-WebSocket session (Phase 21)."""
    await websocket.accept()
    scheduler: InferenceScheduler | None = getattr(
        websocket.app.state, "scheduler", None
    )
    if scheduler is None:
        await _send(
            websocket,
            _error_envelope(503, "server_error", "scheduler not initialized"),
        )
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    cache = _TurnCache()
    start = _now()
    limit_event = asyncio.Event()
    abuse_event = asyncio.Event()
    inbox: asyncio.Queue = asyncio.Queue()
    pending: deque = deque()

    async def _reader() -> None:
        # Single owner of every websocket.receive() so mid-turn disconnects are
        # observed without racing the turn for the socket, and no message is
        # ever lost (everything lands in ``inbox``). Overflow past
        # ``WS_INBOX_MAX`` (inbox + pending) is protocol abuse: stop reading and
        # signal the main loop to close with a 400 envelope.
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    await inbox.put(message)
                    return
                if inbox.qsize() + len(pending) >= WS_INBOX_MAX:
                    abuse_event.set()
                    return
                await inbox.put(message)
        except Exception:  # noqa: BLE001 - socket gone: signal disconnect
            await inbox.put({"type": "websocket.disconnect"})

    async def _limiter() -> None:
        remaining = WS_CONNECTION_LIMIT_SECONDS - (_now() - start)
        if remaining > 0:
            await asyncio.sleep(remaining)
        limit_event.set()

    async def _close_on_limit() -> None:
        await _send(
            websocket,
            _error_envelope(
                400,
                "websocket_connection_limit_reached",
                "WebSocket connection limit of 60 minutes reached.",
            ),
        )
        with contextlib.suppress(Exception):
            await websocket.close()

    async def _read_message() -> Any:
        """Next inbound message, ``_LIMIT_HIT`` if the connection cap fires
        while idle, or ``_ABUSE`` if the client overflowed the inbox."""
        if pending:
            if abuse_event.is_set():
                return _ABUSE
            return pending.popleft()
        get_task = asyncio.create_task(inbox.get())
        limit_task = asyncio.create_task(limit_event.wait())
        abuse_task = asyncio.create_task(abuse_event.wait())
        done, _ = await asyncio.wait(
            {get_task, limit_task, abuse_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in (get_task, limit_task, abuse_task):
            if task not in done:
                task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.gather(
                get_task, limit_task, abuse_task, return_exceptions=True
            )
        if get_task in done:
            return get_task.result()
        if abuse_task in done:
            return _ABUSE
        return _LIMIT_HIT

    async def _close_with_error(envelope: dict[str, Any]) -> None:
        await _send(websocket, envelope)
        with contextlib.suppress(Exception):
            await websocket.close()

    reader_task = asyncio.create_task(_reader())
    limiter_task = asyncio.create_task(_limiter())
    try:
        while True:
            message = await _read_message()
            if message is _LIMIT_HIT:
                await _close_on_limit()
                return
            if message is _ABUSE:
                await _close_with_error(
                    _error_envelope(
                        400,
                        "invalid_request",
                        "too many queued messages",
                    )
                )
                return
            if message.get("type") == "websocket.disconnect":
                return
            # Decision 8, per-message path.
            if _now() - start >= WS_CONNECTION_LIMIT_SECONDS:
                await _close_on_limit()
                return
            try:
                params = await _prepare_turn(websocket, scheduler, cache, message)
                if params is None:
                    continue
                if await _run_turn_raced(
                    websocket, scheduler, cache, params, inbox, pending
                ):
                    return  # client disconnected mid-turn
            except Exception:  # noqa: BLE001 - keep the socket open
                # Details go to the log only; the client gets a generic 500.
                logger.exception("responses websocket turn failed")
                if not await _send(
                    websocket,
                    _error_envelope(500, "server_error", "internal server error"),
                ):
                    return
    finally:
        for task in (reader_task, limiter_task):
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.gather(reader_task, limiter_task, return_exceptions=True)
