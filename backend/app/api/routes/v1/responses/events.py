"""OpenResponses streaming events (spec v2026-04-24).

Semantic SSE events with monotonic sequence_number. Event `type` values
match the spec; the SSE `event:` field equals `type` in the body.
Terminal SSE line is the literal [DONE].
"""

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel

from app.api.routes.v1.responses.schemas import (
    Error,
    IncompleteDetails,
    ResponseResource,
    Usage,
)

EventType = Literal[
    "response.created",
    "response.queued",
    "response.in_progress",
    "response.completed",
    "response.failed",
    "response.incomplete",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.refusal.delta",
    "response.refusal.done",
    "response.reasoning.delta",
    "response.reasoning.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.output_text.annotation.added",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "error",
]


class StreamingEvent(BaseModel):
    """Base for all streaming events; subclasses add payload fields."""

    type: EventType
    sequence_number: int

    model_config = {"extra": "allow"}


class SSEmitter:
    """Assigns monotonic sequence numbers and frames SSE lines.

    make() records every built event into a pending queue; generators
    drain it with drain_frames() so item/delta events built by StreamState
    actually reach the wire.
    """

    def __init__(self) -> None:
        self._seq = 0
        self._pending: list[dict[str, Any]] = []

    @property
    def sequence_number(self) -> int:
        return self._seq

    def next_sequence(self) -> int:
        self._seq += 1
        return self._seq

    def frame(self, event: dict[str, Any]) -> str:
        """Frame one event as an SSE block (event: + data: lines)."""
        etype = event["type"]
        return f"event: {etype}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    def make(self, etype: EventType, **payload: Any) -> dict[str, Any]:
        event = {"type": etype, "sequence_number": self.next_sequence(), **payload}
        self._pending.append(event)
        return event

    def drain_frames(self) -> list[str]:
        """Frame and clear all pending events."""
        frames = [self.frame(event) for event in self._pending]
        self._pending.clear()
        return frames

    async def stream(self, events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
        """Wrap an event iterator, framing each event and ending with [DONE]."""
        async for event in events:
            yield self.frame(event)
        yield "data: [DONE]\n\n"


# ============================================================================
# Event builders (spec shapes)
# ============================================================================


def response_created(seq: SSEmitter, response: ResponseResource) -> dict[str, Any]:
    return seq.make("response.created", response=response.model_dump())


def response_queued(seq: SSEmitter, response: ResponseResource) -> dict[str, Any]:
    return seq.make("response.queued", response=response.model_dump())


def response_in_progress(seq: SSEmitter, response: ResponseResource) -> dict[str, Any]:
    return seq.make("response.in_progress", response=response.model_dump())


def response_completed(seq: SSEmitter, response: ResponseResource) -> dict[str, Any]:
    return seq.make("response.completed", response=response.model_dump())


def response_failed(seq: SSEmitter, response: ResponseResource) -> dict[str, Any]:
    return seq.make("response.failed", response=response.model_dump())


def response_incomplete(seq: SSEmitter, response: ResponseResource) -> dict[str, Any]:
    return seq.make("response.incomplete", response=response.model_dump())


def output_item_added(
    seq: SSEmitter, output_index: int, item: dict[str, Any]
) -> dict[str, Any]:
    return seq.make("response.output_item.added", output_index=output_index, item=item)


def output_item_done(
    seq: SSEmitter, output_index: int, item: dict[str, Any]
) -> dict[str, Any]:
    return seq.make("response.output_item.done", output_index=output_index, item=item)


def content_part_added(
    seq: SSEmitter,
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict[str, Any],
) -> dict[str, Any]:
    return seq.make(
        "response.content_part.added",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        part=part,
    )


def content_part_done(
    seq: SSEmitter,
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict[str, Any],
) -> dict[str, Any]:
    return seq.make(
        "response.content_part.done",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        part=part,
    )


def output_text_delta(
    seq: SSEmitter, item_id: str, output_index: int, content_index: int, delta: str
) -> dict[str, Any]:
    return seq.make(
        "response.output_text.delta",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def output_text_done(
    seq: SSEmitter, item_id: str, output_index: int, content_index: int, text: str
) -> dict[str, Any]:
    return seq.make(
        "response.output_text.done",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        text=text,
    )


def refusal_delta(
    seq: SSEmitter, item_id: str, output_index: int, content_index: int, delta: str
) -> dict[str, Any]:
    return seq.make(
        "response.refusal.delta",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def refusal_done(
    seq: SSEmitter, item_id: str, output_index: int, content_index: int, refusal: str
) -> dict[str, Any]:
    return seq.make(
        "response.refusal.done",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        refusal=refusal,
    )


def reasoning_delta(
    seq: SSEmitter, item_id: str, output_index: int, content_index: int, delta: str
) -> dict[str, Any]:
    return seq.make(
        "response.reasoning.delta",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def reasoning_done(
    seq: SSEmitter, item_id: str, output_index: int, content_index: int, text: str
) -> dict[str, Any]:
    return seq.make(
        "response.reasoning.done",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        text=text,
    )


def reasoning_summary_text_delta(
    seq: SSEmitter,
    item_id: str,
    output_index: int,
    content_index: int,
    summary_index: int,
    delta: str,
) -> dict[str, Any]:
    return seq.make(
        "response.reasoning_summary_text.delta",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        summary_index=summary_index,
        delta=delta,
    )


def reasoning_summary_text_done(
    seq: SSEmitter,
    item_id: str,
    output_index: int,
    content_index: int,
    summary_index: int,
    text: str,
) -> dict[str, Any]:
    return seq.make(
        "response.reasoning_summary_text.done",
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        summary_index=summary_index,
        text=text,
    )


def reasoning_summary_part_added(
    seq: SSEmitter,
    item_id: str,
    output_index: int,
    summary_index: int,
    part: dict[str, Any],
) -> dict[str, Any]:
    return seq.make(
        "response.reasoning_summary_part.added",
        item_id=item_id,
        output_index=output_index,
        summary_index=summary_index,
        part=part,
    )


def reasoning_summary_part_done(
    seq: SSEmitter,
    item_id: str,
    output_index: int,
    summary_index: int,
    part: dict[str, Any],
) -> dict[str, Any]:
    return seq.make(
        "response.reasoning_summary_part.done",
        item_id=item_id,
        output_index=output_index,
        summary_index=summary_index,
        part=part,
    )


def function_call_arguments_delta(
    seq: SSEmitter, item_id: str, output_index: int, delta: str
) -> dict[str, Any]:
    return seq.make(
        "response.function_call_arguments.delta",
        item_id=item_id,
        output_index=output_index,
        delta=delta,
    )


def function_call_arguments_done(
    seq: SSEmitter, item_id: str, output_index: int, arguments: str
) -> dict[str, Any]:
    return seq.make(
        "response.function_call_arguments.done",
        item_id=item_id,
        output_index=output_index,
        arguments=arguments,
    )


def error_event(seq: SSEmitter, error: Error) -> dict[str, Any]:
    return seq.make("error", error=error.model_dump())


def incomplete_response(response: ResponseResource, reason: str) -> ResponseResource:
    """Mark a response incomplete with details."""
    response.status = "incomplete"
    response.incomplete_details = IncompleteDetails(reason=reason)
    return response


def failed_response(
    response: ResponseResource, code: str, message: str
) -> ResponseResource:
    """Mark a response failed with an error."""
    response.status = "failed"
    response.error = Error(code=code, message=message)
    return response


def usage_from_llama(
    prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0
) -> Usage:
    return Usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        input_tokens_details={"cached_tokens": 0},
        output_tokens_details={"reasoning_tokens": reasoning_tokens},
    )


__all__ = [
    "EventType",
    "StreamingEvent",
    "SSEmitter",
    "content_part_added",
    "content_part_done",
    "error_event",
    "failed_response",
    "function_call_arguments_delta",
    "function_call_arguments_done",
    "incomplete_response",
    "output_item_added",
    "output_item_done",
    "output_text_delta",
    "output_text_done",
    "reasoning_delta",
    "reasoning_done",
    "reasoning_summary_part_added",
    "reasoning_summary_part_done",
    "reasoning_summary_text_delta",
    "reasoning_summary_text_done",
    "refusal_delta",
    "refusal_done",
    "response_completed",
    "response_created",
    "response_failed",
    "response_in_progress",
    "response_incomplete",
    "response_queued",
    "usage_from_llama",
]
