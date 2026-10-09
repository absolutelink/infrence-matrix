"""Server-Sent Events emitter for /v1/responses (Phase 6).

Consumes litellm's (pass-through-faithful) OpenResponses events and
re-frames them for the client with the admin's own identity and ordering:

- The admin owns the client-facing ``resp_<uuid>``: on every lifecycle
  frame that carries a ``response`` object (created / in_progress /
  completed / failed / incomplete), ``response.id`` is replaced with our
  minted id. litellm's affinity-wrapped upstream id is captured by the
  route for persistence (FINDINGS.md §1).
- ``sequence_number`` is reassigned monotonically (0, 1, 2, ...) on every
  emitted event, overwriting whatever the upstream sent, so the client
  sees a gapless stream even if we synthesize terminal frames.
- Spec-required positional fields (``output_index``/``content_index``/
  ``summary_index``/``item_id``) that some backends (e.g. llama.cpp)
  omit from item/part/delta events are filled from the emitter's tracked
  stream position — provider-sent values are never overwritten. The
  tracking assumes items stream sequentially (one in flight at a time).
  ``output_text`` parts missing their required ``annotations`` array are
  likewise filled with ``[]``.
- Everything else (usage, output[], non-canonical event types) passes
  through untouched — native streaming preserves them (FINDINGS.md
  results table).
- ``failed()`` synthesizes the spec ``response.failed`` frame because
  litellm raises ``MidStreamFallbackError`` instead of yielding it
  (FINDINGS.md §2).

litellm events are pydantic objects or plain dicts; :func:`to_dict`
normalizes both. Pydantic serializer warnings on ``model_dump()`` of
plain-dict upstream payloads are suppressed (``warnings=False``) since
they are expected noise (values still correct).
"""

import enum
import json
import logging
from typing import Any

logger = logging.getLogger("admin.sse")


def event_type_of(value: Any) -> str:
    """Resolve an event ``type`` field to its dotted string.

    litellm returns some events (``output_item.added``, ``content_part.*``,
    ``output_text.delta/done``) with ``type`` as a
    ``ResponsesAPIStreamEvents`` member — a ``(str, Enum)`` mixin whose
    ``str()`` is the class-qualified name (``"ResponsesAPIStreamEvents.
    OUTPUT_TEXT_DELTA"``), not the dotted value. Unwrap ``.value`` so the
    SSE ``event:`` line is always spec-clean. Plain strings pass through.
    """
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


# Lifecycle event types whose `response` object carries the client-facing id.
LIFECYCLE_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.queued",
        "response.completed",
        "response.failed",
        "response.incomplete",
    }
)


def to_dict(event: Any) -> dict[str, Any]:
    """Tolerantly convert a litellm event (object or dict) to a dict."""
    if isinstance(event, dict):
        return event
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        try:
            # litellm stuffs raw dicts into typed model fields (part/item/usage);
            # serializer warnings are expected noise, output is unaffected.
            return dump(exclude_none=False, warnings=False)
        except Exception:  # noqa: BLE001 - fall back to a loose conversion
            logger.debug("model_dump failed for event %r", type(event), exc_info=True)
    try:
        return dict(event)  # type: ignore[call-overload]
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"cannot convert event {type(event)!r} to dict") from exc


# Event types whose spec schema requires positional fields that some
# backends (e.g. llama.cpp) omit. The emitter tracks the current item /
# content position as events flow and fills any missing required key.
_INDEXED_EVENTS: dict[str, tuple[str, ...]] = {
    "response.output_item.added": ("output_index",),
    "response.output_item.done": ("output_index",),
    "response.content_part.added": ("item_id", "output_index", "content_index"),
    "response.content_part.done": ("item_id", "output_index", "content_index"),
    "response.output_text.delta": ("item_id", "output_index", "content_index"),
    "response.output_text.done": ("item_id", "output_index", "content_index"),
    "response.refusal.delta": ("item_id", "output_index", "content_index"),
    "response.refusal.done": ("item_id", "output_index", "content_index"),
    "response.function_call_arguments.delta": ("item_id", "output_index"),
    "response.function_call_arguments.done": ("item_id", "output_index"),
    "response.reasoning.delta": ("item_id", "output_index", "content_index"),
    "response.reasoning.done": ("item_id", "output_index", "content_index"),
    "response.reasoning_summary_part.added": (
        "item_id",
        "output_index",
        "summary_index",
    ),
    "response.reasoning_summary_part.done": (
        "item_id",
        "output_index",
        "summary_index",
    ),
    "response.reasoning_summary_text.delta": (
        "item_id",
        "output_index",
        "summary_index",
    ),
    "response.reasoning_summary_text.done": (
        "item_id",
        "output_index",
        "summary_index",
    ),
    "response.output_text.annotation.added": (
        "item_id",
        "output_index",
        "content_index",
        "annotation_index",
    ),
}


class SSEEmitter:
    """Frames OpenResponses events as SSE with our id and sequence numbers."""

    def __init__(self, client_response_id: str) -> None:
        self.client_response_id = client_response_id
        self._seq = 0
        self._output_index = -1
        self._content_index = -1
        self._summary_index = -1
        self._annotation_index = -1
        self._item_id: str | None = None

    def _track_position(self, event_type: str, data: dict[str, Any]) -> None:
        """Advance the item/content/summary position from events that
        carry it. Assumes items stream sequentially (one in flight)."""
        if event_type == "response.output_item.added":
            self._output_index += 1
            self._content_index = -1
            self._summary_index = -1
            self._annotation_index = -1
            item = data.get("item")
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                self._item_id = item["id"]
        elif event_type == "response.content_part.added":
            self._content_index += 1
            self._annotation_index = -1
        elif event_type == "response.reasoning_summary_part.added":
            self._summary_index += 1
        elif event_type == "response.output_text.annotation.added":
            self._annotation_index += 1
        if isinstance(data.get("item_id"), str):
            self._item_id = data["item_id"]
        if isinstance(data.get("output_index"), int):
            self._output_index = data["output_index"]
        if isinstance(data.get("content_index"), int):
            self._content_index = data["content_index"]
        if isinstance(data.get("summary_index"), int):
            self._summary_index = data["summary_index"]
        if isinstance(data.get("annotation_index"), int):
            self._annotation_index = data["annotation_index"]

    def _fill_position(self, event_type: str, data: dict[str, Any]) -> None:
        """Fill required positional keys the provider omitted (never
        overwrite values the provider sent). Indices seen before any
        corresponding ``*.added`` event clamp to 0."""
        for key in _INDEXED_EVENTS.get(event_type, ()):
            if data.get(key) is not None:
                continue
            value: Any = {
                "output_index": max(self._output_index, 0),
                # Deltas before any content_part.added belong to part 0.
                "content_index": max(self._content_index, 0),
                "summary_index": max(self._summary_index, 0),
                "annotation_index": max(self._annotation_index, 0),
                "item_id": self._item_id,
            }[key]
            if value is not None:
                data[key] = value

    def event_data(self, event: Any) -> dict[str, Any]:
        """Produce the normalized event dict: replace our id on lifecycle
        frames, reassign sequence_number, and fill spec-required positional
        fields some backends omit. Transport-agnostic — SSE serializes the
        result with :meth:`frame`; the WebSocket transport (Phase 21) emits
        the dict as JSON.

        Operates on a shallow copy so the caller's ``response`` sub-dict
        is never mutated (the wrapped id stays intact for capture)."""
        data = dict(to_dict(event))
        event_type = event_type_of(data.get("type", "message"))
        data["type"] = event_type
        if event_type in LIFECYCLE_EVENTS:
            response = data.get("response")
            if isinstance(response, dict):
                response = dict(response)
                response["id"] = self.client_response_id
                data["response"] = response
        self._track_position(event_type, data)
        self._fill_position(event_type, data)
        if event_type in ("response.content_part.added", "response.content_part.done"):
            # output_text parts require an `annotations` array; llama.cpp's
            # content_part.added omits it. Fill on a copy (shared upstream).
            part = data.get("part")
            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and "annotations" not in part
            ):
                data["part"] = {**part, "annotations": []}
        data["sequence_number"] = self._seq
        self._seq += 1
        return data

    def frame(self, event: Any) -> str:
        """Serialize one event as SSE ``event:``/``data:`` lines.

        Delegates dict production to :meth:`event_data` (which assigns the
        ``sequence_number`` and unwraps the ``type``) so the SSE and
        WebSocket transports share identical event content; here we only
        format. Byte-identical to the pre-split implementation."""
        data = self.event_data(event)
        return _sse(data["type"], data)

    def done(self) -> str:
        """Stream terminator (SSE-only; the WebSocket transport has no
        ``[DONE]`` — terminal events end a turn there)."""
        return "data: [DONE]\n\n"

    def failed_events(
        self, error: dict[str, Any], response: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Synthesize the spec terminal ``response.failed`` event dicts.

        ``response`` is the caller's spec-normalized response skeleton
        (the suite validates failed frames against the full
        ResponseResource schema); the id/status/error are always owned
        by the emitter. Returns the event dicts (the failed lifecycle
        event plus a trailing ``error`` event), each already carrying its
        ``sequence_number``. Transport-agnostic: SSE serializes them with
        :meth:`failed`; the WebSocket transport emits them as JSON.
        """
        # The streaming `error` event schema requires `param` (nullable).
        error = {**error, "param": error.get("param")}
        body: dict[str, Any] = dict(response) if response else {}
        body.update(
            {
                "id": self.client_response_id,
                "object": "response",
                "status": "failed",
                "error": error,
            }
        )
        failed_event: dict[str, Any] = {
            "type": "response.failed",
            "response": body,
            "sequence_number": self._seq,
        }
        self._seq += 1
        error_event: dict[str, Any] = {
            "type": "error",
            "error": error,
            "sequence_number": self._seq,
        }
        self._seq += 1
        return [failed_event, error_event]

    def failed(
        self, error: dict[str, Any], response: dict[str, Any] | None = None
    ) -> list[str]:
        """Serialize :meth:`failed_events` to SSE strings so callers can
        ``yield from`` the failed lifecycle frame plus a trailing
        ``error`` frame."""
        return [
            _sse(event["type"], event) for event in self.failed_events(error, response)
        ]


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
