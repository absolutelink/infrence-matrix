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
- Everything else (usage, output[], non-canonical event types like
  ``response.reasoning_text.delta``) passes through untouched — native
  streaming preserves them (FINDINGS.md results table).
- ``failed()`` synthesizes the spec ``response.failed`` frame because
  litellm raises ``MidStreamFallbackError`` instead of yielding it
  (FINDINGS.md §2).

litellm events are pydantic objects or plain dicts; :func:`to_dict`
normalizes both. Pydantic serializer warnings on ``model_dump()`` of
plain-dict upstream payloads are harmless (values still correct).
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
            return dump(exclude_none=False)
        except Exception:  # noqa: BLE001 - fall back to a loose conversion
            logger.debug("model_dump failed for event %r", type(event), exc_info=True)
    try:
        return dict(event)  # type: ignore[call-overload]
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"cannot convert event {type(event)!r} to dict") from exc


class SSEEmitter:
    """Frames OpenResponses events as SSE with our id and sequence numbers."""

    def __init__(self, client_response_id: str) -> None:
        self.client_response_id = client_response_id
        self._seq = 0

    def frame(self, event: Any) -> str:
        """Serialize one event: replace our id on lifecycle frames,
        reassign sequence_number, and emit ``event:``/``data:`` lines.

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
        data["sequence_number"] = self._seq
        self._seq += 1
        return _sse(event_type, data)

    def done(self) -> str:
        """Stream terminator."""
        return "data: [DONE]\n\n"

    def failed(self, error: dict[str, Any]) -> list[str]:
        """Synthesize the spec terminal ``response.failed`` frame(s).

        Returns a list of SSE strings (the failed lifecycle frame plus a
        trailing ``error`` frame) so callers can ``yield from`` it.
        """
        failed_frame = _sse(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "id": self.client_response_id,
                    "object": "response",
                    "status": "failed",
                    "error": error,
                },
                "sequence_number": self._seq,
            },
        )
        self._seq += 1
        error_frame = _sse("error", {"type": "error", "error": error})
        return [failed_frame, error_frame]


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
