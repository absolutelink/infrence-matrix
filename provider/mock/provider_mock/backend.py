"""Mock backend driver for hardware-free provider development.

Implements `provider_lib.backend.BackendDriver` with instant,
deterministic behavior: `start()` always succeeds, `stream_responses()`
emits a configurable OpenResponses SSE event sequence, and
`stream_chat_completions()` emits OpenAI chat.completion.chunk events.

The stream generator records its own close (normal exhaustion or early
aclose) in `stream_close_count` so tests can assert the
release-on-upstream-close invariant end to end.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

from provider_lib.backend import BackendDriver


class MockBackend(BackendDriver):
    """A backend that fakes inference with a canned token stream."""

    def __init__(
        self,
        model: str = "mock-model",
        *,
        delta_count: int = 3,
        delta_delay: float = 0.0,
        hold: asyncio.Event | None = None,
    ) -> None:
        self.model = model
        self.delta_count = delta_count
        self.delta_delay = delta_delay
        # When set, the stream pauses until this event is set before
        # finishing; lets tests hold an upstream stream open on demand.
        self.hold = hold
        self.start_calls = 0
        self.stop_calls = 0
        self.started = False
        self.stream_open_count = 0
        self.stream_close_count = 0

    async def start(self) -> None:
        self.start_calls += 1
        self.started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.started = False

    async def health(self) -> bool:
        return self.started

    async def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": self.model,
                "object": "model",
                "owned_by": "inference-matrix-mock",
            }
        ]

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        return self._stream_responses(request)

    async def _stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        self.stream_open_count += 1
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        item_id = f"msg_{uuid.uuid4().hex[:12]}"
        model = request.get("model") or self.model
        try:
            base_response: dict[str, Any] = {
                "id": response_id,
                "object": "response",
                "model": model,
                "status": "in_progress",
                "output": [],
            }
            yield {"type": "response.created", "response": dict(base_response)}
            yield {"type": "response.in_progress", "response": dict(base_response)}
            yield {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            }
            yield {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            }
            text = ""
            for i in range(self.delta_count):
                if self.hold is not None:
                    await self.hold.wait()
                chunk = f"mock-token-{i} "
                text += chunk
                if self.delta_delay:
                    await asyncio.sleep(self.delta_delay)
                yield {
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": chunk,
                }
            yield {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            }
            yield {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text},
            }
            message_item = {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
            completion_tokens = max(1, self.delta_count)
            prompt_tokens = 1
            yield {
                "type": "response.completed",
                "response": {
                    **base_response,
                    "status": "completed",
                    "output": [message_item],
                    # Spec names plus legacy aliases so either consumer works.
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                },
            }
        finally:
            self.stream_close_count += 1

    def stream_chat_completions(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        return self._stream_chat_completions(request)

    async def _stream_chat_completions(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        self.stream_open_count += 1
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = request.get("model") or self.model
        try:
            for i in range(self.delta_count):
                if self.hold is not None:
                    await self.hold.wait()
                chunk = f"mock-token-{i} "
                if self.delta_delay:
                    await asyncio.sleep(self.delta_delay)
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        finally:
            self.stream_close_count += 1
