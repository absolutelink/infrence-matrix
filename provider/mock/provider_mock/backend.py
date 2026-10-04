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
import time
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

    def _base_response(self, response_id: str, model: str) -> dict[str, Any]:
        """A spec-complete ResponseResource skeleton.

        The openresponses.org schema marks most fields required (nullable
        or defaulted); emitting them here keeps the pass-through admin
        emitter compliant without inventing transformations.
        """
        now = int(time.time())
        return {
            "id": response_id,
            "object": "response",
            "created_at": now,
            "completed_at": None,
            "status": "in_progress",
            "incomplete_details": None,
            "model": model,
            "previous_response_id": None,
            "instructions": None,
            "output": [],
            "error": None,
            "tools": [],
            "tool_choice": "auto",
            "truncation": "disabled",
            "parallel_tool_calls": True,
            "text": {"format": {"type": "text"}},
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_logprobs": 0,
            "temperature": 1,
            "reasoning": {"effort": None, "summary": None},
            "usage": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "store": True,
            "background": False,
            "service_tier": "default",
            "metadata": {},
            "safety_identifier": None,
            "prompt_cache_key": None,
        }

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        return self._stream_responses(request)

    async def _stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        self.stream_open_count += 1
        tools = request.get("tools")
        if isinstance(tools, list) and tools:
            async for event in self._stream_tool_call(request, tools):
                yield event
            return
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        item_id = f"msg_{uuid.uuid4().hex[:12]}"
        model = request.get("model") or self.model
        try:
            base_response = self._base_response(response_id, model)
            yield {"type": "response.created", "response": dict(base_response)}
            yield {"type": "response.in_progress", "response": dict(base_response)}
            yield {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }
            yield {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
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
                "part": {"type": "output_text", "text": text, "annotations": []},
            }
            message_item = {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            completion_tokens = max(1, self.delta_count)
            prompt_tokens = 1
            yield {
                "type": "response.completed",
                "response": {
                    **base_response,
                    "status": "completed",
                    "completed_at": int(time.time()),
                    "output": [message_item],
                    # Spec names plus legacy aliases so either consumer works.
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                },
            }
        finally:
            self.stream_close_count += 1

    async def _stream_tool_call(
        self, request: dict[str, Any], tools: list[Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Canned function_call turn: calls the first tool with fixed args.

        Lets the full tool-forwarding path (client -> litellm -> provider)
        be exercised without a real model.
        """
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        item_id = f"fc_{uuid.uuid4().hex[:12]}"
        model = request.get("model") or self.model
        first = tools[0] if isinstance(tools[0], dict) else {}
        name = first.get("name") or "mock_tool"
        arguments = '{"location": "San Francisco, CA"}'
        base_response = self._base_response(response_id, model)
        yield {"type": "response.created", "response": dict(base_response)}
        yield {"type": "response.in_progress", "response": dict(base_response)}
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": f"call_{uuid.uuid4().hex[:12]}",
                "name": name,
                "arguments": "",
            },
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "output_index": 0,
            "delta": arguments,
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": item_id,
            "output_index": 0,
            "arguments": arguments,
        }
        call_item = {
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": f"call_{uuid.uuid4().hex[:12]}",
            "name": name,
            "arguments": arguments,
        }
        yield {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": call_item,
        }
        completion_tokens = max(1, self.delta_count)
        yield {
            "type": "response.completed",
            "response": {
                **base_response,
                "status": "completed",
                "completed_at": int(time.time()),
                "output": [call_item],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": completion_tokens,
                    "total_tokens": 1 + completion_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        }

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
            completion_tokens = max(1, self.delta_count)
            prompt_tokens = 1
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                # Honors stream_options.include_usage: final usage chunk
                # with empty delta so admin can persist token counts.
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        finally:
            self.stream_close_count += 1
