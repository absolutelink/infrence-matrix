"""Mock backend driver for hardware-free provider development.

Implements `provider_lib.backend.BackendDriver` with instant,
deterministic behavior: `start()` always succeeds, `stream_responses()`
emits a configurable OpenResponses SSE event sequence, and
`stream_chat_completions()` emits OpenAI chat.completion.chunk events.
`embeddings()` returns a deterministic, spec-shaped CreateEmbeddingResponse
(fake fixed-length vectors derived from sha256) so the full local
`/v1/embeddings` path runs with no GPU (Phase 18 slice 6).

Phase 24 (audio): `speech()` returns a deterministic sine-wave WAV (or
incremental PCM chunks), `transcribe()` echoes a fake transcript, and an
in-memory voice store (seeded with two voices) backs `voices()` /
`voice_put()` / `voice_delete()`, so the whole `/v1/audio/*` data plane runs
with no GPU.

The stream generator records its own close (normal exhaustion or early
aclose) in `stream_close_count` so tests can assert the
release-on-upstream-close invariant end to end.
"""

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import math
import struct
import time
import uuid
import wave
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from provider_lib.backend import BackendDriver, SpeechStream
from provider_lib.models import ModelSpec
from provider_lib.schema import load_schema, validate_backend_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import WebSocket

logger = logging.getLogger("provider.mock.backend")

# The mock's committed backend_config schema (Phase 12): the sectioned
# shape the admin registers, validates against, and renders in the UI.
SCHEMA: dict[str, Any] = load_schema("provider_mock")


class MockBackend(BackendDriver):
    """A backend that fakes inference with a canned token stream."""

    # Phase 18 slice 6: fixed dimensionality of the mock's fake embedding
    # vectors (sha256 yields 32 bytes, so 8 dims is comfortably derivable).
    EMBEDDING_DIM = 8

    def __init__(
        self,
        model: str = "mock-model",
        *,
        delta_count: int = 3,
        delta_delay: float = 0.0,
        hold: asyncio.Event | None = None,
    ) -> None:
        self.model = model
        # Phase 25 multi-model: the served-model list adopted via set_models.
        # Empty (default) => list_models falls back to the single ``self.model``
        # so legacy single-model behavior is byte-identical.
        self.models: list[ModelSpec] = []
        self.set_models_calls = 0
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
        # Phase 18 slice 6: embeddings call counter (mirrors the stream
        # counters) so tests can assert the driver was reached.
        self.embeddings_calls = 0
        # Phase 24 (audio): call counters + close tracking for the speech /
        # transcription surface (mirrors the stream/embeddings counters).
        self.speech_calls = 0
        self.speech_close_count = 0
        self.transcribe_calls = 0
        self.transcribe_stream_calls = 0
        # Audio behavioral knobs (overridable via backend_config `audio`).
        self.audio_sample_rate = 24000
        self.audio_duration = 0.5
        self.audio_frequency = 440
        self.pcm_chunk_count = 4
        self.pcm_chunk_delay = 0.0
        # In-memory saved-voice store, seeded with two fake voices so the
        # catalog + enrollment roundtrip is exercisable with no engine.
        self._voices: dict[str, dict[str, Any]] = {
            "alloy": {"voice_id": "alloy", "name": "alloy", "language": "en"},
            "echo": {"voice_id": "echo", "name": "echo", "language": "en"},
        }
        # Phase 25: the served name the last voice op targeted (assertable).
        self.last_voice_model: str | None = None
        # Phase 9: artifact tracking for storage.prune_unused. The mock
        # downloads nothing; tests can seed this list directly.
        self.resolved_artifacts: list[str] = []
        self.applied_configs: list[dict[str, Any]] = []
        # Phase 12: the sectioned backend_config, recorded per section so
        # tests (and log inspection) can assert what the driver read.
        # The mock fakes everything: these values never drive a real
        # engine, they only prove the nested config paths flow.
        self.context: dict[str, Any] = {}
        self.sampling: dict[str, Any] = {}
        self.server: dict[str, Any] = {}
        self.artifacts: dict[str, Any] = {}

    def apply_config(self, backend_config: dict[str, Any]) -> None:
        """Adopt the sectioned backend_config (Phase 12 shape).

        Sections: `artifacts` (hf-file descriptors; accepted, never
        downloaded), `context` (ctx/predict), `sampling`
        (temperature/top_k/top_p/seed), `stream` (delta_count/
        delta_delay — the mock's real behavioral knobs), `server`
        (backend_port). Schema violations are logged as warnings, not
        raised: the authoritative gates are the admin's registration
        gate and definition validation; the mock stays maximally
        permissive for dev flows.
        """
        cfg = backend_config or {}
        self.applied_configs.append(dict(cfg))
        errors = validate_backend_config(SCHEMA, cfg)
        for err in errors:
            logger.warning("backend_config schema violation: %s", err)

        stream = cfg.get("stream") or {}
        if isinstance(stream.get("delta_count"), int):
            self.delta_count = stream["delta_count"]
        if isinstance(stream.get("delta_delay"), (int, float)):
            self.delta_delay = float(stream["delta_delay"])

        audio = cfg.get("audio") or {}
        if isinstance(audio.get("sample_rate"), int):
            self.audio_sample_rate = audio["sample_rate"]
        if isinstance(audio.get("duration_seconds"), (int, float)):
            self.audio_duration = float(audio["duration_seconds"])
        if isinstance(audio.get("frequency"), int):
            self.audio_frequency = audio["frequency"]
        if isinstance(audio.get("pcm_chunk_count"), int):
            self.pcm_chunk_count = max(1, audio["pcm_chunk_count"])
        if isinstance(audio.get("pcm_chunk_delay"), (int, float)):
            self.pcm_chunk_delay = float(audio["pcm_chunk_delay"])

        self.artifacts = dict(cfg.get("artifacts") or {})
        self.context = dict(cfg.get("context") or {})
        self.sampling = dict(cfg.get("sampling") or {})
        self.server = dict(cfg.get("server") or {})
        # Config changed: previously resolved artifacts are stale.
        self.resolved_artifacts = []

    async def start(self) -> None:
        self.start_calls += 1
        self.started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.started = False

    async def health(self) -> bool:
        return self.started

    def set_models(self, models: list[ModelSpec]) -> None:
        """Adopt the served-model list (Phase 25 multi-model).

        Records the specs (so ``list_models`` advertises every enabled served
        name) and mirrors the first enabled name onto ``self.model`` for
        back-compat with the single-model path. The mock serves all names from
        its one lifecycle; ``stream_responses`` echoes the requested name back.
        """
        self.set_models_calls += 1
        self.models = list(models)
        enabled = [m.name for m in models if m.enabled]
        if enabled:
            self.model = enabled[0]

    async def list_models(self) -> list[dict[str, Any]]:
        names = (
            [m.name for m in self.models if m.enabled] if self.models else [self.model]
        )
        return [
            {
                "id": name,
                "object": "model",
                "owned_by": "inference-matrix-mock",
            }
            for name in names
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
                        # Phase 23: representative rate + latency telemetry so
                        # dev/smoke exercises the admin persist_turn path.
                        "completion_tokens_details": {
                            "prompt_per_second": 1412.21,
                            "predicted_per_second": 36.2534,
                            "prompt_time": 0.12,
                            "prediction_time": 3.4,
                        },
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
        tools = request.get("tools")
        if isinstance(tools, list) and tools:
            async for chunk in self._stream_chat_tool_call(completion_id, model, tools):
                yield chunk
            return
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
                    # Phase 23: representative rate + latency telemetry so
                    # dev/smoke exercises the admin persist_turn path.
                    "completion_tokens_details": {
                        "prompt_per_second": 1412.21,
                        "predicted_per_second": 36.2534,
                        "prompt_time": 0.12,
                        "prediction_time": 3.4,
                    },
                },
            }
        finally:
            self.stream_close_count += 1

    async def _stream_chat_tool_call(
        self, completion_id: str, model: str, tools: list[Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Canned chat.completion tool_call turn (mirrors ``_stream_tool_call``
        for the chat surface): calls the first tool with fixed arguments so the
        full chat tool-forwarding path (client -> litellm -> provider) is
        exercisable without a real model."""
        first = tools[0] if isinstance(tools[0], dict) else {}
        fn = first.get("function") if isinstance(first.get("function"), dict) else {}
        name = fn.get("name") or first.get("name") or "mock_tool"
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        arguments = '{"location": "San Francisco, CA"}'
        completion_tokens = max(1, self.delta_count)
        prompt_tokens = 1
        try:
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": arguments}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        finally:
            self.stream_close_count += 1

    def _fake_vector(self, text: str) -> list[float]:
        """Deterministic fixed-length vector derived from sha256(text).

        Same text -> identical vector across processes (never Python's
        randomized ``hash()``); different text -> different vector. Each
        byte maps linearly to [-1.0, 1.0] and is rounded for stable JSON.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [round(byte / 127.5 - 1.0, 4) for byte in digest[: self.EMBEDDING_DIM]]

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return a spec-shaped, deterministic CreateEmbeddingResponse.

        Accepts ``input`` as a ``str`` or a ``list`` of strings (one
        embedding per item); an empty/missing input is treated as a single
        empty text. The ``usage`` carries a naive whitespace token estimate
        so the admin can persist a TokenUsageSample.
        """
        self.embeddings_calls += 1
        raw = request.get("input")
        if isinstance(raw, str):
            texts = [raw]
        elif isinstance(raw, list):
            texts = [t if isinstance(t, str) else json.dumps(t) for t in raw]
            if not texts:
                texts = [""]
        else:
            texts = [""]
        model = request.get("model") or self.model
        data = [
            {"object": "embedding", "index": i, "embedding": self._fake_vector(text)}
            for i, text in enumerate(texts)
        ]
        prompt_tokens = sum(max(1, len(text.split())) for text in texts)
        return {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    # ------------------------------------------------------------------
    # Phase 24 (audio): deterministic fakes for speech / transcription /
    # voices so the whole provider data-plane path runs with no GPU.
    # ------------------------------------------------------------------
    def _sine_pcm16(self, sample_rate: int, duration: float, freq: float) -> bytes:
        """Little-endian mono 16-bit sine-wave PCM frames (no container)."""
        n = max(1, int(sample_rate * duration))
        amp = 0.3 * 32767
        return b"".join(
            struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sample_rate)))
            for i in range(n)
        )

    def _make_wav(self, sample_rate: int, duration: float, freq: float) -> bytes:
        """Wrap the sine PCM in a valid single-frame WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(self._sine_pcm16(sample_rate, duration, freq))
        return buf.getvalue()

    def speech(self, request: dict[str, Any]) -> SpeechStream:
        """Return a deterministic fake speech stream.

        ``response_format`` selects the container: ``pcm`` yields raw 16-bit
        PCM split into ``pcm_chunk_count`` incremental chunks (with
        ``pcm_chunk_delay`` between them) and an ``X-Sample-Rate`` header;
        any other format (``wav`` default, ``mp3``/``opus``/``flac`` treated
        identically) yields a single valid WAV buffer. The byte generator is
        a closeable async generator so the lifecycle pump drives the
        release-on-upstream-close invariant end to end.
        """
        self.speech_calls += 1
        fmt = str(request.get("response_format") or "wav").lower()
        sr = self.audio_sample_rate
        if fmt == "pcm":
            headers = {
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": str(sr),
            }
            chunks = self._stream_pcm()
        else:
            headers = {"Content-Type": f"audio/{fmt}"}
            chunks = self._stream_wav(
                self._make_wav(sr, self.audio_duration, self.audio_frequency)
            )
        return SpeechStream(headers=headers, chunks=chunks)

    async def _stream_wav(self, data: bytes) -> AsyncIterator[bytes]:
        try:
            yield data
        finally:
            self.speech_close_count += 1

    async def _stream_pcm(self) -> AsyncIterator[bytes]:
        try:
            pcm = self._sine_pcm16(
                self.audio_sample_rate, self.audio_duration, self.audio_frequency
            )
            size = max(1, len(pcm) // self.pcm_chunk_count)
            for i in range(0, len(pcm), size):
                if self.pcm_chunk_delay:
                    await asyncio.sleep(self.pcm_chunk_delay)
                yield pcm[i : i + size]
        finally:
            self.speech_close_count += 1

    async def transcribe(
        self,
        form: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
    ) -> tuple[int, dict[str, str], bytes]:
        """Echo a deterministic transcription.

        The transcript text is derived from the ``prompt`` form field, else
        the first uploaded filename, else a fixed string. ``response_format``
        (json default / text) selects the body shape; other formats pass
        through as JSON. Returns ``(status, headers, body)`` for the relay.
        """
        self.transcribe_calls += 1
        text = form.get("prompt") or (files[0][1] if files else "") or "mock transcript"
        fmt = str(form.get("response_format") or "json").lower()
        if fmt == "text":
            return 200, {"content-type": "text/plain"}, text.encode()
        body = json.dumps({"text": text}).encode()
        return 200, {"content-type": "application/json"}, body

    async def voices(self, model: str | None = None) -> dict[str, Any]:
        """Return the in-memory voice catalog.

        ``model`` (the resolved served name, Phase 25 multi-model) is recorded
        for assertion; the mock's catalog is name-agnostic (single store).
        """
        self.last_voice_model = model
        return {
            "object": "list",
            "data": [
                {
                    "voice_id": v.get("voice_id", name),
                    "name": v.get("name", name),
                    "language": v.get("language"),
                }
                for name, v in self._voices.items()
            ],
        }

    async def voice_put(
        self,
        name: str,
        wav: bytes,
        ref_text: str | None,
        language: str | None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Store a saved voice in the in-memory catalog (fake enrollment)."""
        self.last_voice_model = model
        entry: dict[str, Any] = {"voice_id": name, "name": name}
        if language is not None:
            entry["language"] = language
        if ref_text is not None:
            entry["ref_text"] = ref_text
        self._voices[name] = entry
        return {
            "object": "voice",
            "voice_id": name,
            "name": name,
            "model": model,
            "language": language,
            "ref_text": ref_text,
            "size_bytes": len(wav),
            "deleted": False,
        }

    async def voice_delete(self, name: str, model: str | None = None) -> dict[str, Any]:
        """Remove a saved voice from the in-memory catalog."""
        self.last_voice_model = model
        existed = self._voices.pop(name, None) is not None
        return {"object": "voice", "voice_id": name, "model": model, "deleted": existed}

    async def transcribe_stream(self, websocket: WebSocket) -> None:
        """Live ASR bridge fake: send one canned transcript, then drain until
        the client disconnects (the lifecycle releases the slot on return)."""
        self.transcribe_stream_calls += 1
        from starlette.websockets import WebSocketState

        try:
            await websocket.send_json(
                {"type": "transcript", "text": "mock live transcript", "is_final": True}
            )
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
        except Exception:  # noqa: BLE001 - client gone / socket closed
            pass
        finally:
            if getattr(websocket, "client_state", None) == WebSocketState.CONNECTED:
                with contextlib.suppress(Exception):
                    await websocket.close()
