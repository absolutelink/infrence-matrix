"""Backend driver interface + lifecycle state machine for provider instances.

A provider package implements `BackendDriver` around its real inference
backend (llama.cpp `llama-server`, etc.). `BackendLifecycle` wraps a
driver with the admin-visible state machine (`backend.status` events) and
connection-lifecycle slot admission.

Slot admission invariant (critical): an acquired slot is released when
the UPSTREAM driver stream closes — exhausted, errored, or cancelled —
NOT when the downstream HTTP client finishes consuming the SSE
response. This is implemented by draining the driver into a queue from a
background producer task that owns the release, so a completed backend
request never holds a slot open because a client connection lingers
(the exact failure the legacy agent notes call out).

Provider-side admission is defense-in-depth; the admin scheduler
(Phase 6) is what queues requests globally.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from provider_lib.models import ModelSpec
from provider_lib.wire import BackendStatusValue

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard import cycle
    from fastapi import WebSocket

logger = logging.getLogger("provider.backend")

# Sentinel pushed onto the drain queue to mark upstream termination.
_STREAM_END = object()

# Backend states that accept new inference slots.
_SERVING_STATES = (BackendStatusValue.RUNNING, BackendStatusValue.IN_USE)

# async (new_status, reason) callback, wired to AdminClient.send_event.
StatusCallback = Callable[[str, str | None], Awaitable[None]]


class BackendNotReady(RuntimeError):
    """Backend is not in a serving state (-> HTTP 503)."""


class BackendBusy(RuntimeError):
    """All inference slots are in use (-> HTTP 429).

    Also raised by :meth:`BackendLifecycle.stop_if_idle` when a stop is
    refused because live requests hold slots (drain semantics). Carries
    ``in_flight`` so the caller can report the count that triggered the
    refusal without re-reading (and racing) the counter.
    """

    def __init__(
        self, message: str = "backend busy", *, in_flight: int | None = None
    ) -> None:
        super().__init__(message)
        self.in_flight = in_flight


class _StreamFailure:
    """Wraps an upstream exception for delivery through the drain queue."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


@dataclass
class SpeechStream:
    """Contract for a slot-admitted speech (TTS) response.

    A driver's :meth:`BackendDriver.speech` returns one of these instead of a
    bare byte iterator so it can convey the response headers the OpenAI TTS
    contract requires (``Content-Type`` and, for raw PCM, ``X-Sample-Rate``)
    alongside the audio chunks:

    * ``headers`` — response headers to place on the HTTP ``StreamingResponse``
      (e.g. ``{"Content-Type": "audio/wav"}`` for a buffered format, or
      ``{"Content-Type": "application/octet-stream", "X-Sample-Rate": "24000"}``
      for incremental PCM). The route layer forwards these verbatim.
    * ``chunks`` — an async iterator of already-encoded audio byte buffers.
      Buffered formats (wav/mp3/opus/...) yield once; streaming PCM yields
      incrementally. The iterator MUST be a closeable async generator so the
      lifecycle's pump can drive the release-on-upstream-close invariant
      exactly like the SSE streams.
    """

    headers: dict[str, str]
    chunks: AsyncIterator[bytes]


class BackendDriver(ABC):
    """Minimal per-type backend interface a provider package implements.

    `stream_responses` / `stream_chat_completions` are synchronous
    methods returning async iterators (typically async generator
    functions) so the lifecycle can start draining them inside a
    background producer task and observe their close semantics.
    """

    # Phase 18 slice 3: the endpoint kind this backend serves ("llm" |
    # "embedding"; "audio" reserved). Set by provider_lib from the wire
    # (the registration definition, the ``provider.config.update`` payload,
    # and the ``agent.assignments.update`` entry all carry ``modality``) at
    # every apply point; drivers read it to decide engine behavior (e.g.
    # llama-cpp boots ``llama-server --embedding`` for ``embedding``).
    # Defaults to ``"llm"`` so a package that never receives the field (or a
    # driver constructed directly in tests) behaves as a chat backend.
    modality: str = "llm"

    @abstractmethod
    async def start(self) -> None:
        """Bring the backend up. Idempotent-ish; raises on failure."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear the backend down."""

    @abstractmethod
    async def health(self) -> bool:
        """True when the backend is ready to serve inference."""

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """OpenAI model objects (entries of {"object":"list","data":[...]})."""

    @abstractmethod
    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield spec-shaped OpenResponses SSE event dicts for a request body.

        Must have async-generator semantics (closeable) so stream-close
        behavior is testable. The terminal `response.completed` event
        must carry `usage`.
        """

    def stream_chat_completions(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield OpenAI chat.completion.chunk dicts.

        Optional; providers without a chat surface leave the default
        (raises NotImplementedError -> HTTP 501).
        """
        raise NotImplementedError("this backend has no chat/completions stream")

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return a spec CreateEmbeddingResponse dict for an embedding request.

        Optional; providers without an embeddings surface leave the default
        (raises NotImplementedError -> HTTP 501).
        """
        raise NotImplementedError("this backend has no embeddings surface")

    # ------------------------------------------------------------------
    # Phase 24 (audio) — optional speech / transcription / voice surface.
    # Every hook defaults to NotImplementedError so a provider without an
    # audio surface answers 501 (exactly the embeddings pattern). The
    # lifecycle wraps the slot-consuming ones (speech / transcribe /
    # transcribe_stream) with the same eager-acquire + release-on-upstream-
    # close discipline as stream_responses / embeddings.
    # ------------------------------------------------------------------
    def speech(self, request: dict[str, Any]) -> SpeechStream:
        """Return a :class:`SpeechStream` (headers + audio byte chunks).

        Synchronous (returns an async iterator) so the lifecycle can start
        draining it in a background pump and observe close semantics, exactly
        like ``stream_responses``. Buffered formats yield once; streaming PCM
        yields incrementally. Optional; the default raises NotImplementedError
        -> HTTP 501.
        """
        raise NotImplementedError("this backend has no audio/speech surface")

    async def transcribe(
        self,
        form: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
    ) -> tuple[int, dict[str, str], bytes]:
        """Relay a multipart transcription request to the engine.

        ``form`` holds non-file text fields; ``files`` is a list of
        ``(field_name, filename, body_bytes, content_type)`` tuples. Returns
        the upstream ``(status, headers, body)`` triple so the response format
        (json / text / verbose_json / srt / vtt) passes through untouched.
        Optional; the default raises NotImplementedError -> HTTP 501.
        """
        raise NotImplementedError("this backend has no audio/transcription surface")

    async def voices(self) -> dict[str, Any]:
        """Return the saved-voice catalog (GET passthrough).

        Optional; the default raises NotImplementedError -> HTTP 501. Catalog
        reads do not consume an inference slot.
        """
        raise NotImplementedError("this backend has no voice catalog surface")

    async def voice_put(
        self,
        name: str,
        wav: bytes,
        ref_text: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        """Write a saved voice clip into the agent-side store (enrollment).

        The driver decides the on-disk layout (e.g. a ``custom-voices/`` dir
        with ``<name>.wav`` + sibling ``.txt``/``.lang`` sidecars). Returns a
        JSON-serializable summary dict. Optional; default -> HTTP 501. Does
        not consume an inference slot.
        """
        raise NotImplementedError("this backend has no voice enrollment surface")

    async def voice_delete(self, name: str) -> dict[str, Any]:
        """Remove a saved voice from the agent-side store. Returns a summary
        dict. Optional; default -> HTTP 501. Does not consume a slot."""
        raise NotImplementedError("this backend has no voice enrollment surface")

    async def transcribe_stream(self, websocket: WebSocket) -> None:
        """Bridge a live ASR WebSocket between the agent and the engine.

        The driver owns pumping bytes/messages between the agent-facing
        FastAPI ``WebSocket`` and its engine's WS. The lifecycle holds an
        inference slot for the full connection lifetime and releases it when
        this coroutine returns (on any disconnect). Optional; the default
        raises NotImplementedError -> the route closes the socket with 1011.
        """
        raise NotImplementedError("this backend has no live ASR stream surface")

    def set_models(self, models: list[ModelSpec]) -> None:  # noqa: B027
        """Adopt the backend's served-model list (Phase 25 multi-model).

        Called at handle construction (before ``start()``) and on every
        ``agent.assignments.update`` / ``provider.config.update`` that changes
        the served set, so a driver can rebuild its internal name->engine map
        (talkies: name->slug; gufo: ``served_model_name``; halogen-flash: which
        sub-model to invoke) **without a restart where possible**. ``models`` is
        always non-empty for a real entry (a single-model definition carries one
        synthesized spec). The default is a no-op for drivers that serve exactly
        one model or read the request's ``model`` field directly.
        """

    def apply_config(self, backend_config: dict[str, Any]) -> None:  # noqa: B027
        """Adopt a (possibly updated) backend_config.

        Called at registration and by the Phase 9 `provider.config.update`
        flow *before* `start()`, so the next boot picks up the new model
        artifacts / args / options. The default is a no-op for drivers
        that are configured entirely at construction time; drivers that
        read `backend_config` lazily in `start()` may ignore this.
        """

    async def aclose(self) -> None:  # noqa: B027  - optional hook, default no-op
        """Release driver-held resources (httpx clients, subprocesses)."""


class BackendLifecycle:
    """State machine + slot accounting around a `BackendDriver`.

    Emits every backend-status transition through `status_callback`
    (wired by the provider package to
    `AdminClient.send_event("backend.status", ...)`). Tracks
    `capacity` / `in_flight` locally and records `last_request_at` for
    the admin's idle-timeout logic (Phase 6).
    """

    def __init__(
        self,
        driver: BackendDriver,
        *,
        capacity: int = 1,
        status_callback: StatusCallback | None = None,
        instance_id: str | None = None,
    ) -> None:
        self._driver = driver
        self.capacity = max(1, capacity)
        self._status_callback = status_callback
        # Phase 16 (H3): the ProviderInstance id this lifecycle serves. An
        # agent may host several backends; per-backend commands and events
        # (backend.status / backend.metadata / backend.logs) are addressed by
        # this id so the right lifecycle is driven and the admin keys state
        # correctly.
        self.instance_id = instance_id
        self._status = BackendStatusValue.STOPPED
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self.last_request_at: datetime | None = None
        # Drivers that spend a long time between spawn and first healthy
        # response (halogen / halogen-flash: engine-side downloads) can
        # report progress without touching the state machine. See `report`.
        attach = getattr(driver, "attach_status_callback", None)
        if callable(attach):
            attach(self.report)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def backend_status(self) -> str:
        return self._status

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def driver(self) -> BackendDriver:
        return self._driver

    async def _emit(self, status: str, reason: str | None = None) -> None:
        if self._status_callback is None:
            return
        try:
            await self._status_callback(status, reason)
        except Exception:  # noqa: BLE001
            logger.exception("backend.status callback failed (status=%s)", status)

    async def report(self, status: str, reason: str | None = None) -> None:
        """Emit a `backend.status` frame WITHOUT moving the state machine.

        Drivers use this for long-init progress: a halogen-flash boot may
        download its checkpoint and companions before the API answers
        `/health`, and the operator deserves to see `initializing` (with the
        engine's current line) rather than a silent `starting`. Slot
        admission still reads `self._status`, which only `start()` /
        `stop()` change — a report can never make a not-yet-serving backend
        look servable.
        """
        await self._emit(status, reason)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """STOPPED -> STARTING -> (health ok) -> RUNNING; on failure -> ERROR.

        No-op when already RUNNING/IN_USE (idempotent).
        """
        async with self._lock:
            if self._status in _SERVING_STATES:
                return
            self._status = BackendStatusValue.STARTING
            await self._emit(BackendStatusValue.STARTING, "backend starting")
            try:
                await self._driver.start()
                healthy = await self._driver.health()
            except Exception as exc:  # noqa: BLE001
                self._status = BackendStatusValue.ERROR
                await self._emit(BackendStatusValue.ERROR, f"start failed: {exc}")
                raise
            if not healthy:
                self._status = BackendStatusValue.ERROR
                await self._emit(
                    BackendStatusValue.ERROR, "backend failed health check"
                )
                raise RuntimeError("backend failed health check after start")
            self._status = BackendStatusValue.RUNNING
            await self._emit(BackendStatusValue.RUNNING, "backend running")

    async def ensure_started(self) -> None:
        """Idempotent alias of `start()`."""
        await self.start()

    async def stop(self) -> None:
        """Any non-stopped state -> STOPPING -> STOPPED; failure -> ERROR.

        In-flight streams are not force-cancelled here; their producer
        tasks release their slots as the upstream closes. Use
        :meth:`stop_if_idle` when the caller must *not* stop under load.
        """
        async with self._lock:
            await self._stop_locked()

    async def stop_if_idle(self) -> None:
        """Atomically refuse to stop while a slot is held; otherwise stop.

        Raises ``BackendBusy`` if ``in_flight > 0`` or the status is
        ``IN_USE``. The busy check and the STOPPING transition happen
        under the same lock acquisition with **no intervening await**, so
        a concurrent ``acquire_slot()`` can never slip in between the
        check and the stop (which would SIGTERM the process under a live
        stream).
        """
        async with self._lock:
            if self._in_flight > 0 or self._status == BackendStatusValue.IN_USE:
                raise BackendBusy(
                    f"backend busy: {self._in_flight} request(s) in flight",
                    in_flight=self._in_flight,
                )
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Stop body assuming ``self._lock`` is already held.

        asyncio.Lock is not reentrant — callers must hold the lock and
        must not await anything between their busy check and here.
        """
        if self._status == BackendStatusValue.STOPPED:
            return
        self._status = BackendStatusValue.STOPPING
        await self._emit(BackendStatusValue.STOPPING, "backend stopping")
        try:
            await self._driver.stop()
        except Exception as exc:  # noqa: BLE001
            self._status = BackendStatusValue.ERROR
            await self._emit(BackendStatusValue.ERROR, f"stop failed: {exc}")
            raise
        self._status = BackendStatusValue.STOPPED
        await self._emit(BackendStatusValue.STOPPED, "backend stopped")

    # ------------------------------------------------------------------
    # Slot admission
    # ------------------------------------------------------------------
    async def acquire_slot(self) -> None:
        """Admit one inference request.

        Raises `BackendNotReady` unless RUNNING/IN_USE, `BackendBusy` at
        capacity. The first acquisition flips RUNNING -> IN_USE
        (emitted); records `last_request_at`.
        """
        async with self._lock:
            if self._status not in _SERVING_STATES:
                raise BackendNotReady(
                    f"backend is {self._status}; not accepting requests"
                )
            if self._in_flight >= self.capacity:
                raise BackendBusy(
                    f"backend busy: {self._in_flight}/{self.capacity} slots in use"
                )
            self._in_flight += 1
            self.last_request_at = datetime.now(UTC)
            if self._status == BackendStatusValue.RUNNING:
                self._status = BackendStatusValue.IN_USE
                await self._emit(BackendStatusValue.IN_USE, "backend in use")

    async def release_slot(self) -> None:
        """Release one inference slot.

        When `in_flight` returns to 0, flips IN_USE -> RUNNING
        (emitted). Tolerates extra releases (never goes below 0) and
        never resurrects a non-serving state.
        """
        async with self._lock:
            if self._in_flight <= 0:
                return
            self._in_flight -= 1
            if self._in_flight == 0 and self._status == BackendStatusValue.IN_USE:
                self._status = BackendStatusValue.RUNNING
                await self._emit(BackendStatusValue.RUNNING, "backend idle")

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Async context manager over acquire_slot/release_slot."""
        await self.acquire_slot()
        try:
            yield
        finally:
            await self.release_slot()

    # ------------------------------------------------------------------
    # Streaming with release-on-upstream-close
    # ------------------------------------------------------------------
    async def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Acquire a slot and return a slot-owned OpenResponses event stream.

        The slot is acquired eagerly so the caller can map
        `BackendBusy` / `BackendNotReady` to an HTTP status *before*
        the response starts. Release happens exactly once, in the
        producer task's `finally`, when the driver stream closes —
        independent of downstream consumption.
        """
        await self.acquire_slot()
        try:
            upstream = self._driver.stream_responses(request)
        except BaseException:
            await self.release_slot()
            raise
        return self._owned_stream(upstream)

    async def stream_chat_completions(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Chat-completions counterpart of `stream_responses`."""
        await self.acquire_slot()
        try:
            upstream = self._driver.stream_chat_completions(request)
        except BaseException:
            await self.release_slot()
            raise
        return self._owned_stream(upstream)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Slot-admitted, non-streaming embeddings call.

        The non-streaming counterpart of `stream_responses` /
        `stream_chat_completions`: a single awaited driver call returns a
        spec `CreateEmbeddingResponse` dict, so no pump/`_owned_stream`
        machinery is needed. The slot is acquired eagerly (so
        `BackendBusy` / `BackendNotReady` surface as real HTTP statuses
        before any bytes are committed) and released in a `finally` on
        every exit path, including a driver exception.
        """
        await self.acquire_slot()
        try:
            return await self._driver.embeddings(request)
        finally:
            # Deliberate simplification vs the streaming `_pump` shielded
            # release: a non-streaming call completes within this single
            # awaited driver call, so a plain `await release_slot()` here is
            # sufficient -- no lingering downstream consumer can hold the slot.
            await self.release_slot()

    # ------------------------------------------------------------------
    # Phase 24 (audio) — slot-admitted wrappers.
    # ------------------------------------------------------------------
    def _require_serving(self) -> None:
        """Raise BackendNotReady unless RUNNING/IN_USE, WITHOUT taking a slot.

        Voice catalog / enrollment ops (voices / voice_put / voice_delete)
        require a serving backend but are not inference, so they gate on the
        state directly instead of acquiring a slot.
        """
        if self._status not in _SERVING_STATES:
            raise BackendNotReady(f"backend is {self._status}; not accepting requests")

    async def speech(self, request: dict[str, Any]) -> SpeechStream:
        """Acquire a slot and return a slot-owned speech stream.

        The slot is acquired eagerly (so BackendBusy / BackendNotReady map to
        real HTTP statuses before any audio bytes are committed). The driver's
        ``speech()`` is called synchronously (a NotImplementedError from a
        provider without the surface releases the slot and re-raises -> 501);
        its byte chunks are then wrapped in the same pump as the SSE streams
        so the slot releases on upstream close, never on client disconnect.
        """
        await self.acquire_slot()
        try:
            stream = self._driver.speech(request)
        except BaseException:
            await self.release_slot()
            raise
        return SpeechStream(
            headers=dict(stream.headers), chunks=self._owned_stream(stream.chunks)
        )

    async def transcribe(
        self,
        form: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
    ) -> tuple[int, dict[str, str], bytes]:
        """Slot-admitted, non-streaming transcription relay.

        Mirrors :meth:`embeddings`: a single awaited driver call returns the
        upstream ``(status, headers, body)`` triple, so no pump is needed; the
        slot is acquired eagerly and released in a ``finally`` on every exit
        path (including a driver exception).
        """
        await self.acquire_slot()
        try:
            return await self._driver.transcribe(form, files)
        finally:
            await self.release_slot()

    async def transcribe_stream(self, websocket: WebSocket) -> None:
        """Slot-admitted live ASR WebSocket bridge.

        The slot is held for the full connection lifetime and released when the
        driver's bridge coroutine returns (on any disconnect). A provider
        without the surface raises NotImplementedError (the route closes the
        socket with 1011); BackendBusy / BackendNotReady surface before the
        bridge starts.
        """
        await self.acquire_slot()
        try:
            await self._driver.transcribe_stream(websocket)
        finally:
            await self.release_slot()

    async def voices(self) -> dict[str, Any]:
        """Voice catalog passthrough (no slot; requires a serving backend)."""
        self._require_serving()
        return await self._driver.voices()

    async def voice_put(
        self,
        name: str,
        wav: bytes,
        ref_text: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        """Saved-voice enrollment write (no slot; requires a serving backend)."""
        self._require_serving()
        return await self._driver.voice_put(name, wav, ref_text, language)

    async def voice_delete(self, name: str) -> dict[str, Any]:
        """Saved-voice removal (no slot; requires a serving backend)."""
        self._require_serving()
        return await self._driver.voice_delete(name)

    def _owned_stream(self, upstream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Return a consumer stream whose slot is owned by a pump task.

        The pump starts immediately, so the upstream is always drained and
        closed and the slot always released, even if the downstream
        consumer never iterates. If the consumer *does* iterate and then
        abandons the stream, closing the returned generator cancels the
        pump, which closes the upstream — so the release is still driven
        by the upstream generator's lifetime, never by the client's.

        Item-type agnostic: it drains SSE event dicts for chat/responses and
        raw ``bytes`` for the Phase 24 speech stream alike.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()
        pump = asyncio.create_task(self._pump(upstream, queue))

        async def consume() -> AsyncIterator[Any]:
            try:
                while True:
                    item = await queue.get()
                    if item is _STREAM_END:
                        return
                    if isinstance(item, _StreamFailure):
                        raise item.exc
                    yield item
            finally:
                if not pump.done():
                    pump.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await pump

        return consume()

    async def _pump(
        self, upstream: AsyncIterator[Any], queue: asyncio.Queue[Any]
    ) -> None:
        """Drain the upstream driver stream; own the slot release.

        The `finally` runs on normal exhaustion, upstream exception, or
        task cancellation — i.e., on upstream close — which is the
        release trigger, never the downstream client's lifecycle.

        Cancellation discipline (the production-observed leak this guards
        against): when this task is cancelled mid-iteration, EVERY `await`
        in the finally re-raises CancelledError until it actually completes
        — and `suppress(Exception)` does NOT catch CancelledError
        (BaseException since 3.8). An unsuppressed aclose here skipped both
        `_STREAM_END` and `release_slot()`, leaking the slot while the
        provider's health endpoint kept reporting `in_flight: 1` forever.

        So: shield every finally-await (shield keeps the child running to
        completion even when the parent is being cancelled), suppress
        BaseException around each (CancelledError included), and run the
        slot release LAST under its own shield. The final
        `asyncio.current_task().uncancel()` is the 3.11+ belt: a cancelling
        task's awaits would otherwise keep raising even after the shielded
        work is done.
        """
        try:
            async for event in upstream:
                await queue.put(event)
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(BaseException):  # noqa: SIM105 - noqa: BLE001
                await asyncio.shield(queue.put(_StreamFailure(exc)))
            raise
        finally:
            with contextlib.suppress(BaseException):
                await asyncio.shield(upstream.aclose())  # type: ignore[attr-defined]
            with contextlib.suppress(BaseException):
                await asyncio.shield(queue.put(_STREAM_END))
            try:
                asyncio.current_task().uncancel()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - pre-3.11 / no running task
                pass
            with contextlib.suppress(BaseException):
                await asyncio.shield(self.release_slot())
