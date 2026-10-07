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

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from provider_lib.wire import BackendStatusValue

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


class BackendDriver(ABC):
    """Minimal per-type backend interface a provider package implements.

    `stream_responses` / `stream_chat_completions` are synchronous
    methods returning async iterators (typically async generator
    functions) so the lifecycle can start draining them inside a
    background producer task and observe their close semantics.
    """

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
    ) -> None:
        self._driver = driver
        self.capacity = max(1, capacity)
        self._status_callback = status_callback
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

    def _owned_stream(
        self, upstream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Return a consumer stream whose slot is owned by a pump task.

        The pump starts immediately, so the upstream is always drained and
        closed and the slot always released, even if the downstream
        consumer never iterates. If the consumer *does* iterate and then
        abandons the stream, closing the returned generator cancels the
        pump, which closes the upstream — so the release is still driven
        by the upstream generator's lifetime, never by the client's.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()
        pump = asyncio.create_task(self._pump(upstream, queue))

        async def consume() -> AsyncIterator[dict[str, Any]]:
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
        self, upstream: AsyncIterator[dict[str, Any]], queue: asyncio.Queue[Any]
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
