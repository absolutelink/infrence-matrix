"""Multi-port HTTP serving for a provider agent (Phase 16 slice 6).

An agent hosts 1..N backends, each of which the admin dials on its own port
(``base_port + offset`` — the backend's assignment ``port``). This module runs
one ``uvicorn`` server per hosted backend in a single process, each serving that
backend's normalized ``/v1`` surface (built by
:func:`provider_lib.app_factory.create_provider_app`) bound to the backend's
port. The control plane stays a single agent WebSocket; only the data plane fans
out across ports.

Design notes:

* **Additive, not disruptive.** A single-backend agent produces exactly one
  server on ``PROVIDER_PORT`` — byte-for-byte the pre-slice-6 behavior.
* **Dynamic.** When an ``agent.assignments.update`` reconcile adds or drops a
  backend it fires the registry's change listeners; :meth:`MultiPortServer.sync`
  diffs the served ports against the hosted backends and starts/stops the
  affected listeners. Backends added after startup become reachable without a
  process restart.
* **No per-server signal capture.** ``uvicorn.Server.serve()`` installs process
  signal handlers (``capture_signals``); running N of them would fight over
  SIGINT/SIGTERM. We drive each server through its ``startup`` / ``main_loop`` /
  ``shutdown`` phases directly (mirroring ``Server._serve`` minus the signal
  capture) so the process owns shutdown and cancelling the serve task tears every
  listener down cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import uvicorn

from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendHandle, BackendRegistry

logger = logging.getLogger("provider.serve")


class _BackendListener:
    """One uvicorn server bound to one backend's port.

    Runs uvicorn's serve phases without installing signal handlers (the process
    owns shutdown); ``stop`` flips ``should_exit`` so ``main_loop`` returns and
    the socket is closed gracefully.
    """

    def __init__(self, app: Any, host: str, port: int, log_level: str) -> None:
        self.port = port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level=log_level)
        )
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"serve:{self.port}")

    async def _run(self) -> None:
        server = self._server
        config = server.config
        if not config.loaded:
            config.load()
        server.lifespan = config.lifespan_class(config)
        await server.startup()
        try:
            if not server.should_exit:
                await server.main_loop()
        finally:
            if server.started:
                await server.shutdown()

    async def stop(self) -> None:
        self._server.should_exit = True
        task = self._task
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._task = None


class MultiPortServer:
    """Serve each hosted backend's ``/v1`` on its own port, in one process."""

    def __init__(
        self,
        settings: ProviderSettings,
        provider_type: str,
        version: str,
        registry: BackendRegistry,
        *,
        calculate_usage: Any | None = None,
        host: str = "0.0.0.0",
        log_level: str = "info",
    ) -> None:
        self._settings = settings
        self._provider_type = provider_type
        self._version = version
        self._registry = registry
        self._calculate_usage = calculate_usage
        self._host = host
        self._log_level = log_level
        # port -> listener. Keyed by port because the admin addresses a backend
        # by port; two hosted backends never share a port (admin port_conflict).
        self._listeners: dict[int, _BackendListener] = {}
        self._lock = asyncio.Lock()

    def _build_app(self, handle: BackendHandle) -> Any:
        overrides = BackendOverrides(
            provider_type=self._provider_type,
            version=self._version,
            calculate_usage=self._calculate_usage,
            lifecycle=handle.lifecycle,
        )
        return create_provider_app(self._settings, overrides)

    def _desired(self) -> dict[int, BackendHandle]:
        """Map each hosted backend's port -> handle (single-backend handles
        without an explicit port fall back to the agent base port)."""
        desired: dict[int, BackendHandle] = {}
        for handle in self._registry.handles():
            port = (
                handle.port if handle.port is not None else self._settings.PROVIDER_PORT
            )
            desired[int(port)] = handle
        return desired

    async def sync(self) -> None:
        """Reconcile served ports with the hosted backends (idempotent)."""
        async with self._lock:
            desired = self._desired()
            # Stop listeners whose backend is no longer hosted.
            for port in list(self._listeners):
                if port not in desired:
                    await self._listeners.pop(port).stop()
                    logger.info("stopped backend listener on port %s", port)
            # Start listeners for newly hosted backends. A single backend whose
            # app fails to build must NOT strand the others (M1): the exception
            # would otherwise propagate out of sync() (swallowed by the registry
            # change-listener), leaving every later backend unbound until the
            # next reconcile. Log-and-continue so the healthy backends still
            # bind; the broken one is retried on the next sync.
            for port, handle in desired.items():
                if port in self._listeners:
                    continue
                try:
                    listener = _BackendListener(
                        self._build_app(handle), self._host, port, self._log_level
                    )
                    listener.start()
                except Exception:  # noqa: BLE001 - one bad backend must not abort sync
                    logger.exception(
                        "failed to start backend listener on port %s (instance %s); "
                        "continuing with the remaining backends",
                        port,
                        handle.instance_id or "?",
                    )
                    continue
                self._listeners[port] = listener
                logger.info(
                    "serving backend %s on port %s", handle.instance_id or "?", port
                )

    async def aclose(self) -> None:
        """Stop every listener (process shutdown)."""
        async with self._lock:
            for port in list(self._listeners):
                await self._listeners.pop(port).stop()

    async def serve_forever(self) -> None:
        """Bind every hosted backend's port, keep them in sync with the
        registry, and block until the task is cancelled."""
        self._registry.add_change_listener(self.sync)
        await self.sync()
        try:
            # Block until cancelled by the caller (Ctrl-C / task cancel). The
            # never-set event is the idiomatic "wait forever".
            await asyncio.Event().wait()
        finally:
            await self.aclose()


class AgentServer:
    """Serve the agent's ONE admin-facing ``/v1`` surface on ``PROVIDER_PORT``.

    Port model overhaul: instead of one listener per backend (``MultiPortServer``)
    the agent publishes a single uvicorn server bound to its env ``PROVIDER_PORT``
    whose app resolves the target backend per request from the live registry (by
    the request's ``model`` = definition alias). Because resolution happens at
    request time against the shared registry, there is NO per-backend listener
    churn and NO registry change-listener sync — a backend added by an
    ``agent.assignments.update`` push is immediately routable on the same port.

    The uvicorn server is driven through its startup/main_loop/shutdown phases
    (via :class:`_BackendListener`) so no process signal handlers are installed —
    the owning process handles shutdown; cancelling :meth:`serve_forever` (or
    calling :meth:`aclose`) tears the socket down cleanly.
    """

    def __init__(
        self,
        settings: ProviderSettings,
        provider_type: str,
        version: str,
        registry: BackendRegistry,
        *,
        calculate_usage: Any | None = None,
        host: str = "0.0.0.0",
        log_level: str = "info",
    ) -> None:
        self._settings = settings
        self._provider_type = provider_type
        self._version = version
        self._registry = registry
        self._calculate_usage = calculate_usage
        self._host = host
        self._log_level = log_level
        self._listener: _BackendListener | None = None

    def _build_app(self) -> Any:
        overrides = BackendOverrides(
            provider_type=self._provider_type,
            version=self._version,
            calculate_usage=self._calculate_usage,
            registry=self._registry,
        )
        return create_provider_app(self._settings, overrides)

    @property
    def port(self) -> int:
        return int(self._settings.PROVIDER_PORT)

    async def serve_forever(self) -> None:
        """Bind the single env-port listener and block until cancelled."""
        # Build + start into locals first so a startup failure never leaves a
        # half-initialized listener behind; log and re-raise cleanly instead.
        try:
            listener = _BackendListener(
                self._build_app(), self._host, self.port, self._log_level
            )
            listener.start()
        except Exception:
            logger.exception("agent /v1 surface failed to start on port %s", self.port)
            self._listener = None
            raise
        self._listener = listener
        logger.info("agent /v1 surface serving on port %s", self.port)
        try:
            # The never-set event is the idiomatic "wait forever"; the caller
            # cancels this task to stop serving.
            await asyncio.Event().wait()
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Stop the listener (process shutdown)."""
        if self._listener is not None:
            await self._listener.stop()
            self._listener = None


__all__ = ["MultiPortServer", "AgentServer"]
