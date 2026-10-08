"""Single env-port HTTP serving for a provider agent (port model overhaul).

An agent hosts 1..N backends but publishes exactly **one** admin-facing HTTP
port — its container env ``PROVIDER_PORT``. :class:`AgentServer` runs a single
``uvicorn`` server on that port whose app (built by
:func:`provider_lib.app_factory.create_provider_app`) resolves the target
backend per request from the live :class:`~provider_lib.registry.BackendRegistry`
(matching the request's ``model`` field against each backend's definition
``alias``). The control plane stays a single agent WebSocket; the data plane is
one multiplexed ``/v1`` surface.

Design notes:

* **One listener, no per-backend churn.** Because routing happens at request time
  against the shared registry, a backend added or dropped by an
  ``agent.assignments.update`` reconcile is immediately routable (or removed)
  without touching the socket — there is no per-backend listener to start/stop
  and no registry change-listener to sync.
* **Engine ports are private.** Each backend's engine (llama-server / gufo /
  halogen / halogen-flash) binds a private local port inside the container that
  the admin never learns or dials; the admin only ever reaches the agent's single
  ``PROVIDER_PORT``.
* **No signal capture.** ``uvicorn.Server.serve()`` installs process signal
  handlers (``capture_signals``); we drive the server through its ``startup`` /
  ``main_loop`` / ``shutdown`` phases directly (mirroring ``Server._serve`` minus
  the signal capture) so the process owns shutdown and cancelling the serve task
  tears the socket down cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import uvicorn

from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendRegistry

logger = logging.getLogger("provider.serve")


class _BackendListener:
    """One uvicorn server bound to a port.

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


class AgentServer:
    """Serve the agent's ONE admin-facing ``/v1`` surface on ``PROVIDER_PORT``.

    The agent publishes a single uvicorn server bound to its env ``PROVIDER_PORT``
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


__all__ = ["AgentServer"]
