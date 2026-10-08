"""Port model overhaul: ``AgentServer`` runs ONE uvicorn listener on the agent's
env ``PROVIDER_PORT``, regardless of how many backends the registry hosts.

The per-backend app is stubbed (``_build_app``) so this exercises the *serving
bookkeeping* — a single listener bound to the env port, clean teardown on
cancel/aclose, and a failed startup leaving no half-initialized listener —
without needing a real driver. Request-time routing to the correct backend
against the live registry is covered by ``test_v1_endpoints.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.serve import AgentServer


def _settings(tmp_path: Path) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="serve-test",
        MACHINE_SECRET="tok",
        AGENT_ID="serve-agent",
        ADMIN_BASE_URL="http://localhost:9999",
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
        PROVIDER_PORT=8300,
    )


async def _noop_app(scope: Any, receive: Any, send: Any) -> None:  # pragma: no cover
    pass


def _handle(iid: str, alias: str | None) -> BackendHandle:
    return BackendHandle(iid, lifecycle=None, config_state=None, alias=alias)  # type: ignore[arg-type]


async def test_agent_server_binds_single_env_port(tmp_path: Path) -> None:
    """Regardless of how many backends the registry hosts, AgentServer runs a
    single uvicorn listener on ``settings.PROVIDER_PORT`` (the agent routes by
    model at request time — no per-backend listener churn)."""
    settings = _settings(tmp_path)  # PROVIDER_PORT = 8300
    registry = BackendRegistry()
    registry.add(_handle("i1", "alpha"))
    registry.add(_handle("i2", "beta"))
    server = AgentServer(settings, "mock", "dev", registry)
    server._build_app = lambda: _noop_app  # type: ignore[method-assign]

    task = asyncio.create_task(server.serve_forever())
    while server._listener is None:
        await asyncio.sleep(0.01)
    # Exactly one listener, on the env port — not one per backend.
    assert server._listener.port == settings.PROVIDER_PORT == 8300

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    # aclose() (run in serve_forever's finally) tears the listener down.
    assert server._listener is None


async def test_agent_server_empty_registry_still_binds_one_port(tmp_path: Path) -> None:
    """An agent with no backends hosted yet still publishes its single /v1 port
    (requests 404 until a backend is assigned; the socket is always up)."""
    settings = _settings(tmp_path)
    registry = BackendRegistry()
    assert len(registry) == 0
    server = AgentServer(settings, "mock", "dev", registry)
    server._build_app = lambda: _noop_app  # type: ignore[method-assign]

    task = asyncio.create_task(server.serve_forever())
    while server._listener is None:
        await asyncio.sleep(0.01)
    assert server._listener.port == settings.PROVIDER_PORT

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert server._listener is None


async def test_agent_server_add_backend_no_listener_change(tmp_path: Path) -> None:
    """Adding a backend to the registry after startup does NOT create a second
    listener — the single env-port surface resolves it at request time."""
    settings = _settings(tmp_path)
    registry = BackendRegistry()
    registry.add(_handle("i1", "alpha"))
    server = AgentServer(settings, "mock", "dev", registry)
    server._build_app = lambda: _noop_app  # type: ignore[method-assign]

    task = asyncio.create_task(server.serve_forever())
    while server._listener is None:
        await asyncio.sleep(0.01)
    first = server._listener

    # A reconcile adds a backend; the listener is untouched (same object, same
    # single port) — no sync, no new socket.
    registry.add(_handle("i2", "beta"))
    await asyncio.sleep(0.05)
    assert server._listener is first
    assert server._listener.port == settings.PROVIDER_PORT

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_agent_server_failed_startup_leaves_no_listener(tmp_path: Path) -> None:
    """If the app fails to build, serve_forever logs and re-raises without
    leaving a half-initialized listener behind."""
    settings = _settings(tmp_path)
    registry = BackendRegistry()
    registry.add(_handle("i1", "alpha"))
    server = AgentServer(settings, "mock", "dev", registry)

    def _boom() -> Any:
        raise RuntimeError("app build failed")

    server._build_app = _boom  # type: ignore[method-assign]

    try:
        await server.serve_forever()
    except RuntimeError:
        pass
    else:  # pragma: no cover - serve_forever must re-raise
        raise AssertionError("serve_forever should have re-raised")
    assert server._listener is None


async def test_agent_server_aclose_is_idempotent(tmp_path: Path) -> None:
    """aclose() before/after serve_forever never raises (no listener to stop)."""
    settings = _settings(tmp_path)
    registry = BackendRegistry()
    server = AgentServer(settings, "mock", "dev", registry)
    server._build_app = lambda: _noop_app  # type: ignore[method-assign]
    # No listener yet.
    await server.aclose()
    assert server._listener is None

    task = asyncio.create_task(server.serve_forever())
    while server._listener is None:
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    # Already torn down by serve_forever's finally; a second aclose is a no-op.
    await server.aclose()
    assert server._listener is None
