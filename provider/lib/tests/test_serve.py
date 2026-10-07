"""Phase 16 slice 6: MultiPortServer runs one uvicorn listener per hosted
backend, keyed by the backend's assignment ``port``, and reconciles dynamically
when the registry changes.

The per-backend app is stubbed (``_build_app``) so this exercises the *serving
bookkeeping* — start/stop listeners on sync, wire the registry change listener,
fall back to the base port when a handle has no explicit port — without needing
a real driver or binding many production ports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.serve import MultiPortServer


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


def _stub_build_app(server: MultiPortServer) -> None:
    # Replace the real create_provider_app path with a trivial ASGI callable so
    # the listener binds a bare uvicorn server (no driver / lifespan needed).
    server._build_app = lambda handle: _noop_app  # type: ignore[method-assign]


def _handle(iid: str, port: int | None) -> BackendHandle:
    return BackendHandle(iid, lifecycle=None, config_state=None, port=port)  # type: ignore[arg-type]


async def test_sync_starts_one_listener_per_port(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = BackendRegistry()
    registry.add(_handle("i1", 8391))
    registry.add(_handle("i2", 8392))
    server = MultiPortServer(settings, "mock", "dev", registry)
    _stub_build_app(server)
    await server.sync()
    assert set(server._listeners) == {8391, 8392}
    await server.aclose()
    assert server._listeners == {}


async def test_handle_without_port_falls_back_to_base(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = BackendRegistry()
    registry.add(_handle("solo", None))
    server = MultiPortServer(settings, "mock", "dev", registry)
    _stub_build_app(server)
    await server.sync()
    # Single-backend convenience: the agent base port is served.
    assert set(server._listeners) == {settings.PROVIDER_PORT}
    await server.aclose()


async def test_sync_reconciles_add_and_remove(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = BackendRegistry()
    registry.add(_handle("i1", 8391))
    registry.add(_handle("i2", 8392))
    server = MultiPortServer(settings, "mock", "dev", registry)
    _stub_build_app(server)
    await server.sync()
    assert set(server._listeners) == {8391, 8392}

    # Drop i1, add i3 -> i1's listener stops, i3's starts, i2 untouched.
    registry.remove("i1")
    registry.add(_handle("i3", 8393))
    await server.sync()
    assert set(server._listeners) == {8392, 8393}
    await server.aclose()


async def test_serve_forever_registers_change_listener(tmp_path: Path) -> None:
    import asyncio

    settings = _settings(tmp_path)
    registry = BackendRegistry()
    registry.add(_handle("i1", 8391))
    server = MultiPortServer(settings, "mock", "dev", registry)
    _stub_build_app(server)

    task = asyncio.create_task(server.serve_forever())
    # serve_forever syncs once, then blocks; a registry change triggers a sync.
    while not server._listeners:
        await asyncio.sleep(0.01)
    assert set(server._listeners) == {8391}

    registry.add(_handle("i2", 8392))
    await registry.notify_changed()
    for _ in range(100):
        if 8392 in server._listeners:
            break
        await asyncio.sleep(0.01)
    assert set(server._listeners) == {8391, 8392}

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_sync_one_bad_backend_does_not_strand_others(tmp_path: Path) -> None:
    """M1: if one backend's app fails to build, sync must log-and-continue so the
    other backends still get their listeners bound (the exception must not
    propagate out of sync and abort the whole reconcile)."""
    settings = _settings(tmp_path)
    registry = BackendRegistry()
    registry.add(_handle("good1", 8391))
    registry.add(_handle("bad", 8392))
    registry.add(_handle("good2", 8393))
    server = MultiPortServer(settings, "mock", "dev", registry)

    def _build(handle: BackendHandle) -> Any:
        if handle.instance_id == "bad":
            raise RuntimeError("boom: app build failed")
        return _noop_app

    server._build_app = _build  # type: ignore[method-assign]
    await server.sync()
    # The two healthy backends bound despite the middle one raising.
    assert set(server._listeners) == {8391, 8393}
    await server.aclose()
    assert server._listeners == {}
