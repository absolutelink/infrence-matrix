"""Mock provider package.

A hardware-free provider instance used for local development and to exercise
the full admin -> provider registration + WebSocket path (Phase 3) and the
provider backend lifecycle + /v1 translation surface (Phase 4).

Startup sequence (see ``run_async``):
  1. Register with the admin (POST /admin/api/providers/register) using
     MACHINE_UID + MACHINE_SECRET + AGENT_ID from the environment.
  2. Build a `BackendLifecycle` around a `MockBackend`, with capacity
     from the registration response (`provider_definition.capacity`).
  3. Install command handlers (backend.start / backend.stop) that drive
     the shared lifecycle and emit backend.status via the admin client.
  4. Dial the admin WebSocket and emit an initial provider.status event.
  5. Serve the provider app (/health + /v1/*) with uvicorn. The /v1
     surface uses the SAME lifecycle instance as the command handlers.

Boot is admin-driven: after connect the backend stays STOPPED until the
admin sends `backend.start`.

The registration + connect logic is factored into async helpers so an
integration test can drive it programmatically without uvicorn.
"""

import asyncio
import logging
from typing import Any

from provider_lib.admin_client import AdminClient, RegistrationResult
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState, install_config_handlers
from provider_lib.log_stream import install_log_streaming
from provider_lib.ops import install_backend_ops
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.serve import MultiPortServer
from provider_lib.wire import InstanceStatusValue

from provider_mock.backend import SCHEMA, MockBackend

logger = logging.getLogger("provider.mock")

PROVIDER_TYPE = "mock"
VERSION = "dev"
DEFAULT_CAPACITY = 1
DEFAULT_MODEL = "mock-model"

FAKE_HARDWARE: dict[str, Any] = {
    "gpus": [
        {
            "uuid": "mock-gpu-1",
            "vendor": "mock",
            "name": "Mock GPU",
            "total_vram_bytes": 24 * 1024**3,
        }
    ],
    "total_vram_bytes": 24 * 1024**3,
    "cpu": {"cores": 8, "model": "mock-cpu"},
    "ram": {"total_bytes": 32 * 1024**3},
}


def make_lifecycle(
    client: AdminClient, *, instance_id: str | None = None, **stream_kwargs: Any
) -> BackendLifecycle:
    """Build a mock lifecycle with status events routed to the admin.

    Phase 16: ``backend.status`` is per-backend and must carry the owning
    ``instance_id``. When built from a registration backend the id is passed
    in directly; otherwise it is read lazily from the registration (single-
    backend convenience). ``provider.status`` is agent-level.
    """

    def _iid() -> str | None:
        if instance_id is not None:
            return instance_id
        reg = client.registration if client.is_registered else None
        return reg.instance_id if reg is not None else None

    async def emit(status: str, reason: str | None) -> None:
        payload: dict[str, Any] = {
            "backend_status": status,
            "agent_status": InstanceStatusValue.RUNNING,
            "reason": reason,
        }
        iid = _iid()
        if iid is not None:
            payload["instance_id"] = iid
        await client.send_event("backend.status", payload)

    driver = MockBackend(model=DEFAULT_MODEL, **stream_kwargs)
    return BackendLifecycle(
        driver, capacity=DEFAULT_CAPACITY, status_callback=emit, instance_id=_iid()
    )


def build_registry(client: AdminClient, result: RegistrationResult) -> BackendRegistry:
    """Host one lifecycle per backend the admin placed on this agent (H3).

    A single-backend registration yields a one-handle registry (transparent
    pass-through dispatch); a multi-backend registration yields N handles so
    per-backend commands route to the correct lifecycle by ``instance_id``.
    """
    registry = BackendRegistry()
    for backend in result.backends:
        iid = backend.get("instance_id")
        if iid is None:  # pragma: no cover - admin always assigns an id
            continue
        port = backend.get("port")
        serve_port = port if isinstance(port, int) else None
        lifecycle = make_lifecycle(client, instance_id=iid)
        config_state = ConfigState()
        _apply_backend(lifecycle, backend, config_state)
        registry.add(BackendHandle(str(iid), lifecycle, config_state, port=serve_port))
    return registry


def install_command_handlers(
    client: AdminClient,
    target: BackendLifecycle | BackendRegistry,
    config_state: ConfigState | None = None,
) -> None:
    """Register admin->provider command handlers on the client.

    ``target`` is a single lifecycle (single-backend) or a
    :class:`BackendRegistry` (multi-backend). Per-backend commands are routed
    to the correct lifecycle by ``instance_id`` (H3).
    """
    resolved_state = config_state if config_state is not None else ConfigState()

    async def re_register() -> dict[str, Any] | None:
        """Re-run the registration handshake and adopt its response.

        Called by `provider.initialize` (provider_lib.ops): a fresh
        `/register` re-checks the version + schema gates, re-adopts the
        definition(s), rewrites `provider_config.json` and mints a new agent
        secret for future reconnects. The live socket stays up — it is what
        carries the status events the operator is watching.
        """
        result = await client.register(
            provider_type=PROVIDER_TYPE,
            version=VERSION,
            base_port=client.settings.PROVIDER_PORT,
            hardware=FAKE_HARDWARE,
            schema=SCHEMA,
        )
        if isinstance(target, BackendRegistry):
            by_id = {h.instance_id: h for h in target.handles()}
            for backend in result.backends:
                handle = by_id.get(str(backend.get("instance_id")))
                if handle is not None:
                    _apply_backend(handle.lifecycle, backend, handle.config_state)
        else:
            apply_registration(target, result, resolved_state)
        return result.provider_definition

    # backend.start / stop / restart + provider.initialize come from
    # provider_lib.ops, shared across providers: an operator-driven boot
    # runs in the background (`wait_for_running: false`) so a slow engine
    # never blocks the admin's request.
    #
    # Phase 16 slice 5: a multi-backend agent supplies a `make_handle` factory
    # so an `agent.assignments.update` push can spawn a fresh lifecycle for a
    # newly-placed backend (and retire one that was dropped). A single-
    # lifecycle agent passes none — the shared handler then refuses adds it
    # cannot host rather than crashing.
    make_handle = None
    if isinstance(target, BackendRegistry):

        def make_handle(entry: dict[str, Any]) -> BackendHandle:
            iid = str(entry["instance_id"])
            port = entry.get("port")
            serve_port = port if isinstance(port, int) else None
            lifecycle = make_lifecycle(client, instance_id=iid)
            config_state = ConfigState()
            _apply_assignment(lifecycle, entry, config_state)
            return BackendHandle(iid, lifecycle, config_state, port=serve_port)

    install_backend_ops(
        client,
        target,
        backend_name=PROVIDER_TYPE,
        re_register=re_register,
        make_handle=make_handle,
    )
    install_config_handlers(client, target, resolved_state, client.settings)


def _apply_backend(
    lifecycle: BackendLifecycle,
    backend: dict[str, Any],
    config_state: ConfigState | None = None,
) -> None:
    """Adopt capacity + model alias + config from ONE backend dict."""
    definition = backend.get("definition") or {}
    iid = backend.get("instance_id")
    if iid is not None:
        # H1/H2: key this lifecycle to its backend id.
        lifecycle.instance_id = str(iid)
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    alias = definition.get("alias")
    if isinstance(alias, str) and alias and isinstance(lifecycle.driver, MockBackend):
        lifecycle.driver.model = alias
    backend_config = definition.get("backend_config")
    if isinstance(backend_config, dict):
        lifecycle.driver.apply_config(backend_config)
    if config_state is not None:
        fp = definition.get("config_fingerprint")
        config_state.applied_fingerprint = fp if isinstance(fp, str) else None


def _apply_assignment(
    lifecycle: BackendLifecycle,
    entry: dict[str, Any],
    config_state: ConfigState,
) -> None:
    """Adopt an ``agent.assignments.update`` entry (flat shape) into a freshly
    built lifecycle: instance id, capacity, model alias, backend_config, and
    the applied fingerprint (slice 5)."""
    iid = entry.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = entry.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    alias = entry.get("alias")
    if isinstance(alias, str) and alias and isinstance(lifecycle.driver, MockBackend):
        lifecycle.driver.model = alias
    backend_config = entry.get("backend_config")
    if isinstance(backend_config, dict):
        lifecycle.driver.apply_config(backend_config)
    fp = entry.get("config_fingerprint")
    config_state.applied_fingerprint = fp if isinstance(fp, str) else None


def apply_registration(
    lifecycle: BackendLifecycle,
    result: RegistrationResult,
    config_state: ConfigState | None = None,
) -> None:
    """Adopt capacity and model alias from the (first) backend of the
    registration response — single-backend convenience."""
    backend = result.backends[0] if result.backends else {"definition": {}}
    _apply_backend(lifecycle, backend, config_state)


async def emit_provider_status(
    client: AdminClient, lifecycle: BackendLifecycle
) -> None:
    """Announce current agent/backend state to the admin (Phase 16:
    ``provider.status`` is agent-level; ``backend.status`` carries the
    per-backend ``instance_id`` and is emitted by the lifecycle callback)."""
    await client.send_event(
        "provider.status",
        {
            "agent_status": InstanceStatusValue.RUNNING,
            "backend_status": lifecycle.backend_status,
            "provider_type": PROVIDER_TYPE,
            "version": VERSION,
        },
    )


async def bootstrap_agent(
    client: AdminClient,
) -> tuple[RegistrationResult, BackendRegistry, BackendLifecycle]:
    """Register over HTTP and host one lifecycle per placed backend (slice 6).

    Returns ``(result, registry, primary)``; the primary (first) backend drives
    agent-level log streaming. Does NOT dial the WS.
    """
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        base_port=client.settings.PROVIDER_PORT,
        hardware=FAKE_HARDWARE,
        # Single shared load (provider_mock.backend.SCHEMA) so the schema
        # registered with the admin and the schema the driver validates
        # against are provably the same object.
        schema=SCHEMA,
    )
    registry = build_registry(client, result)
    install_command_handlers(client, registry)
    primary = (
        registry.handles()[0].lifecycle if len(registry) else make_lifecycle(client)
    )
    return result, registry, primary


async def register_provider(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> tuple[RegistrationResult, BackendLifecycle]:
    """Install command handlers, register over HTTP, adopt the response.

    When ``lifecycle`` is supplied (single-backend tests) it is used directly.
    Otherwise the agent hosts one lifecycle per placed backend (H3): the
    registration response's ``backends`` drive a :class:`BackendRegistry`, and
    the first backend's lifecycle is returned as the primary for the caller's
    log-streaming / app-serving wiring.

    Does NOT dial the WS — see AdminClient.run_forever for the persistent
    connection with reconnect/backoff.
    """
    if lifecycle is not None:
        config_state = ConfigState()
        install_command_handlers(client, lifecycle, config_state)
        result = await client.register(
            provider_type=PROVIDER_TYPE,
            version=VERSION,
            base_port=client.settings.PROVIDER_PORT,
            hardware=FAKE_HARDWARE,
            schema=SCHEMA,
        )
        apply_registration(lifecycle, result, config_state)
        return result, lifecycle

    _result, _registry, primary = await bootstrap_agent(client)
    return _result, primary


async def register_and_connect(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> RegistrationResult:
    """Register with the admin, adopt its config, dial the WS once, emit status.

    One-shot connect for tests and simple callers. Production entrypoints
    (run_async) split this into register_provider + run_forever so the
    connection survives admin restarts (ARCHITECTURE.md §5).
    """
    result, lifecycle = await register_provider(client, lifecycle)
    await client.connect()
    await emit_provider_status(client, lifecycle)
    return result


def build_app(
    lifecycle: BackendLifecycle | None = None,
    settings: ProviderSettings | None = None,
):
    settings = settings or ProviderSettings()
    overrides = BackendOverrides(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        lifecycle=lifecycle,
    )
    return create_provider_app(settings, overrides)


async def run_async() -> None:
    settings = ProviderSettings()
    client = AdminClient(settings)
    # Host one lifecycle per placed backend (slice 6); `lifecycle` is the
    # primary (first) backend used for agent-level log streaming.
    _result, registry, lifecycle = await bootstrap_agent(client)
    # Phase 13: provider.logs streaming + backend.logs.get handler (the
    # mock driver has no subprocess ring, so only provider.logs flows).
    log_bundle = install_log_streaming(client, lifecycle)

    # Keep the admin WS alive for the process lifetime alongside the per-backend
    # HTTP servers: run_forever dials in, re-emits provider.status on every
    # (re)connect, and reconnects with exponential backoff if the admin restarts
    # (ARCHITECTURE.md §5).
    first_connect = asyncio.Event()

    async def on_connected() -> None:
        log_bundle.start()
        await emit_provider_status(client, lifecycle)
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "mock agent connected: backends=%d epoch=%s",
                len(registry),
                client.epoch,
            )

    ws_task = asyncio.create_task(
        client.run_forever(
            on_connected=on_connected,
            on_disconnected=log_bundle.stop,
        )
    )
    # Slice 6: serve each hosted backend's /v1 on its own port (base_port +
    # offset). A single-backend agent yields exactly one listener on
    # PROVIDER_PORT — identical to the pre-slice-6 behavior.
    server = MultiPortServer(settings, PROVIDER_TYPE, VERSION, registry)
    serve_task = asyncio.create_task(server.serve_forever())
    try:
        await serve_task
    except asyncio.CancelledError:
        # serve_task only ends via cancellation: uvicorn signal capture is
        # disabled in provider_lib.serve, so SIGINT propagates out of
        # asyncio.run and is handled in main(), not here.
        pass
    finally:
        ws_task.cancel()
        await asyncio.gather(ws_task, return_exceptions=True)
        await server.aclose()
        await log_bundle.stop()
        await client.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
