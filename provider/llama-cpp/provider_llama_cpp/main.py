"""llama-cpp provider package entrypoint.

Wires `LlamaCppBackend` into the generic provider machinery following the
mock provider's pattern:

  1. Register with the admin (hardware report from provider_lib.metrics).
  2. Build a BackendLifecycle around the driver; status transitions are
     emitted as `backend.status` WS events.
  3. Command handlers: backend.start / backend.stop drive the shared
     lifecycle; metrics.assign / metrics.unassign start/stop the
     machine-level metrics emitter.
  4. Serve the provider /v1 surface (built by provider_lib) on
     PROVIDER_PORT; the backend_port defaults to PROVIDER_PORT + 1.

Boot is admin-driven: after connect the backend stays STOPPED until the
admin sends `backend.start`. The backend_config comes from the
registration response (`provider_definition.backend_config`).
"""

import asyncio
import logging
from typing import Any

from provider_lib.admin_client import AdminClient, RegistrationResult
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState, install_config_handlers
from provider_lib.log_stream import LogStreamingBundle, install_log_streaming
from provider_lib.metrics import (
    MachineMetricsEmitter,
    collect_machine_snapshot,
    gpu_uuid,
    parse_gpu_assignment,
)
from provider_lib.ops import install_backend_ops
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.serve import MultiPortServer
from provider_lib.wire import Frame, InstanceStatusValue

from provider_llama_cpp.driver import SCHEMA, LlamaCppBackend

logger = logging.getLogger("provider.llama_cpp.main")

PROVIDER_TYPE = "llama-cpp"
VERSION = "dev"
DEFAULT_CAPACITY = 1


async def build_hardware_report(settings: ProviderSettings) -> dict[str, Any]:
    """Best-effort hardware inventory for registration.

    Uses the metrics collectors; whatever the host cannot report is
    simply omitted (empty lists / zeros are acceptable). Phase 17: the GPU
    sample is filtered to ``ASSIGNED_GPU_UUIDS`` (empty = every GPU the
    container sees) so a device-isolated agent reports only its own GPU and
    its own VRAM sum.
    """
    snapshot = await collect_machine_snapshot(
        {"vram", "gpu_usage", "os_ram", "cpu", "storage"},
        models_dir=settings.MODELS_DIR,
        cache_dir=settings.CACHE_DIR,
        gpu_assignment=parse_gpu_assignment(settings.assigned_gpu_tokens),
    )
    vram = snapshot.get("vram", {})
    return {
        "gpus": [
            {
                "uuid": gpu_uuid(g),
                "vendor": g.get("vendor", "unknown"),
                "name": g.get("name", "Unknown GPU"),
                "total_vram_bytes": g.get("vram_total", 0),
            }
            for g in vram.get("gpus", [])
        ],
        "total_vram_bytes": vram.get("total_bytes", 0),
        "cpu": snapshot.get("cpu", {}),
        "ram": snapshot.get("os_ram", {}),
    }


def make_driver(
    settings: ProviderSettings,
    backend_config: dict[str, Any],
    client: AdminClient | None = None,
    serve_port: int | None = None,
) -> LlamaCppBackend:
    """Build the driver with progress/log events routed to the admin.

    ``serve_port`` (slice 6) is this backend's admin-facing port; the engine
    subprocess binds to ``serve_port + 1`` so multiple backends on one agent
    never collide. ``None`` keeps the single-backend default (PROVIDER_PORT+1).
    """

    async def on_progress(payload: dict[str, Any]) -> None:
        if client is not None:
            await client.send_event("download.progress", payload)

    return LlamaCppBackend(
        settings, backend_config, progress_cb=on_progress, serve_port=serve_port
    )


def make_lifecycle(
    client: AdminClient,
    backend_config: dict[str, Any] | None = None,
    *,
    instance_id: str | None = None,
    serve_port: int | None = None,
):
    """Build the llama-cpp lifecycle with status events to the admin.

    Phase 16 slice 6: an agent may host N backends, so ``backend.status`` is
    keyed to THIS lifecycle's ``instance_id`` (falling back to the registration
    first-backend id for the single-backend path). ``serve_port`` threads the
    backend's assignment port to the driver.
    """
    settings = client.settings

    def _iid() -> str | None:
        if instance_id is not None:
            return instance_id
        if getattr(client, "is_registered", False):
            return client.registration.instance_id
        return None

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

    driver = make_driver(
        settings, backend_config or {}, client=client, serve_port=serve_port
    )
    return BackendLifecycle(
        driver, capacity=DEFAULT_CAPACITY, status_callback=emit, instance_id=_iid()
    )


def _apply_backend(
    lifecycle: BackendLifecycle,
    backend: dict[str, Any],
    config_state: ConfigState | None = None,
) -> None:
    """Adopt capacity + backend_config + fingerprint from ONE registration
    backend dict into its lifecycle/driver."""
    definition = backend.get("definition") or {}
    iid = backend.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, LlamaCppBackend):
        backend_config = definition.get("backend_config")
        if isinstance(backend_config, dict):
            driver.apply_config(backend_config)
    if config_state is not None:
        fp = definition.get("config_fingerprint")
        config_state.applied_fingerprint = fp if isinstance(fp, str) else None


def _apply_assignment(
    lifecycle: BackendLifecycle,
    entry: dict[str, Any],
    config_state: ConfigState,
) -> None:
    """Adopt an ``agent.assignments.update`` entry (flat shape) into a freshly
    built lifecycle: instance id, capacity, backend_config, applied fp."""
    iid = entry.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = entry.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, LlamaCppBackend):
        backend_config = entry.get("backend_config")
        if isinstance(backend_config, dict):
            driver.apply_config(backend_config)
    fp = entry.get("config_fingerprint")
    config_state.applied_fingerprint = fp if isinstance(fp, str) else None


def build_registry(client: AdminClient, result: RegistrationResult) -> BackendRegistry:
    """Host one drivable lifecycle per backend the admin placed on this agent.

    A single-backend registration yields a one-handle registry (transparent
    pass-through dispatch); a multi-backend registration yields N handles so
    per-backend commands route to the correct lifecycle by ``instance_id`` and
    each backend's ``/v1`` is served on its own port (slice 6).
    """
    registry = BackendRegistry()
    for backend in result.backends:
        iid = backend.get("instance_id")
        if iid is None:  # pragma: no cover - admin always assigns an id
            continue
        port = backend.get("port")
        definition = backend.get("definition") or {}
        backend_config = definition.get("backend_config") or {}
        lifecycle = make_lifecycle(
            client,
            backend_config,
            instance_id=str(iid),
            serve_port=port if isinstance(port, int) else None,
        )
        config_state = ConfigState()
        _apply_backend(lifecycle, backend, config_state)
        registry.add(
            BackendHandle(
                str(iid),
                lifecycle,
                config_state,
                port=port if isinstance(port, int) else None,
            )
        )
    return registry


def make_handle(client: AdminClient, entry: dict[str, Any]) -> BackendHandle:
    """Build a drivable lifecycle for a newly-assigned backend (slice 6).

    Supplied to :func:`provider_lib.ops.install_backend_ops` so an
    ``agent.assignments.update`` ADD creates a real ``LlamaCppBackend`` bound to
    the entry's port instead of refusing. Removal (busy-safe) is handled by the
    shared reconcile.
    """
    iid = str(entry["instance_id"])
    port = entry.get("port")
    serve_port = port if isinstance(port, int) else None
    backend_config = entry.get("backend_config") or {}
    lifecycle = make_lifecycle(
        client, backend_config, instance_id=iid, serve_port=serve_port
    )
    config_state = ConfigState()
    _apply_assignment(lifecycle, entry, config_state)
    return BackendHandle(iid, lifecycle, config_state, port=serve_port)


def install_command_handlers(
    client: AdminClient,
    target: BackendLifecycle | BackendRegistry,
    emitter: MachineMetricsEmitter,
    config_state: ConfigState | None = None,
) -> None:
    """Register admin->provider command handlers on the client.

    ``target`` is a single lifecycle (single-backend) or a
    :class:`BackendRegistry` (multi-backend). Per-backend commands route to the
    correct lifecycle by ``instance_id`` (H3); a registry agent supplies a
    ``make_handle`` factory so ``agent.assignments.update`` can spawn/retire
    drivable lifecycles (slice 6).
    """
    resolved_state = config_state if config_state is not None else ConfigState()

    async def re_register() -> dict[str, Any] | None:
        """Re-run the registration handshake and adopt its response.

        Called by `provider.initialize` (provider_lib.ops): a fresh
        `/register` re-checks the version + schema gates, re-adopts the
        definition(s), rewrites `CACHE_DIR/provider_config.json` and mints a
        new agent secret for *future* reconnects. The live socket is
        deliberately not recycled — it is already authenticated at the current
        epoch and keeps carrying the status events the operator is watching.
        """
        settings = client.settings
        hardware = await build_hardware_report(settings)
        result = await client.register(
            provider_type=PROVIDER_TYPE,
            version=VERSION,
            base_port=settings.PROVIDER_PORT,
            hardware=hardware,
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

    async def on_metrics_assign(frame: Frame) -> dict[str, Any]:
        logger.info("metrics.assign received: %s", frame.payload)
        # Phase 17: the loop already runs from connect; ownership only toggles
        # whether machine-wide categories are included.
        emitter.start()
        emitter.set_owned(True)
        return {"ok": True, "detail": {"emitting": True}}

    async def on_metrics_unassign(frame: Frame) -> dict[str, Any]:
        logger.info("metrics.unassign received: %s", frame.payload)
        # Drop machine-wide categories but keep the loop alive so GPU
        # categories (per-agent) keep flowing.
        emitter.set_owned(False)
        return {"ok": True, "detail": {"emitting": False}}

    # backend.start / stop / restart + provider.initialize come from
    # provider_lib.ops (shared across providers): an operator-driven boot
    # runs in the background (`wait_for_running: false`) so a cold engine
    # that has to download its weights never blocks the admin's HTTP
    # request or the command ack window. The scheduler keeps sending
    # `wait_for_running: true`, so its contract ("acked == /v1 is live")
    # is unchanged.
    make_handle_fn = (
        (lambda entry: make_handle(client, entry))
        if isinstance(target, BackendRegistry)
        else None
    )
    install_backend_ops(
        client,
        target,
        backend_name=PROVIDER_TYPE,
        re_register=re_register,
        make_handle=make_handle_fn,
    )
    client.on_command("metrics.assign", on_metrics_assign)
    client.on_command("metrics.unassign", on_metrics_unassign)
    install_config_handlers(client, target, resolved_state, client.settings)


def apply_registration(
    lifecycle: BackendLifecycle,
    result: RegistrationResult,
    config_state: ConfigState | None = None,
) -> None:
    """Adopt capacity and the backend_config from the registration response."""
    definition = result.provider_definition
    if result.instance_id is not None:
        # H1/H2: the lifecycle must know its own instance_id so backend.status
        # / backend.metadata / backend.logs frames are keyed to the right
        # backend on the admin.
        lifecycle.instance_id = result.instance_id
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    backend_config = definition.get("backend_config")
    driver = lifecycle.driver
    if isinstance(backend_config, dict) and isinstance(driver, LlamaCppBackend):
        driver.apply_config(backend_config)
    if config_state is not None:
        fp = definition.get("config_fingerprint")
        config_state.applied_fingerprint = fp if isinstance(fp, str) else None


async def emit_provider_status(
    client: AdminClient, lifecycle: BackendLifecycle
) -> None:
    """Announce current instance/backend state to the admin."""
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
) -> tuple[
    RegistrationResult,
    BackendRegistry,
    BackendLifecycle,
    MachineMetricsEmitter,
    LogStreamingBundle,
]:
    """Register over HTTP and host one drivable lifecycle per placed backend.

    Slice 6: builds a :class:`BackendRegistry` (N handles) so the agent can
    serve each backend's ``/v1`` on its own port and drive per-backend
    commands. Returns ``(result, registry, primary_lifecycle, emitter,
    log_bundle)``; the primary (first) backend drives the agent-level metrics
    emitter and log streamer. Does NOT dial the WS — see
    :meth:`AdminClient.run_forever`.
    """
    settings = client.settings
    hardware = await build_hardware_report(settings)
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        base_port=settings.PROVIDER_PORT,
        hardware=hardware,
        # Single shared load (provider_llama_cpp.driver.SCHEMA) so the
        # schema registered with the admin and the schema the driver
        # validates against are provably the same object.
        schema=SCHEMA,
    )
    registry = build_registry(client, result)
    primary = (
        registry.handles()[0].lifecycle if len(registry) else make_lifecycle(client)
    )
    emitter = MachineMetricsEmitter(primary, client, settings)
    install_command_handlers(client, registry, emitter)
    # Phase 13: batched backend.logs / provider.logs streaming + the
    # backend.logs.get catch-up handler (agent-level; keyed to the primary).
    log_bundle = install_log_streaming(client, primary)
    return result, registry, primary, emitter, log_bundle


async def register_provider(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> tuple[
    RegistrationResult,
    BackendLifecycle,
    MachineMetricsEmitter,
    LogStreamingBundle,
]:
    """Build hardware report, register over HTTP, adopt the response config.

    When ``lifecycle`` is supplied (single-backend tests) it is used directly.
    Otherwise the agent hosts one lifecycle per placed backend (slice 6) and
    the first backend's lifecycle is returned as the primary.

    Installs command handlers and creates the metrics emitter but does NOT
    dial the WS — see AdminClient.run_forever for the persistent
    connection with reconnect/backoff.
    """
    if lifecycle is not None:
        settings = client.settings
        hardware = await build_hardware_report(settings)
        result = await client.register(
            provider_type=PROVIDER_TYPE,
            version=VERSION,
            base_port=settings.PROVIDER_PORT,
            hardware=hardware,
            schema=SCHEMA,
        )
        config_state = ConfigState()
        emitter = MachineMetricsEmitter(lifecycle, client, settings)
        install_command_handlers(client, lifecycle, emitter, config_state)
        apply_registration(lifecycle, result, config_state)
        log_bundle = install_log_streaming(client, lifecycle)
        return result, lifecycle, emitter, log_bundle

    result, _registry, primary, emitter, log_bundle = await bootstrap_agent(client)
    return result, primary, emitter, log_bundle


async def register_and_connect(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> tuple[
    RegistrationResult,
    BackendLifecycle,
    MachineMetricsEmitter,
    LogStreamingBundle,
]:
    """Register, adopt config, dial the WS once, emit initial status.

    One-shot connect for tests and simple callers. Production entrypoints
    (run_async) split this into register_provider + run_forever so the
    connection survives admin restarts (ARCHITECTURE.md §5).
    """
    result, lifecycle, emitter, log_bundle = await register_provider(client, lifecycle)
    await client.connect()
    log_bundle.start()
    # Phase 17: start the metrics loop on connect so GPU categories emit even
    # when this agent is not the machine-wide owner.
    emitter.start()
    await emit_provider_status(client, lifecycle)
    return result, lifecycle, emitter, log_bundle


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
    _result, registry, lifecycle, emitter, log_bundle = await bootstrap_agent(client)

    # Keep the admin WS alive for the process lifetime alongside the per-backend
    # HTTP servers: run_forever dials in, re-emits provider.status on every
    # (re)connect, and reconnects with exponential backoff if the admin restarts
    # (ARCHITECTURE.md §5). The metrics emitter is stopped on disconnect so it
    # never queues events onto a dead socket; the admin re-assigns ownership
    # (metrics.assign) when the new connection is accepted.
    first_connect = asyncio.Event()

    async def on_connected() -> None:
        log_bundle.start()
        # Phase 17: GPU categories emit from every connected agent; the loop
        # runs for the socket lifetime (machine-wide categories stay gated on
        # metrics.assign ownership).
        emitter.start()
        await emit_provider_status(client, lifecycle)
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "llama-cpp agent connected: backends=%d epoch=%s",
                len(registry),
                client.epoch,
            )

    async def on_disconnected() -> None:
        await log_bundle.stop()
        emitter.set_owned(False)
        await emitter.stop()

    ws_task = asyncio.create_task(
        client.run_forever(
            on_connected=on_connected,
            on_disconnected=on_disconnected,
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
        await emitter.stop()
        await client.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
