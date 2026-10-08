"""halogen provider package entrypoint.

Wires `HalogenBackend` into the generic provider machinery following the
llama-cpp/mock pattern: register once, drive the shared
BackendLifecycle from `backend.start` / `backend.stop`, keep the admin
WS alive with `run_forever`, and serve the agent's single /v1 surface on
PROVIDER_PORT (routing each request to a backend by its ``model`` =
definition alias). Each backend's halogen engine binds a private
OS-assigned (api, engine) port pair the admin never sees. Boot is
admin-driven.

Capacity note: the lifecycle capacity comes from the registration
response (`provider_definition.capacity`); the halogen engine's
effective KV-slot capacity (`options.kv_slots` -> HALOGEN_KV_SLOTS) is
reported in the `backend.start` ack detail, along with both ports.
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
from provider_lib.ops import emit_backend_status_snapshot, install_backend_ops
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.serve import AgentServer
from provider_lib.wire import Frame, InstanceStatusValue

from provider_halogen.driver import SCHEMA, HalogenBackend

logger = logging.getLogger("provider.halogen.main")

PROVIDER_TYPE = "halogen"
VERSION = "dev"
DEFAULT_CAPACITY = 1


async def build_hardware_report(settings: ProviderSettings) -> dict[str, Any]:
    """Best-effort hardware inventory for registration.

    Phase 17: the GPU sample is filtered to ``ASSIGNED_GPU_UUIDS`` (empty =
    every GPU the container sees) so a device-isolated agent reports only its
    own GPU and its own VRAM sum.
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
) -> HalogenBackend:
    """Build the driver with progress/log events routed to the admin.

    Port model overhaul: the engine's private (api, engine) pair is allocated by
    the driver at start (OS-assigned free ports); nothing is threaded in here.
    """

    async def on_progress(payload: dict[str, Any]) -> None:
        if client is not None:
            await client.send_event("download.progress", payload)

    return HalogenBackend(settings, backend_config, progress_cb=on_progress)


def make_lifecycle(
    client: AdminClient,
    backend_config: dict[str, Any] | None = None,
    *,
    instance_id: str | None = None,
):
    """Build the halogen lifecycle with status events to the admin (keyed to
    THIS backend's ``instance_id``)."""
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

    driver = make_driver(settings, backend_config or {}, client=client)
    return BackendLifecycle(
        driver, capacity=DEFAULT_CAPACITY, status_callback=emit, instance_id=_iid()
    )


def _apply_backend(
    lifecycle: BackendLifecycle,
    backend: dict[str, Any],
    config_state: ConfigState | None = None,
) -> None:
    """Adopt capacity + backend_config + fingerprint from ONE registration
    backend dict into its halogen lifecycle/driver."""
    definition = backend.get("definition") or {}
    iid = backend.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, HalogenBackend):
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
    built halogen lifecycle."""
    iid = entry.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = entry.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, HalogenBackend):
        backend_config = entry.get("backend_config")
        if isinstance(backend_config, dict):
            driver.apply_config(backend_config)
    fp = entry.get("config_fingerprint")
    config_state.applied_fingerprint = fp if isinstance(fp, str) else None


def build_registry(client: AdminClient, result: RegistrationResult) -> BackendRegistry:
    """Host one drivable lifecycle per backend the admin placed on this agent.

    The agent's single ``/v1`` surface routes inference by the request ``model``
    (= definition alias); each backend's engine binds its own private port pair.
    """
    registry = BackendRegistry()
    for backend in result.backends:
        iid = backend.get("instance_id")
        if iid is None:  # pragma: no cover - admin always assigns an id
            continue
        definition = backend.get("definition") or {}
        raw_alias = definition.get("alias")
        alias = raw_alias if isinstance(raw_alias, str) else None
        backend_config = definition.get("backend_config") or {}
        lifecycle = make_lifecycle(client, backend_config, instance_id=str(iid))
        config_state = ConfigState()
        _apply_backend(lifecycle, backend, config_state)
        # Port model overhaul: the handle carries its alias so the agent's single
        # /v1 surface routes by model; no per-backend port.
        registry.add(BackendHandle(str(iid), lifecycle, config_state, alias=alias))
    return registry


def make_handle(client: AdminClient, entry: dict[str, Any]) -> BackendHandle:
    """Build a drivable halogen lifecycle for a newly-assigned backend."""
    iid = str(entry["instance_id"])
    raw_alias = entry.get("alias")
    alias = raw_alias if isinstance(raw_alias, str) else None
    backend_config = entry.get("backend_config") or {}
    lifecycle = make_lifecycle(client, backend_config, instance_id=iid)
    config_state = ConfigState()
    _apply_assignment(lifecycle, entry, config_state)
    # Port model overhaul: alias for model routing; no per-backend port.
    return BackendHandle(iid, lifecycle, config_state, alias=alias)


def install_command_handlers(
    client: AdminClient,
    target: BackendLifecycle | BackendRegistry,
    emitter: MachineMetricsEmitter,
    config_state: ConfigState | None = None,
) -> None:
    """Register admin->provider command handlers on the client.

    ``target`` is a single lifecycle (single-backend) or a
    :class:`BackendRegistry` (multi-backend, slice 6).
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
        # H1/H2: the lifecycle must know its own instance_id so backend.status /
        # backend.metadata / backend.logs frames are keyed to the right backend.
        lifecycle.instance_id = result.instance_id
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    backend_config = definition.get("backend_config")
    driver = lifecycle.driver
    if isinstance(backend_config, dict) and isinstance(driver, HalogenBackend):
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
    """Register over HTTP and host one drivable lifecycle per placed backend
    (slice 6). Returns ``(result, registry, primary, emitter, log_bundle)``."""
    settings = client.settings
    hardware = await build_hardware_report(settings)
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        base_port=settings.PROVIDER_PORT,
        hardware=hardware,
        schema=SCHEMA,
    )
    registry = build_registry(client, result)
    primary = (
        registry.handles()[0].lifecycle if len(registry) else make_lifecycle(client)
    )
    emitter = MachineMetricsEmitter(primary, client, settings)
    install_command_handlers(client, registry, emitter)
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
    """Register over HTTP, adopt config, install handlers, build emitter.

    When ``lifecycle`` is supplied (single-backend tests) it is used directly;
    otherwise the agent hosts N backends (slice 6) and the first is returned
    as the primary. Does NOT dial the WS — see AdminClient.run_forever.
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
    """Register, adopt config, dial the WS once, emit initial status."""
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

    first_connect = asyncio.Event()

    async def on_connected() -> None:
        log_bundle.start()
        # Phase 17: GPU categories emit from every connected agent; the loop
        # runs for the socket lifetime (machine-wide categories stay gated on
        # metrics.assign ownership).
        emitter.start()
        await emit_provider_status(client, lifecycle)
        await emit_backend_status_snapshot(client, registry)
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "halogen agent connected: backends=%d epoch=%s",
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
    # Port model overhaul: serve the agent's single /v1 surface on
    # settings.PROVIDER_PORT and route each request to a backend by model (the
    # request's `model` == the definition alias). One listener, no per-backend
    # port churn — a backend added by an assignments push is routable immediately.
    server = AgentServer(settings, PROVIDER_TYPE, VERSION, registry)
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
