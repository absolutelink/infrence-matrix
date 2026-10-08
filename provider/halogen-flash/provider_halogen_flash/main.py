"""halogen-flash provider package entrypoint.

Wires `HalogenFlashBackend` into the generic provider machinery
(llama-cpp/mock pattern): register once, drive the shared
BackendLifecycle from `backend.start` / `backend.stop`, keep the admin
WS alive with `run_forever`, serve the /v1 surface on PROVIDER_PORT.
Boot is admin-driven.

This package also demonstrates the **usage-normalization override**: the
driver normalizes the backend's non-spec usage into a spec dict before
yielding terminal events (see `provider_halogen_flash.usage`), and the
same function is exposed on `BackendOverrides.calculate_usage` so any
generic lib consumer sees the per-type override hook wired.
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

from provider_halogen_flash.driver import SCHEMA, HalogenFlashBackend
from provider_halogen_flash.usage import calculate_usage

logger = logging.getLogger("provider.halogen_flash.main")

PROVIDER_TYPE = "halogen-flash"
VERSION = "dev"
DEFAULT_CAPACITY = 1


async def build_hardware_report(settings: ProviderSettings) -> dict[str, Any]:
    """Best-effort hardware inventory for registration.

    NPU probe results are reported under ``npu`` so the admin/UI can
    gate NPU small-model options on capable machines (legacy parity).

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
    from provider_halogen_flash.npu import probe_npu

    try:
        npu = await asyncio.to_thread(
            probe_npu,
            device_path=settings.NPU_DEVICE_PATH,
            xrt_lib_dir=settings.NPU_XRT_LIB_DIR,
            binary_path=settings.NPU_BINARY_PATH,
        )
    except Exception:  # noqa: BLE001 - hardware report is best-effort
        logger.warning("NPU probe failed during hardware report", exc_info=True)
        npu = {"available": False, "reasons": ["probe error"]}
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
        "npu": npu,
    }


def make_driver(
    settings: ProviderSettings,
    backend_config: dict[str, Any],
    client: AdminClient | None = None,
) -> HalogenFlashBackend:
    """Build the driver with progress/log events routed to the admin."""

    async def on_progress(payload: dict[str, Any]) -> None:
        if client is not None:
            await client.send_event("download.progress", payload)

    return HalogenFlashBackend(settings, backend_config, progress_cb=on_progress)


def make_lifecycle(
    client: AdminClient,
    backend_config: dict[str, Any] | None = None,
    *,
    instance_id: str | None = None,
) -> BackendLifecycle:
    """Build the halogen-flash lifecycle with status events to the admin
    (slice 6: keyed to THIS backend's ``instance_id``)."""
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
    backend dict into its halogen-flash lifecycle/driver."""
    definition = backend.get("definition") or {}
    iid = backend.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, HalogenFlashBackend):
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
    built halogen-flash lifecycle."""
    iid = entry.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = entry.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, HalogenFlashBackend):
        backend_config = entry.get("backend_config")
        if isinstance(backend_config, dict):
            driver.apply_config(backend_config)
    fp = entry.get("config_fingerprint")
    config_state.applied_fingerprint = fp if isinstance(fp, str) else None


def build_registry(client: AdminClient, result: RegistrationResult) -> BackendRegistry:
    """Host one drivable lifecycle per backend the admin placed on this agent.

    halogen-flash declares ``x-max-running-backends: 1``, so in practice the
    admin places at most one backend here; the registry keeps the code path
    uniform with the other providers.
    """
    registry = BackendRegistry()
    for backend in result.backends:
        iid = backend.get("instance_id")
        if iid is None:  # pragma: no cover - admin always assigns an id
            continue
        port = backend.get("port")
        serve_port = port if isinstance(port, int) else None
        definition = backend.get("definition") or {}
        backend_config = definition.get("backend_config") or {}
        lifecycle = make_lifecycle(client, backend_config, instance_id=str(iid))
        config_state = ConfigState()
        _apply_backend(lifecycle, backend, config_state)
        registry.add(BackendHandle(str(iid), lifecycle, config_state, port=serve_port))
    return registry


def make_handle(client: AdminClient, entry: dict[str, Any]) -> BackendHandle:
    """Build a drivable halogen-flash lifecycle for a newly-assigned backend."""
    iid = str(entry["instance_id"])
    port = entry.get("port")
    serve_port = port if isinstance(port, int) else None
    backend_config = entry.get("backend_config") or {}
    lifecycle = make_lifecycle(client, backend_config, instance_id=iid)
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
    :class:`BackendRegistry` (multi-backend, slice 6).
    """
    resolved_state = config_state if config_state is not None else ConfigState()

    async def re_register() -> dict[str, Any] | None:
        """Re-run the registration handshake and adopt its response.

        `provider.initialize` calls this: a fresh `/register` POST re-checks
        the version + schema gates, re-adopts capacity and
        `backend_config`, rewrites `CACHE_DIR/provider_config.json` and
        mints a new instance secret for *future* reconnects. The live socket
        is deliberately not recycled — it is already authenticated at the
        current epoch and is what carries the status events the operator is
        watching.
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
    # provider_lib.ops (shared across providers): manual control may run the
    # boot in the background, which is how a multi-GB engine-side download
    # stays out of the admin's HTTP request and ack windows.
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
    install_config_handlers(
        client,
        target,
        resolved_state,
        client.settings,
        # The Flash engine's HALOGEN_CACHE_DIR is its on-disk prompt
        # cache; cleared alongside the shared prompt_cache root. Model
        # files (MODELS_DIR) are never touched.
        extra_cache_dirs=[client.settings.CACHE_DIR / "halogen-flash"],
    )


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
    if isinstance(backend_config, dict) and isinstance(driver, HalogenFlashBackend):
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


def build_app(lifecycle: BackendLifecycle | None = None):
    # Reuse the settings bound to the driver when an embedded caller already
    # constructed the lifecycle. This avoids a second env parse and keeps the
    # provider HTTP app aligned with the backend configuration.
    settings = (
        getattr(getattr(lifecycle, "driver", None), "_settings", None)
        if lifecycle is not None
        else None
    )
    if settings is None:
        # Route construction is useful without provider wiring (for schema
        # checks and health probes); mandatory registration settings are only
        # needed by AdminClient, not by create_provider_app.
        settings = ProviderSettings.model_construct()
    overrides = BackendOverrides(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        calculate_usage=calculate_usage,
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
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "halogen-flash agent connected: backends=%d epoch=%s",
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
    # offset); a single-backend agent yields exactly one listener on
    # PROVIDER_PORT — identical to the pre-slice-6 behavior. The usage
    # normalization override is preserved on every served backend.
    server = MultiPortServer(
        settings,
        PROVIDER_TYPE,
        VERSION,
        registry,
        calculate_usage=calculate_usage,
    )
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
