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

import uvicorn
from provider_lib.admin_client import AdminClient, RegistrationResult
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState, install_config_handlers
from provider_lib.metrics import MachineMetricsEmitter, collect_machine_snapshot
from provider_lib.wire import Frame, InstanceStatusValue

from provider_halogen_flash.driver import HalogenFlashBackend
from provider_halogen_flash.usage import calculate_usage

logger = logging.getLogger("provider.halogen_flash.main")

PROVIDER_TYPE = "halogen-flash"
VERSION = "dev"
DEFAULT_CAPACITY = 1


async def build_hardware_report(settings: ProviderSettings) -> dict[str, Any]:
    """Best-effort hardware inventory for registration.

    NPU probe results are reported under ``npu`` so the admin/UI can
    gate NPU small-model options on capable machines (legacy parity).
    """
    snapshot = await collect_machine_snapshot(
        {"vram", "gpu_usage", "os_ram", "cpu", "storage"},
        models_dir=settings.MODELS_DIR,
        cache_dir=settings.CACHE_DIR,
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
                "uuid": g.get("uuid") or f"gpu-{g.get('id')}",
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

    async def on_log(stream: str, line: str) -> None:
        if client is not None:
            await client.send_event("backend.logs", {"stream": stream, "line": line})

    return HalogenFlashBackend(
        settings, backend_config, progress_cb=on_progress, log_cb=on_log
    )


def make_lifecycle(
    client: AdminClient, backend_config: dict[str, Any] | None = None
) -> BackendLifecycle:
    """Build the halogen-flash lifecycle with status events to the admin."""
    settings = client.settings

    async def emit(status: str, reason: str | None) -> None:
        await client.send_event(
            "backend.status",
            {
                "backend_status": status,
                "instance_status": InstanceStatusValue.RUNNING,
                "reason": reason,
            },
        )

    driver = make_driver(settings, backend_config or {}, client=client)
    return BackendLifecycle(driver, capacity=DEFAULT_CAPACITY, status_callback=emit)


def install_command_handlers(
    client: AdminClient,
    lifecycle: BackendLifecycle,
    emitter: MachineMetricsEmitter,
    config_state: ConfigState | None = None,
) -> None:
    """Register admin->provider command handlers on the client."""

    async def on_backend_start(frame: Frame) -> dict[str, Any]:  # noqa: ARG001
        await lifecycle.start()
        driver = lifecycle.driver
        return {
            "ok": True,
            "detail": {
                "backend": PROVIDER_TYPE,
                "capacity": lifecycle.capacity,
                "effective_capacity": getattr(driver, "effective_capacity", None),
                "api_port": getattr(driver, "api_port", None),
                "engine_port": getattr(driver, "engine_port", None),
            },
        }

    async def on_backend_stop(frame: Frame) -> dict[str, Any]:  # noqa: ARG001
        await lifecycle.stop()
        return {"ok": True, "detail": {"backend": PROVIDER_TYPE}}

    async def on_metrics_assign(frame: Frame) -> dict[str, Any]:
        logger.info("metrics.assign received: %s", frame.payload)
        emitter.start()
        return {"ok": True, "detail": {"emitting": True}}

    async def on_metrics_unassign(frame: Frame) -> dict[str, Any]:
        logger.info("metrics.unassign received: %s", frame.payload)
        await emitter.stop()
        return {"ok": True, "detail": {"emitting": False}}

    client.on_command("backend.start", on_backend_start)
    client.on_command("backend.stop", on_backend_stop)
    client.on_command("metrics.assign", on_metrics_assign)
    client.on_command("metrics.unassign", on_metrics_unassign)
    install_config_handlers(
        client,
        lifecycle,
        config_state or ConfigState(),
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
            "instance_status": InstanceStatusValue.RUNNING,
            "backend_status": lifecycle.backend_status,
            "provider_type": PROVIDER_TYPE,
            "version": VERSION,
        },
    )


async def register_provider(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> tuple[RegistrationResult, BackendLifecycle, MachineMetricsEmitter]:
    """Register over HTTP, adopt config, install handlers, build emitter.

    Does NOT dial the WS — see AdminClient.run_forever.
    """
    settings = client.settings
    hardware = await build_hardware_report(settings)
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        port=settings.PROVIDER_PORT,
        hardware=hardware,
    )
    backend_config = result.provider_definition.get("backend_config") or {}
    if lifecycle is None:
        lifecycle = make_lifecycle(client, backend_config)
    config_state = ConfigState()
    emitter = MachineMetricsEmitter(lifecycle, client, settings)
    install_command_handlers(client, lifecycle, emitter, config_state)
    apply_registration(lifecycle, result, config_state)
    return result, lifecycle, emitter


async def register_and_connect(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> tuple[RegistrationResult, BackendLifecycle, MachineMetricsEmitter]:
    """Register, adopt config, dial the WS once, emit initial status."""
    result, lifecycle, emitter = await register_provider(client, lifecycle)
    await client.connect()
    await emit_provider_status(client, lifecycle)
    return result, lifecycle, emitter


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
    _result, lifecycle, emitter = await register_provider(client)

    first_connect = asyncio.Event()

    async def on_connected() -> None:
        await emit_provider_status(client, lifecycle)
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "halogen-flash provider connected: instance=%s epoch=%s capacity=%s",
                client.registration.instance_id,
                client.epoch,
                lifecycle.capacity,
            )

    ws_task = asyncio.create_task(
        client.run_forever(
            on_connected=on_connected,
            on_disconnected=emitter.stop,
        )
    )
    config = uvicorn.Config(
        build_app(lifecycle),
        host="0.0.0.0",
        port=settings.PROVIDER_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        ws_task.cancel()
        await asyncio.gather(ws_task, return_exceptions=True)
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
