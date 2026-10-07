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

import uvicorn
from provider_lib.admin_client import AdminClient, RegistrationResult
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState, install_config_handlers
from provider_lib.log_stream import LogStreamingBundle, install_log_streaming
from provider_lib.metrics import MachineMetricsEmitter, collect_machine_snapshot
from provider_lib.wire import Frame, InstanceStatusValue

from provider_llama_cpp.driver import SCHEMA, LlamaCppBackend

logger = logging.getLogger("provider.llama_cpp.main")

PROVIDER_TYPE = "llama-cpp"
VERSION = "dev"
DEFAULT_CAPACITY = 1


async def build_hardware_report(settings: ProviderSettings) -> dict[str, Any]:
    """Best-effort hardware inventory for registration.

    Uses the metrics collectors; whatever the host cannot report is
    simply omitted (empty lists / zeros are acceptable).
    """
    snapshot = await collect_machine_snapshot(
        {"vram", "gpu_usage", "os_ram", "cpu", "storage"},
        models_dir=settings.MODELS_DIR,
        cache_dir=settings.CACHE_DIR,
    )
    vram = snapshot.get("vram", {})
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
    }


def make_driver(
    settings: ProviderSettings,
    backend_config: dict[str, Any],
    client: AdminClient | None = None,
) -> LlamaCppBackend:
    """Build the driver with progress/log events routed to the admin."""

    async def on_progress(payload: dict[str, Any]) -> None:
        if client is not None:
            await client.send_event("download.progress", payload)

    return LlamaCppBackend(settings, backend_config, progress_cb=on_progress)


def make_lifecycle(client: AdminClient, backend_config: dict[str, Any] | None = None):
    """Build the llama-cpp lifecycle with status events to the admin."""
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

    async def on_backend_start(frame: Frame) -> dict[str, Any]:
        # Phase 14: fence — a shell definition (no applied config) never
        # starts a backend on defaults.
        nak_fn = getattr(client, "no_config_nak", None)
        if nak_fn is not None:
            nak = await nak_fn(frame)
            if nak is not None:
                return nak
        await lifecycle.start()
        driver = lifecycle.driver
        return {
            "ok": True,
            "detail": {
                "backend": PROVIDER_TYPE,
                "capacity": lifecycle.capacity,
                "backend_port": getattr(driver, "backend_port", None),
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
        client, lifecycle, config_state or ConfigState(), client.settings
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
            "instance_status": InstanceStatusValue.RUNNING,
            "backend_status": lifecycle.backend_status,
            "provider_type": PROVIDER_TYPE,
            "version": VERSION,
        },
    )


async def register_provider(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> tuple[
    RegistrationResult,
    BackendLifecycle,
    MachineMetricsEmitter,
    LogStreamingBundle,
]:
    """Build hardware report, register over HTTP, adopt the response config.

    Installs command handlers and creates the metrics emitter but does NOT
    dial the WS — see AdminClient.run_forever for the persistent
    connection with reconnect/backoff.
    """
    settings = client.settings
    hardware = await build_hardware_report(settings)
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        port=settings.PROVIDER_PORT,
        hardware=hardware,
        # Single shared load (provider_llama_cpp.driver.SCHEMA) so the
        # schema registered with the admin and the schema the driver
        # validates against are provably the same object.
        schema=SCHEMA,
    )
    backend_config = result.provider_definition.get("backend_config") or {}
    if lifecycle is None:
        lifecycle = make_lifecycle(client, backend_config)
    config_state = ConfigState()
    emitter = MachineMetricsEmitter(lifecycle, client, settings)
    install_command_handlers(client, lifecycle, emitter, config_state)
    apply_registration(lifecycle, result, config_state)
    # Phase 13: batched backend.logs / provider.logs streaming + the
    # backend.logs.get catch-up handler. The streamer is started on every
    # WS connect (run_async) and stopped on disconnect; cursors persist
    # so lines produced while disconnected ship on reconnect.
    log_bundle = install_log_streaming(client, lifecycle)
    return result, lifecycle, emitter, log_bundle


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
    await emit_provider_status(client, lifecycle)
    return result, lifecycle, emitter, log_bundle


def build_app(lifecycle: BackendLifecycle | None = None):
    settings = ProviderSettings()
    overrides = BackendOverrides(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        lifecycle=lifecycle,
    )
    return create_provider_app(settings, overrides)


async def run_async() -> None:
    settings = ProviderSettings()
    client = AdminClient(settings)
    _result, lifecycle, emitter, log_bundle = await register_provider(client)

    # Keep the admin WS alive for the process lifetime alongside uvicorn:
    # run_forever dials in, re-emits provider.status on every (re)connect,
    # and reconnects with exponential backoff if the admin restarts
    # (ARCHITECTURE.md §5). The metrics emitter is stopped on disconnect so
    # it never queues events onto a dead socket; the admin re-assigns
    # ownership (metrics.assign) when the new connection is accepted.
    first_connect = asyncio.Event()

    async def on_connected() -> None:
        log_bundle.start()
        await emit_provider_status(client, lifecycle)
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "llama-cpp provider connected: instance=%s epoch=%s capacity=%s",
                client.registration.instance_id,
                client.epoch,
                lifecycle.capacity,
            )

    async def on_disconnected() -> None:
        await log_bundle.stop()
        await emitter.stop()

    ws_task = asyncio.create_task(
        client.run_forever(
            on_connected=on_connected,
            on_disconnected=on_disconnected,
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
