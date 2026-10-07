"""halogen provider package entrypoint.

Wires `HalogenBackend` into the generic provider machinery following the
llama-cpp/mock pattern: register once, drive the shared
BackendLifecycle from `backend.start` / `backend.stop`, keep the admin
WS alive with `run_forever`, and serve the provider /v1 surface on
PROVIDER_PORT. Boot is admin-driven.

Capacity note: the lifecycle capacity comes from the registration
response (`provider_definition.capacity`); the halogen engine's
effective KV-slot capacity (`options.kv_slots` -> HALOGEN_KV_SLOTS) is
reported in the `backend.start` ack detail, along with both ports.
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
from provider_lib.ops import install_backend_ops
from provider_lib.wire import Frame, InstanceStatusValue

from provider_halogen.driver import SCHEMA, HalogenBackend

logger = logging.getLogger("provider.halogen.main")

PROVIDER_TYPE = "halogen"
VERSION = "dev"
DEFAULT_CAPACITY = 1


async def build_hardware_report(settings: ProviderSettings) -> dict[str, Any]:
    """Best-effort hardware inventory for registration."""
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
) -> HalogenBackend:
    """Build the driver with progress/log events routed to the admin."""

    async def on_progress(payload: dict[str, Any]) -> None:
        if client is not None:
            await client.send_event("download.progress", payload)

    return HalogenBackend(settings, backend_config, progress_cb=on_progress)


def make_lifecycle(client: AdminClient, backend_config: dict[str, Any] | None = None):
    """Build the halogen lifecycle with status events to the admin."""
    settings = client.settings

    async def emit(status: str, reason: str | None) -> None:
        payload: dict[str, Any] = {
            "backend_status": status,
            "agent_status": InstanceStatusValue.RUNNING,
            "reason": reason,
        }
        if client.is_registered and client.registration.instance_id is not None:
            payload["instance_id"] = client.registration.instance_id
        await client.send_event("backend.status", payload)

    driver = make_driver(settings, backend_config or {}, client=client)
    return BackendLifecycle(driver, capacity=DEFAULT_CAPACITY, status_callback=emit)


def install_command_handlers(
    client: AdminClient,
    lifecycle: BackendLifecycle,
    emitter: MachineMetricsEmitter,
    config_state: ConfigState | None = None,
) -> None:
    """Register admin->provider command handlers on the client."""
    resolved_state = config_state if config_state is not None else ConfigState()

    async def re_register() -> dict[str, Any] | None:
        """Re-run the registration handshake and adopt its response.

        Called by `provider.initialize` (provider_lib.ops): a fresh
        `/register` re-checks the version + schema gates, re-adopts capacity
        and `backend_config`, rewrites `CACHE_DIR/provider_config.json` and
        mints a new instance secret for *future* reconnects. The live socket
        is deliberately not recycled — it is already authenticated at the
        current epoch and keeps carrying the status events the operator is
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
        apply_registration(lifecycle, result, resolved_state)
        return result.provider_definition

    async def on_metrics_assign(frame: Frame) -> dict[str, Any]:
        logger.info("metrics.assign received: %s", frame.payload)
        emitter.start()
        return {"ok": True, "detail": {"emitting": True}}

    async def on_metrics_unassign(frame: Frame) -> dict[str, Any]:
        logger.info("metrics.unassign received: %s", frame.payload)
        await emitter.stop()
        return {"ok": True, "detail": {"emitting": False}}

    # backend.start / stop / restart + provider.initialize come from
    # provider_lib.ops (shared across providers): an operator-driven boot
    # runs in the background (`wait_for_running: false`) so a cold engine
    # that has to download its weights never blocks the admin's HTTP
    # request or the command ack window. The scheduler keeps sending
    # `wait_for_running: true`, so its contract ("acked == /v1 is live")
    # is unchanged.
    install_backend_ops(
        client, lifecycle, backend_name=PROVIDER_TYPE, re_register=re_register
    )
    client.on_command("metrics.assign", on_metrics_assign)
    client.on_command("metrics.unassign", on_metrics_unassign)
    install_config_handlers(client, lifecycle, resolved_state, client.settings)


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


async def register_provider(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> tuple[
    RegistrationResult,
    BackendLifecycle,
    MachineMetricsEmitter,
    LogStreamingBundle,
]:
    """Register over HTTP, adopt config, install handlers, build emitter.

    Does NOT dial the WS — see AdminClient.run_forever.
    """
    settings = client.settings
    hardware = await build_hardware_report(settings)
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        base_port=settings.PROVIDER_PORT,
        hardware=hardware,
        # Single shared load (provider_halogen.driver.SCHEMA) so the
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
    """Register, adopt config, dial the WS once, emit initial status."""
    result, lifecycle, emitter, log_bundle = await register_provider(client, lifecycle)
    await client.connect()
    log_bundle.start()
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
    _result, lifecycle, emitter, log_bundle = await register_provider(client)

    first_connect = asyncio.Event()

    async def on_connected() -> None:
        log_bundle.start()
        await emit_provider_status(client, lifecycle)
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "halogen provider connected: instance=%s epoch=%s capacity=%s",
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
        build_app(lifecycle, settings),
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
