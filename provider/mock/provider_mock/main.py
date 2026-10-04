"""Mock provider package.

A hardware-free provider instance used for local development and to exercise
the full admin -> provider registration + WebSocket path (Phase 3) and the
provider backend lifecycle + /v1 translation surface (Phase 4).

Startup sequence (see ``run_async``):
  1. Register with the admin (POST /admin/api/providers/register) using
     MACHINE_UID + PROVIDER_REGISTRATION_TOKEN from the environment.
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

import uvicorn
from provider_lib.admin_client import AdminClient, RegistrationResult
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.wire import Frame, InstanceStatusValue

from provider_mock.backend import MockBackend

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


def make_lifecycle(client: AdminClient, **stream_kwargs: Any) -> BackendLifecycle:
    """Build the mock's lifecycle with status events routed to the admin."""

    async def emit(status: str, reason: str | None) -> None:
        await client.send_event(
            "backend.status",
            {
                "backend_status": status,
                "instance_status": InstanceStatusValue.RUNNING,
                "reason": reason,
            },
        )

    driver = MockBackend(model=DEFAULT_MODEL, **stream_kwargs)
    return BackendLifecycle(driver, capacity=DEFAULT_CAPACITY, status_callback=emit)


def install_command_handlers(client: AdminClient, lifecycle: BackendLifecycle) -> None:
    """Register admin->provider command handlers on the client."""

    async def on_backend_start(frame: Frame) -> dict[str, Any]:
        logger.info("mock backend.start received: %s", frame.payload)
        await lifecycle.start()
        return {
            "ok": True,
            "detail": {"backend": "mock", "capacity": lifecycle.capacity},
        }

    async def on_backend_stop(frame: Frame) -> dict[str, Any]:
        logger.info("mock backend.stop received: %s", frame.payload)
        await lifecycle.stop()
        return {"ok": True, "detail": {"backend": "mock"}}

    client.on_command("backend.start", on_backend_start)
    client.on_command("backend.stop", on_backend_stop)


def apply_registration(lifecycle: BackendLifecycle, result: RegistrationResult) -> None:
    """Adopt capacity and model alias from the registration response."""
    definition = result.provider_definition
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    alias = definition.get("alias")
    if isinstance(alias, str) and alias and isinstance(lifecycle.driver, MockBackend):
        lifecycle.driver.model = alias


async def register_and_connect(
    client: AdminClient, lifecycle: BackendLifecycle | None = None
) -> RegistrationResult:
    """Register with the admin, adopt its config, dial the WS, emit status.

    Pass an externally created `lifecycle` to share it with the provider
    app; when omitted, a fresh mock lifecycle is created (registration
    still configures it).
    """
    if lifecycle is None:
        lifecycle = make_lifecycle(client)
    install_command_handlers(client, lifecycle)
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        port=client.settings.PROVIDER_PORT,
        hardware=FAKE_HARDWARE,
    )
    apply_registration(lifecycle, result)
    await client.connect()
    await client.send_event(
        "provider.status",
        {
            "instance_status": InstanceStatusValue.RUNNING,
            "backend_status": lifecycle.backend_status,
            "provider_type": PROVIDER_TYPE,
            "version": VERSION,
        },
    )
    return result


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
    lifecycle = make_lifecycle(client)
    await register_and_connect(client, lifecycle)
    logger.info(
        "mock provider connected: instance=%s epoch=%s capacity=%s",
        client.registration.instance_id,
        client.epoch,
        lifecycle.capacity,
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
        await client.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
