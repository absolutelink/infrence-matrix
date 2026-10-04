"""Mock provider package.

A hardware-free provider instance used for local development and to exercise
the full admin -> provider registration + WebSocket path (Phase 3) and later
the scheduler -> litellm -> SSE path.

Startup sequence (see ``run_mock_provider``):
  1. Register with the admin (POST /admin/api/providers/register) using
     MACHINE_UID + PROVIDER_REGISTRATION_TOKEN from the environment.
  2. Install command handlers (backend.start / backend.stop).
  3. Dial the admin WebSocket (AdminClient.connect) and emit an initial
     provider.status event.
  4. Serve the provider app (GET /health) with uvicorn.

The registration + connect logic is factored into async helpers so an
integration test can drive it programmatically without uvicorn.
"""

import asyncio
import logging
from typing import Any

import uvicorn
from provider_lib.admin_client import AdminClient, RegistrationResult
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.config import ProviderSettings
from provider_lib.wire import BackendStatusValue, Frame, InstanceStatusValue

logger = logging.getLogger("provider.mock")

PROVIDER_TYPE = "mock"
VERSION = "dev"

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


def build_app():
    settings = ProviderSettings()
    overrides = BackendOverrides(provider_type=PROVIDER_TYPE, version=VERSION)
    return create_provider_app(settings, overrides)


def install_command_handlers(client: AdminClient) -> None:
    """Register admin->provider command handlers on the client."""

    async def on_backend_start(frame: Frame) -> dict[str, Any]:
        logger.info("mock backend.start received: %s", frame.payload)
        # Pretend the fake backend came up.
        await client.send_event(
            "backend.status",
            {
                "backend_status": BackendStatusValue.RUNNING,
                "instance_status": InstanceStatusValue.RUNNING,
                "reason": "mock backend started",
            },
        )
        return {"ok": True, "detail": {"backend": "mock"}}

    async def on_backend_stop(frame: Frame) -> dict[str, Any]:
        logger.info("mock backend.stop received: %s", frame.payload)
        await client.send_event(
            "backend.status",
            {
                "backend_status": BackendStatusValue.STOPPED,
                "instance_status": InstanceStatusValue.RUNNING,
                "reason": "mock backend stopped",
            },
        )
        return {"ok": True, "detail": {"backend": "mock"}}

    client.on_command("backend.start", on_backend_start)
    client.on_command("backend.stop", on_backend_stop)


async def register_and_connect(
    client: AdminClient,
) -> RegistrationResult:
    """Register with the admin, dial the WS, and emit initial status."""
    install_command_handlers(client)
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        port=client.settings.PROVIDER_PORT,
        hardware=FAKE_HARDWARE,
    )
    await client.connect()
    await client.send_event(
        "provider.status",
        {
            "instance_status": InstanceStatusValue.RUNNING,
            "backend_status": BackendStatusValue.STOPPED,
            "provider_type": PROVIDER_TYPE,
            "version": VERSION,
        },
    )
    return result


async def run_async() -> None:
    settings = ProviderSettings()
    client = AdminClient(settings)
    await register_and_connect(client)
    logger.info(
        "mock provider connected: instance=%s epoch=%s",
        client.registration.instance_id,
        client.epoch,
    )
    config = uvicorn.Config(
        build_app(),
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
