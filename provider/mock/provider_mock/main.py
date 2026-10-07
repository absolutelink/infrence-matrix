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
from provider_lib.config_update import ConfigState, install_config_handlers
from provider_lib.log_stream import install_log_streaming
from provider_lib.ops import install_backend_ops
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


def install_command_handlers(
    client: AdminClient,
    lifecycle: BackendLifecycle,
    config_state: ConfigState | None = None,
) -> None:
    """Register admin->provider command handlers on the client."""
    resolved_state = config_state if config_state is not None else ConfigState()

    async def re_register() -> dict[str, Any] | None:
        """Re-run the registration handshake and adopt its response.

        Called by `provider.initialize` (provider_lib.ops): a fresh
        `/register` re-checks the version + schema gates, re-adopts the
        definition, rewrites `provider_config.json` and mints a new instance
        secret for future reconnects. The live socket stays up — it is what
        carries the status events the operator is watching.
        """
        result = await client.register(
            provider_type=PROVIDER_TYPE,
            version=VERSION,
            port=client.settings.PROVIDER_PORT,
            hardware=FAKE_HARDWARE,
            schema=SCHEMA,
        )
        apply_registration(lifecycle, result, resolved_state)
        return result.provider_definition

    # backend.start / stop / restart + provider.initialize come from
    # provider_lib.ops, shared across providers: an operator-driven boot
    # runs in the background (`wait_for_running: false`) so a slow engine
    # never blocks the admin's request.
    install_backend_ops(
        client, lifecycle, backend_name=PROVIDER_TYPE, re_register=re_register
    )
    install_config_handlers(client, lifecycle, resolved_state, client.settings)


def apply_registration(
    lifecycle: BackendLifecycle,
    result: RegistrationResult,
    config_state: ConfigState | None = None,
) -> None:
    """Adopt capacity and model alias from the registration response."""
    definition = result.provider_definition
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
        # Phase 14: null fingerprint (shell) stays None — never coerced
        # to a hash of {} (that would let the no_config fence pass).
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
) -> tuple[RegistrationResult, BackendLifecycle]:
    """Install command handlers, register over HTTP, adopt the response.

    Does NOT dial the WS — see AdminClient.run_forever for the persistent
    connection with reconnect/backoff.
    """
    if lifecycle is None:
        lifecycle = make_lifecycle(client)
    config_state = ConfigState()
    install_command_handlers(client, lifecycle, config_state)
    result = await client.register(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        port=client.settings.PROVIDER_PORT,
        hardware=FAKE_HARDWARE,
        # Single shared load (provider_mock.backend.SCHEMA) so the schema
        # registered with the admin and the schema the driver validates
        # against are provably the same object.
        schema=SCHEMA,
    )
    apply_registration(lifecycle, result, config_state)
    return result, lifecycle


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
    await register_provider(client, lifecycle)
    # Phase 13: provider.logs streaming + backend.logs.get handler (the
    # mock driver has no subprocess ring, so only provider.logs flows).
    log_bundle = install_log_streaming(client, lifecycle)

    # Keep the admin WS alive for the process lifetime alongside uvicorn:
    # run_forever dials in, re-emits provider.status on every (re)connect,
    # and reconnects with exponential backoff if the admin restarts
    # (ARCHITECTURE.md §5).
    first_connect = asyncio.Event()

    async def on_connected() -> None:
        log_bundle.start()
        await emit_provider_status(client, lifecycle)
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "mock provider connected: instance=%s epoch=%s capacity=%s",
                client.registration.instance_id,
                client.epoch,
                lifecycle.capacity,
            )

    ws_task = asyncio.create_task(
        client.run_forever(
            on_connected=on_connected,
            on_disconnected=log_bundle.stop,
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
        await client.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
