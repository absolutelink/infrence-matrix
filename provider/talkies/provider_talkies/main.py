"""talkies provider package entrypoint.

Wires `TalkiesBackend` into the generic provider machinery following the
gufo/llama-cpp pattern:

  1. Register with the admin (hardware report from provider_lib.metrics).
  2. Build a BackendLifecycle per placed backend around the driver; status
     transitions are emitted as `backend.status` WS events.
  3. Command handlers: backend.start / backend.stop / provider.initialize
     drive the shared lifecycle; metrics.assign / metrics.unassign start/
     stop the machine-level metrics emitter; provider.config.update /
     cache.clear / storage.prune_unused come from provider_lib.
  4. Serve the agent's single /v1 surface (built by provider_lib, incl.
     the Phase 24 /v1/audio/* routes) on PROVIDER_PORT; each request is
     routed to a backend by its ``model`` (= definition alias). Each
     backend's talkies engine binds a private OS-assigned local port the
     admin never sees.

Boot is admin-driven: after connect a backend stays STOPPED until the
admin sends `backend.start`. The backend_config comes from the
registration response (`provider_definition.backend_config`).

One talkies process per definition (Phase 24 spike finding): talkies reads
its registry + enabled-model set only at import, so the per-definition
ProviderInstance lifecycle requires a per-definition process. There is no
x-max-running-backends cap — VRAM admission governs how many talkies
processes share a machine.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from provider_lib.admin_client import AdminClient, RegistrationResult
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
from provider_lib.models import ModelSpec, models_from_entry
from provider_lib.ops import emit_backend_status_snapshot, install_backend_ops
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.serve import AgentServer
from provider_lib.wire import Frame, InstanceStatusValue

from provider_talkies.driver import SCHEMA, TalkiesBackend

logger = logging.getLogger("provider.talkies.main")

PROVIDER_TYPE = "talkies"
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
) -> TalkiesBackend:
    """Build the driver with download-progress events routed to the admin.

    The engine's private loopback port is allocated by the driver at start
    (OS-assigned free port); nothing is threaded in here.
    """

    async def on_progress(payload: dict[str, Any]) -> None:
        if client is not None:
            await client.send_event("download.progress", payload)

    return TalkiesBackend(settings, backend_config, progress_cb=on_progress)


def make_lifecycle(
    client: AdminClient,
    backend_config: dict[str, Any] | None = None,
    *,
    instance_id: str | None = None,
):
    """Build the talkies lifecycle with status events to the admin (keyed to
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


def _apply_modality(driver: TalkiesBackend, source: dict[str, Any]) -> None:
    """Thread the definition/entry's endpoint kind onto the driver (mirrors
    what provider_lib.assignments does for pushed adds)."""
    raw = source.get("modality")
    if isinstance(raw, str) and raw:
        driver.modality = raw


def _first_enabled_name(models: list[ModelSpec]) -> str | None:
    """The first enabled served name (the multi-model ``alias`` mirror)."""
    for spec in models:
        if spec.enabled:
            return spec.name
    return None


def _apply_backend(
    lifecycle: BackendLifecycle,
    backend: dict[str, Any],
    config_state: ConfigState | None = None,
) -> None:
    """Adopt capacity + backend_config + alias + served models + fingerprint
    from ONE registration backend dict into its talkies lifecycle/driver."""
    definition = backend.get("definition") or {}
    iid = backend.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, TalkiesBackend):
        raw_alias = definition.get("alias")
        if isinstance(raw_alias, str):
            driver.set_alias(raw_alias)
        _apply_modality(driver, definition)
        backend_config = definition.get("backend_config")
        if isinstance(backend_config, dict):
            driver.apply_config(backend_config)
        # Phase 25: adopt the served-model list (prefers ``served_models``) so
        # the driver's name -> slug map is current before any start.
        driver.set_models(models_from_entry(definition))
    if config_state is not None:
        fp = definition.get("config_fingerprint")
        config_state.applied_fingerprint = fp if isinstance(fp, str) else None


def _apply_assignment(
    lifecycle: BackendLifecycle,
    entry: dict[str, Any],
    config_state: ConfigState,
) -> None:
    """Adopt an ``agent.assignments.update`` entry (flat shape) into a freshly
    built talkies lifecycle."""
    iid = entry.get("instance_id")
    if iid is not None:
        lifecycle.instance_id = str(iid)
    capacity = entry.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, TalkiesBackend):
        raw_alias = entry.get("alias")
        if isinstance(raw_alias, str):
            driver.set_alias(raw_alias)
        _apply_modality(driver, entry)
        backend_config = entry.get("backend_config")
        if isinstance(backend_config, dict):
            driver.apply_config(backend_config)
        # Phase 25: same served-model adoption on the assignment path.
        driver.set_models(models_from_entry(entry))
    fp = entry.get("config_fingerprint")
    config_state.applied_fingerprint = fp if isinstance(fp, str) else None


def build_registry(client: AdminClient, result: RegistrationResult) -> BackendRegistry:
    """Host one drivable lifecycle per backend the admin placed on this agent.

    The agent's single ``/v1`` surface routes inference by the request ``model``
    (any enabled served name — Phase 25 multi-model); each backend's talkies
    engine binds its own private port.
    """
    registry = BackendRegistry()
    for backend in result.backends:
        iid = backend.get("instance_id")
        if iid is None:  # pragma: no cover - admin always assigns an id
            continue
        definition = backend.get("definition") or {}
        models = models_from_entry(definition)
        raw_alias = definition.get("alias")
        alias = _first_enabled_name(models) or (
            raw_alias if isinstance(raw_alias, str) else None
        )
        backend_config = definition.get("backend_config") or {}
        lifecycle = make_lifecycle(client, backend_config, instance_id=str(iid))
        config_state = ConfigState()
        _apply_backend(lifecycle, backend, config_state)
        registry.add(
            BackendHandle(str(iid), lifecycle, config_state, alias=alias, models=models)
        )
    return registry


def make_handle(client: AdminClient, entry: dict[str, Any]) -> BackendHandle:
    """Build a drivable talkies lifecycle for a newly-assigned backend."""
    iid = str(entry["instance_id"])
    models = models_from_entry(entry)
    raw_alias = entry.get("alias")
    alias = _first_enabled_name(models) or (
        raw_alias if isinstance(raw_alias, str) else None
    )
    backend_config = entry.get("backend_config") or {}
    lifecycle = make_lifecycle(client, backend_config, instance_id=iid)
    config_state = ConfigState()
    _apply_assignment(lifecycle, entry, config_state)
    return BackendHandle(iid, lifecycle, config_state, alias=alias, models=models)


def install_command_handlers(
    client: AdminClient,
    target: BackendLifecycle | BackendRegistry,
    emitter: MachineMetricsEmitter,
    config_state: ConfigState | None = None,
) -> None:
    """Register admin->provider command handlers on the client.

    ``target`` is a single lifecycle (single-backend) or a
    :class:`BackendRegistry` (multi-backend).
    """
    resolved_state = config_state if config_state is not None else ConfigState()

    async def re_register() -> dict[str, Any] | None:
        """Re-run the registration handshake and adopt its response.

        Called by `provider.initialize` (provider_lib.ops): a fresh
        `/register` re-checks the version + schema gates, re-adopts the
        definition(s), rewrites `CACHE_DIR/provider_config.json` and mints a
        new agent secret for *future* reconnects. The live socket is
        deliberately not recycled.
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
        emitter.start()
        emitter.set_owned(True)
        return {"ok": True, "detail": {"emitting": True}}

    async def on_metrics_unassign(frame: Frame) -> dict[str, Any]:
        logger.info("metrics.unassign received: %s", frame.payload)
        emitter.set_owned(False)
        return {"ok": True, "detail": {"emitting": False}}

    # backend.start / stop / restart + provider.initialize come from
    # provider_lib.ops. A cold talkies boot PREFETCHES its snapshot and
    # loads the checkpoint (minutes), so operator-driven boots use the
    # accept-style contract (wait_for_running: false) with `initializing`
    # heartbeats; the scheduler keeps its blocking contract unchanged.
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
        # talkies' file-staging area ($TALKIES_DATA_DIR/files) is the
        # provider-owned engine cache: cleared alongside the shared
        # prompt-cache root. Model snapshots (models/) and custom voices
        # (custom-voices/) are NEVER touched by cache.clear.
        extra_cache_dirs=lambda _handle: [talkies_data_dir(client.settings) / "files"],
    )


def talkies_data_dir(settings: ProviderSettings) -> Path:
    """The talkies data root (env override, else MODELS_DIR)."""
    if settings.TALKIES_DATA_DIR:
        return Path(settings.TALKIES_DATA_DIR)
    return Path(settings.MODELS_DIR)


def apply_registration(
    lifecycle: BackendLifecycle,
    result: RegistrationResult,
    config_state: ConfigState | None = None,
) -> None:
    """Adopt capacity and the backend_config from the registration response."""
    definition = result.provider_definition
    if result.instance_id is not None:
        lifecycle.instance_id = result.instance_id
    capacity = definition.get("capacity")
    if isinstance(capacity, int) and capacity >= 1:
        lifecycle.capacity = capacity
    driver = lifecycle.driver
    if isinstance(driver, TalkiesBackend):
        raw_alias = definition.get("alias")
        if isinstance(raw_alias, str):
            driver.set_alias(raw_alias)
        _apply_modality(driver, definition)
    backend_config = definition.get("backend_config")
    if isinstance(backend_config, dict) and isinstance(driver, TalkiesBackend):
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

    Returns ``(result, registry, primary, emitter, log_bundle)``.
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
    otherwise the agent hosts N backends and the first is returned as the
    primary. Does NOT dial the WS — see AdminClient.run_forever.
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
    emitter.start()
    await emit_provider_status(client, lifecycle)
    return result, lifecycle, emitter, log_bundle


def build_app(
    lifecycle: BackendLifecycle | None = None,
    settings: ProviderSettings | None = None,
    registry: BackendRegistry | None = None,
):
    from provider_lib.app_factory import BackendOverrides, create_provider_app

    settings = settings or ProviderSettings()
    overrides = BackendOverrides(
        provider_type=PROVIDER_TYPE,
        version=VERSION,
        lifecycle=lifecycle,
        registry=registry,
    )
    return create_provider_app(settings, overrides)


async def run_async() -> None:
    settings = ProviderSettings()
    client = AdminClient(settings)
    _result, registry, lifecycle, emitter, log_bundle = await bootstrap_agent(client)

    first_connect = asyncio.Event()

    async def on_connected() -> None:
        log_bundle.start()
        emitter.start()
        await emit_provider_status(client, lifecycle)
        await emit_backend_status_snapshot(client, registry)
        if not first_connect.is_set():
            first_connect.set()
            logger.info(
                "talkies agent connected: backends=%d epoch=%s",
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
    # Serve the agent's single /v1 surface (incl. the Phase 24 /v1/audio/*
    # routes) on settings.PROVIDER_PORT; each request routes to a backend by
    # model (the request's `model` == the definition alias). One listener, no
    # per-backend port churn — a backend added by an assignments push is
    # routable immediately.
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
