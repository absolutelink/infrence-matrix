"""Shared provider config.update handler tests (fake client + driver)."""

from typing import Any

import pytest
from config_fakes import RecordingClient, TrackDriver

from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import (
    ConfigState,
    install_config_handlers,
    prompt_cache_dir,
)
from provider_lib.models import ModelSpec
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.wire import BackendStatusValue


class _ModelTrackingDriver(TrackDriver):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_models_calls: list[list[ModelSpec]] = []

    def set_models(self, models: list[ModelSpec]) -> None:
        self.set_models_calls.append(list(models))


class _OrderTrackingDriver(_ModelTrackingDriver):
    """Records the relative order of apply_config / set_models / start."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []

    def apply_config(self, backend_config: dict[str, Any]) -> None:
        self.calls.append("apply_config")
        super().apply_config(backend_config)

    def set_models(self, models: list[ModelSpec]) -> None:
        self.calls.append("set_models")
        super().set_models(models)

    async def start(self) -> None:
        self.calls.append("start")
        await super().start()


@pytest.fixture
def settings(tmp_path: Any) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="cu-test",
        MACHINE_SECRET="tok",
        AGENT_ID="cu-test-agent",
        ADMIN_BASE_URL="http://localhost:9999",
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
    )


def _stack(
    settings: ProviderSettings,
    *,
    applied_fp: str | None = "fp-old",
    start_raises: bool = False,
    capacity: int = 1,
) -> tuple[RecordingClient, BackendLifecycle, TrackDriver, ConfigState]:
    client = RecordingClient(settings)
    driver = TrackDriver(start_raises=start_raises)
    lifecycle = BackendLifecycle(driver, capacity=capacity)
    state = ConfigState(applied_fp)
    install_config_handlers(client, lifecycle, state, settings)
    return client, lifecycle, driver, state


async def test_noop_when_fingerprint_matches(settings: Any) -> None:
    client, _lc, driver, state = _stack(settings, applied_fp="fp-same")
    ack = await client.dispatch(
        "provider.config.update",
        {"backend_config": {"a": 1}, "config_fingerprint": "fp-same"},
    )
    assert ack["ok"] is True
    assert ack["detail"]["noop"] is True
    assert driver.applied == []
    assert state.applied_fingerprint == "fp-same"


async def test_drain_refused_while_in_use(settings: Any) -> None:
    client, lifecycle, driver, _state = _stack(settings)
    await lifecycle.start()
    await lifecycle.acquire_slot()  # RUNNING -> IN_USE
    ack = await client.dispatch(
        "provider.config.update",
        {"backend_config": {"a": 1}, "config_fingerprint": "fp-new"},
    )
    assert ack["ok"] is False
    assert ack["error"] == "backend_in_use"
    assert ack["detail"]["step"] == "drain"
    assert ack["detail"]["retry_after"] == 10
    assert ack["detail"]["in_flight"] == 1
    # The backend was never stopped under the live stream.
    assert driver.stop_calls == 0
    assert lifecycle.backend_status == BackendStatusValue.IN_USE
    await lifecycle.release_slot()


async def test_same_fingerprint_acks_under_load(settings: Any) -> None:
    """SF-3 ordering: a same-fp update is safely ackable even while
    in_use (no stop needed -> no drain refusal, no churn)."""
    client, lifecycle, driver, _state = _stack(settings, applied_fp="fp-same")
    await lifecycle.start()
    await lifecycle.acquire_slot()
    ack = await client.dispatch(
        "provider.config.update",
        {"backend_config": {"a": 1}, "config_fingerprint": "fp-same"},
    )
    assert ack["ok"] is True
    assert ack["detail"]["noop"] is True
    assert ack["detail"]["capacity_adopted"] is False
    assert driver.stop_calls == 0
    assert lifecycle.backend_status == BackendStatusValue.IN_USE
    await lifecycle.release_slot()


async def test_capacity_only_change_adopted_without_restart(settings: Any) -> None:
    """SF-2: same fingerprint, different capacity -> adopted in place,
    backend stays running, noop ack with capacity_adopted."""
    client, lifecycle, driver, state = _stack(settings, applied_fp="fp-same")
    await lifecycle.start()
    ack = await client.dispatch(
        "provider.config.update",
        {
            "backend_config": {"a": 1},
            "config_fingerprint": "fp-same",
            "capacity": 5,
        },
    )
    assert ack["ok"] is True
    assert ack["detail"]["noop"] is True
    assert ack["detail"]["capacity_adopted"] is True
    assert ack["detail"]["capacity"] == 5
    assert lifecycle.capacity == 5
    # No stop/start/apply happened: the backend kept serving.
    assert driver.stop_calls == 0
    assert driver.applied == []
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    assert state.applied_fingerprint == "fp-same"


async def test_capacity_adopted_under_load_without_restart(settings: Any) -> None:
    """SF-2 + SF-3: capacity-only PATCH while in_use is adopted (no
    drain refusal, no restart)."""
    client, lifecycle, driver, _state = _stack(settings, applied_fp="fp-same")
    await lifecycle.start()
    await lifecycle.acquire_slot()
    ack = await client.dispatch(
        "provider.config.update",
        {
            "backend_config": {"a": 1},
            "config_fingerprint": "fp-same",
            "capacity": 3,
        },
    )
    assert ack["ok"] is True
    assert ack["detail"]["capacity_adopted"] is True
    assert lifecycle.capacity == 3
    assert driver.stop_calls == 0
    await lifecycle.release_slot()


async def test_capacity_same_value_is_pure_noop(settings: Any) -> None:
    client, lifecycle, _driver, _state = _stack(
        settings, applied_fp="fp-same", capacity=2
    )
    await lifecycle.start()
    ack = await client.dispatch(
        "provider.config.update",
        {
            "backend_config": {"a": 1},
            "config_fingerprint": "fp-same",
            "capacity": 2,
        },
    )
    assert ack["ok"] is True
    assert ack["detail"]["noop"] is True
    assert ack["detail"]["capacity_adopted"] is False
    assert lifecycle.capacity == 2


async def test_fingerprint_change_applies_and_acks(settings: Any) -> None:
    client, lifecycle, driver, state = _stack(settings)
    # Seed old + new fingerprint prompt-cache dirs and a model file.
    old_dir = prompt_cache_dir(settings, "fp-old")
    old_dir.mkdir(parents=True)
    (old_dir / "cache.bin").write_bytes(b"x" * 100)
    models_file = settings.MODELS_DIR / "model.gguf"
    models_file.parent.mkdir(parents=True, exist_ok=True)
    models_file.write_bytes(b"model")

    ack = await client.dispatch(
        "provider.config.update",
        {
            "backend_config": {"delta_count": 7},
            "config_fingerprint": "fp-new",
            "capacity": 3,
        },
    )
    assert ack["ok"] is True
    assert ack["detail"]["config_fingerprint"] == "fp-new"
    assert ack["detail"]["capacity"] == 3
    assert ack["detail"]["model_metadata"] == [{"id": "track-model", "object": "model"}]
    # Old prompt cache deleted; model file untouched.
    assert not old_dir.exists()
    assert models_file.exists()
    # apply_config called with the new config; backend started; capacity adopted.
    assert driver.applied == [{"delta_count": 7}]
    assert lifecycle.capacity == 3
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    # Status transitions: initializing -> running.
    provider_statuses = [
        e[1].get("instance_status") for e in client.events if e[0] == "provider.status"
    ]
    assert provider_statuses == ["initializing", "running"]
    assert state.applied_fingerprint == "fp-new"


async def test_failure_at_start_naks_with_step(settings: Any) -> None:
    client, lifecycle, _driver, state = _stack(settings, start_raises=True)
    ack = await client.dispatch(
        "provider.config.update",
        {"backend_config": {"a": 1}, "config_fingerprint": "fp-new"},
    )
    assert ack["ok"] is False
    assert ack["detail"]["step"] == "start"
    assert "spawn failed" in ack["error"]
    # Fingerprint NOT committed on failure.
    assert state.applied_fingerprint == "fp-old"
    # Error status emitted.
    last = [e for e in client.events if e[0] == "provider.status"][-1]
    assert last[1]["instance_status"] == "error"
    assert lifecycle.backend_status == BackendStatusValue.ERROR


async def test_invalid_payload_rejected(settings: Any) -> None:
    client, _lc, _driver, _state = _stack(settings)
    ack = await client.dispatch(
        "provider.config.update", {"config_fingerprint": "fp-new"}
    )
    assert ack["ok"] is False
    assert ack["detail"]["step"] == "validate"
    ack = await client.dispatch(
        "provider.config.update", {"backend_config": {}, "config_fingerprint": ""}
    )
    assert ack["ok"] is False


async def test_update_from_stopped_skips_stop(settings: Any) -> None:
    """A never-started backend goes straight through apply+start."""
    client, lifecycle, driver, _state = _stack(settings)
    assert lifecycle.backend_status == BackendStatusValue.STOPPED
    ack = await client.dispatch(
        "provider.config.update",
        {"backend_config": {"a": 1}, "config_fingerprint": "fp-x"},
    )
    assert ack["ok"] is True
    assert driver.stop_calls == 0  # nothing to stop
    assert driver.applied == [{"a": 1}]
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_repeated_update_stops_previous_backend(settings: Any) -> None:
    client, lifecycle, driver, _state = _stack(settings)
    await lifecycle.start()
    ack = await client.dispatch(
        "provider.config.update",
        {"backend_config": {"a": 2}, "config_fingerprint": "fp-y"},
    )
    assert ack["ok"] is True
    assert driver.stop_calls == 1
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_config_update_sets_driver_modality(settings: Any) -> None:
    """Phase 18 slice 3: a config.update carrying modality stamps it on the
    driver and the applied-config state before apply_config runs."""
    client, _lc, driver, state = _stack(settings)
    ack = await client.dispatch(
        "provider.config.update",
        {
            "backend_config": {"a": 1},
            "config_fingerprint": "fp-new",
            "modality": "embedding",
        },
    )
    assert ack["ok"] is True
    assert driver.modality == "embedding"
    assert state.modality == "embedding"


async def test_config_update_absent_modality_defaults_llm(settings: Any) -> None:
    """Phase 18 slice 3: an absent modality coerces the driver back to "llm"."""
    client, _lc, driver, state = _stack(settings)
    driver.modality = "embedding"  # prove absent overrides a prior value
    ack = await client.dispatch(
        "provider.config.update",
        {"backend_config": {"a": 1}, "config_fingerprint": "fp-new"},
    )
    assert ack["ok"] is True
    assert driver.modality == "llm"
    assert state.modality == "llm"


async def test_config_update_adopts_modality_on_noop(settings: Any) -> None:
    """Phase 18 slice 3: modality is adopted even on a same-fingerprint noop
    (the admin may push a modality-only change; the driver must reflect it)."""
    client, _lc, driver, _state = _stack(settings, applied_fp="fp-same")
    ack = await client.dispatch(
        "provider.config.update",
        {
            "backend_config": {"a": 1},
            "config_fingerprint": "fp-same",
            "modality": "embedding",
        },
    )
    assert ack["ok"] is True
    assert ack["detail"]["noop"] is True
    assert driver.modality == "embedding"


async def test_config_update_threads_served_models(settings: Any) -> None:
    """Phase 25: a config.update carrying served_models refreshes the handle +
    driver before the restart so the booted backend serves the new names."""
    client = RecordingClient(settings)
    driver = _ModelTrackingDriver()
    lifecycle = BackendLifecycle(driver, instance_id="i1")
    registry = BackendRegistry()
    handle = BackendHandle("i1", lifecycle, ConfigState("fp-old"), alias="old")
    registry.add(handle)
    install_config_handlers(client, registry, handle.config_state, settings)

    ack = await client.dispatch(
        "provider.config.update",
        {
            "instance_id": "i1",
            "backend_config": {"a": 1},
            "config_fingerprint": "fp-new",
            "served_models": [{"name": "x", "modality": "tts"}, {"name": "y"}],
        },
    )
    assert ack["ok"] is True
    assert [m.name for m in handle.models] == ["x", "y"]
    assert registry.resolve_by_model("x") is handle
    assert registry.resolve_by_model("y") is handle
    assert registry.resolve_by_model("old") is None  # alias re-derived to "x"
    assert driver.set_models_calls == [
        [ModelSpec(name="x", modality="tts"), ModelSpec(name="y", modality="llm")]
    ]


async def test_config_update_without_served_models_leaves_models(settings: Any) -> None:
    """A config.update with no served_models must NOT clobber the handle's list
    (models_from_entry finds no alias in the payload -> [])."""
    client = RecordingClient(settings)
    driver = _ModelTrackingDriver()
    lifecycle = BackendLifecycle(driver, instance_id="i1")
    registry = BackendRegistry()
    handle = BackendHandle(
        "i1", lifecycle, ConfigState("fp-old"), alias="keep", models=[ModelSpec("keep")]
    )
    registry.add(handle)
    install_config_handlers(client, registry, handle.config_state, settings)

    ack = await client.dispatch(
        "provider.config.update",
        {
            "instance_id": "i1",
            "backend_config": {"a": 1},
            "config_fingerprint": "fp-new",
        },
    )
    assert ack["ok"] is True
    assert [m.name for m in handle.models] == ["keep"]
    assert driver.set_models_calls == []


async def test_config_update_set_models_before_start(settings: Any) -> None:
    """(d) ordering: the served-model refresh must land BEFORE the restart so the
    booted backend comes up already serving the new names."""
    client = RecordingClient(settings)
    driver = _OrderTrackingDriver()
    lifecycle = BackendLifecycle(driver, instance_id="i1")
    registry = BackendRegistry()
    handle = BackendHandle("i1", lifecycle, ConfigState("fp-old"), alias="old")
    registry.add(handle)
    install_config_handlers(client, registry, handle.config_state, settings)

    ack = await client.dispatch(
        "provider.config.update",
        {
            "instance_id": "i1",
            "backend_config": {"a": 1},
            "config_fingerprint": "fp-new",
            "served_models": [{"name": "x"}, {"name": "y"}],
        },
    )
    assert ack["ok"] is True
    assert "set_models" in driver.calls
    assert driver.calls.index("set_models") < driver.calls.index("start")
    assert driver.calls.index("apply_config") < driver.calls.index("set_models")
