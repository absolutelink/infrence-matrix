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
from provider_lib.wire import BackendStatusValue


@pytest.fixture
def settings(tmp_path: Any) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="cu-test",
        PROVIDER_REGISTRATION_TOKEN="tok",
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
