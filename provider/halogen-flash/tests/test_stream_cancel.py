"""Stream-cancel proof for the halogen-flash driver."""

import time
from typing import Any

import pytest
from conftest import NPU_UNAVAILABLE, make_settings
from provider_lib.backend import BackendLifecycle
from provider_lib.wire import BackendStatusValue

from provider_halogen_flash.driver import HalogenFlashBackend


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition not met in time")


def _driver(tmp_path: Any, binary: str, artifacts: dict[str, str]):
    settings = make_settings(tmp_path, binary)
    return HalogenFlashBackend(
        settings,
        {
            "model": {"path": artifacts["checkpoint"]},
            "tokenizer": {"path": artifacts["tokenizer"]},
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )


async def test_responses_stream_cancelled_on_early_aclose(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        stream = driver.stream_responses({"input": "go"})
        first = await stream.__anext__()
        assert first["type"] == "response.created"
        await stream.aclose()
        _wait_for(lambda: (state_dir / "responses_cancelled").exists())
        assert not (state_dir / "stream_completed").exists()
    finally:
        await driver.aclose()


async def test_lifecycle_slot_released_when_consumer_abandons(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    lifecycle = BackendLifecycle(driver, capacity=1)
    try:
        await lifecycle.start()
        stream = await lifecycle.stream_responses({"input": "go"})
        assert lifecycle.in_flight == 1
        assert lifecycle.backend_status == BackendStatusValue.IN_USE
        got = 0
        async for _event in stream:
            got += 1
            if got >= 2:
                break
        await stream.aclose()
        _wait_for(lambda: (state_dir / "responses_cancelled").exists())
        _wait_for(lambda: lifecycle.in_flight == 0)
        assert lifecycle.backend_status == BackendStatusValue.RUNNING
    finally:
        await driver.aclose()


async def test_full_stream_not_marked_cancelled(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
        assert events[-1]["type"] == "response.completed"
        _wait_for(lambda: (state_dir / "stream_completed").exists())
        assert not (state_dir / "responses_cancelled").exists()
    finally:
        await driver.aclose()
