"""Stream-cancel proof for the gufo driver.

When the consumer of `stream_responses` is aclosed early, the upstream
gufo connection must be cancelled, and through the lifecycle the slot
must be released. The fake marks `responses_cancelled` when the client
hangs up mid-stream.
"""

import time
from typing import Any

import pytest
from conftest import make_settings
from provider_lib.backend import BackendLifecycle
from provider_lib.wire import BackendStatusValue

from provider_gufo.driver import GufoBackend


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition not met in time")


def _driver(tmp_path: Any, fake_gufo_binary: str, local_model_file: str):
    settings = make_settings(tmp_path, fake_gufo_binary)
    return GufoBackend(settings, {"model": {"path": local_model_file}})


async def test_responses_stream_cancelled_on_early_aclose(
    tmp_path, fake_gufo_binary, local_model_file, state_dir
) -> None:
    driver = _driver(tmp_path, fake_gufo_binary, local_model_file)
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
    tmp_path, fake_gufo_binary, local_model_file, state_dir
) -> None:
    driver = _driver(tmp_path, fake_gufo_binary, local_model_file)
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
    tmp_path, fake_gufo_binary, local_model_file, state_dir
) -> None:
    driver = _driver(tmp_path, fake_gufo_binary, local_model_file)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
        assert events[-1]["type"] == "response.completed"
        _wait_for(lambda: (state_dir / "stream_completed").exists())
        assert not (state_dir / "responses_cancelled").exists()
    finally:
        await driver.aclose()
