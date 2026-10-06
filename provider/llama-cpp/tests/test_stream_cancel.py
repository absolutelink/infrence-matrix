"""Stream-cancel proof: abandoning our driver generator closes the upstream.

This is the property Phase 6's lease release depends on: when the consumer
of `stream_responses` / `stream_chat_completions` is aclosed early, the
upstream llama-server connection must be cancelled (not left draining),
and through the lifecycle, the slot must be released.

The fake llama-server marks `responses_cancelled` / `chat_cancelled` in
its state dir when the client hangs up mid-stream (BrokenPipe/
ConnectionReset while writing SSE frames).
"""

import time
from typing import Any

import pytest
from conftest import make_settings
from provider_lib.backend import BackendLifecycle
from provider_lib.wire import BackendStatusValue

from provider_llama_cpp.driver import LlamaCppBackend


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition not met in time")


def _driver(tmp_path: Any, fake_llama_binary: str, local_model_file: str):
    settings = make_settings(tmp_path, fake_llama_binary)
    return LlamaCppBackend(
        settings, {"artifacts": {"model": {"path": local_model_file}}}
    )


async def test_responses_stream_cancelled_on_early_aclose(
    tmp_path, fake_llama_binary, local_model_file, state_dir
) -> None:
    driver = _driver(tmp_path, fake_llama_binary, local_model_file)
    try:
        await driver.start()
        stream = driver.stream_responses({"input": "go"})
        first = await stream.__anext__()
        assert first["type"] == "response.created"
        second = await stream.__anext__()
        assert second["type"] == "response.output_text.delta"

        # Abandon mid-stream: closing the generator must close the
        # upstream response and cancel the llama request.
        await stream.aclose()
        _wait_for(lambda: (state_dir / "responses_cancelled").exists())
        assert not (state_dir / "stream_completed").exists()
    finally:
        await driver.aclose()


async def test_chat_stream_cancelled_on_early_aclose(
    tmp_path, fake_llama_binary, local_model_file, state_dir
) -> None:
    driver = _driver(tmp_path, fake_llama_binary, local_model_file)
    try:
        await driver.start()
        stream = driver.stream_chat_completions({"messages": []})
        await stream.__anext__()
        await stream.aclose()
        _wait_for(lambda: (state_dir / "chat_cancelled").exists())
    finally:
        await driver.aclose()


async def test_lifecycle_slot_released_when_consumer_abandons(
    tmp_path, fake_llama_binary, local_model_file, state_dir
) -> None:
    """End-to-end through BackendLifecycle: early consumer aclose ->
    pump cancelled -> upstream closed -> slot released -> RUNNING."""
    driver = _driver(tmp_path, fake_llama_binary, local_model_file)
    emitted: list[str] = []

    async def cb(status: str, _reason: str | None) -> None:
        emitted.append(status)

    lifecycle = BackendLifecycle(driver, capacity=1, status_callback=cb)
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

        # Slot release is driven by the upstream close; the fake server
        # observed the cancellation.
        _wait_for(lambda: (state_dir / "responses_cancelled").exists())
        _wait_for(lambda: lifecycle.in_flight == 0)
        assert lifecycle.backend_status == BackendStatusValue.RUNNING
    finally:
        await driver.aclose()


async def test_full_stream_not_marked_cancelled(
    tmp_path, fake_llama_binary, local_model_file, state_dir
) -> None:
    """Control: consuming to exhaustion completes upstream cleanly."""
    driver = _driver(tmp_path, fake_llama_binary, local_model_file)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
        assert events[-1]["type"] == "response.completed"
        _wait_for(lambda: (state_dir / "stream_completed").exists())
        assert not (state_dir / "responses_cancelled").exists()
    finally:
        await driver.aclose()
