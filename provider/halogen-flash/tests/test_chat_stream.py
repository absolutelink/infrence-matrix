"""Chat-completions stream tests for the halogen-flash driver.

The API port speaks OpenAI chat SSE natively; the driver must relay the
parsed chunks (forcing ``stream: true``, skipping the ``[DONE]``
sentinel), and closing the stream must cancel the upstream request so
the lifecycle frees the slot on upstream close.
"""

import json
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
            "artifacts": {
                "model": {"path": artifacts["checkpoint"]},
                "tokenizer": {"path": artifacts["tokenizer"]},
            },
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )


async def test_chat_stream_relays_chunks_and_skips_done(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        chunks = [
            chunk async for chunk in driver.stream_chat_completions({"messages": []})
        ]
    finally:
        await driver.aclose()
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c["choices"] and c["choices"][0]["delta"].get("content") is not None
    )
    assert text == "".join(f"tok{i}" for i in range(8))
    finish = [
        c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")
    ]
    assert finish[-1]["choices"][0]["finish_reason"] == "stop"
    # Final usage chunk: OpenAI streaming shape (empty choices + usage).
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["prompt_tokens"] == 100
    _wait_for(lambda: (state_dir / "stream_completed").exists())


async def test_chat_stream_upstream_error_raises(
    tmp_path, fake_flash_binary, local_artifacts, monkeypatch
) -> None:
    """A non-200 from the API port surfaces as a RuntimeError with the
    upstream detail (the provider app maps it to an SSE error frame)."""
    monkeypatch.setenv("FAKE_CHAT_STATUS", "501")
    settings = make_settings(tmp_path, fake_flash_binary)
    driver = HalogenFlashBackend(
        settings,
        {
            "artifacts": {
                "model": {"path": local_artifacts["checkpoint"]},
                "tokenizer": {"path": local_artifacts["tokenizer"]},
            },
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )
    try:
        await driver.start()
        with pytest.raises(RuntimeError, match="HTTP 501"):
            async for _ in driver.stream_chat_completions({"messages": []}):
                pass
    finally:
        await driver.aclose()


async def test_chat_stream_forces_stream_true(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        agen = driver.stream_chat_completions(
            {"messages": [{"role": "user", "content": "go"}], "stream": False}
        )
        first = await agen.__anext__()
        assert first["object"] == "chat.completion.chunk"
        await agen.aclose()
    finally:
        await driver.aclose()
    with open(state_dir / "last_chat_request.json") as fh:
        forwarded = json.load(fh)
    assert forwarded["stream"] is True
    assert forwarded["messages"] == [{"role": "user", "content": "go"}]


async def test_chat_stream_cancelled_on_early_aclose(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        stream = driver.stream_chat_completions({"messages": []})
        first = await stream.__anext__()
        assert first["object"] == "chat.completion.chunk"
        await stream.aclose()
        _wait_for(lambda: (state_dir / "chat_cancelled").exists())
        assert not (state_dir / "stream_completed").exists()
    finally:
        await driver.aclose()


async def test_lifecycle_chat_slot_released_when_consumer_abandons(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    lifecycle = BackendLifecycle(driver, capacity=1)
    try:
        await lifecycle.start()
        stream = await lifecycle.stream_chat_completions({"messages": []})
        assert lifecycle.in_flight == 1
        assert lifecycle.backend_status == BackendStatusValue.IN_USE
        got = 0
        async for _chunk in stream:
            got += 1
            if got >= 2:
                break
        await stream.aclose()
        _wait_for(lambda: (state_dir / "chat_cancelled").exists())
        _wait_for(lambda: lifecycle.in_flight == 0)
        assert lifecycle.backend_status == BackendStatusValue.RUNNING
    finally:
        await driver.aclose()
