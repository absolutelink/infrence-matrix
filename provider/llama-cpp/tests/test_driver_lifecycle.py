"""Driver + lifecycle tests against the FAKE llama-server subprocess.

The fake server (tests/conftest.py) is spawned exactly like a real
llama-server: the driver Popen's the wrapper executable with the built
command line, polls /health, and proxies /v1 SSE streams. The fake
records markers (stream completed / cancelled) in a state directory so
we can assert real socket-level cancellation.
"""

import asyncio
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


def _backend(
    tmp_path: Any, binary: str, model: str, **cfg_extra: Any
) -> LlamaCppBackend:
    settings = make_settings(tmp_path, binary)
    # Phase 12 sectioned shape: artifacts under `artifacts`.
    cfg: dict[str, Any] = {"artifacts": {"model": {"path": model}}}
    cfg.update(cfg_extra)
    return LlamaCppBackend(settings, cfg)


async def test_start_healthy_running_via_lifecycle(
    tmp_path, fake_llama_binary, local_model_file, state_dir
) -> None:
    driver = _backend(tmp_path, fake_llama_binary, local_model_file)
    emitted: list[str] = []

    async def cb(status: str, _reason: str | None) -> None:
        emitted.append(status)

    lifecycle = BackendLifecycle(driver, capacity=1, status_callback=cb)
    try:
        await lifecycle.start()
        assert lifecycle.backend_status == BackendStatusValue.RUNNING
        assert emitted == [
            BackendStatusValue.STARTING,
            BackendStatusValue.RUNNING,
        ]
        assert driver.process is not None and driver.process.poll() is None
        assert (state_dir / "server_started").exists()
        assert await driver.health() is True
    finally:
        await lifecycle.stop()
        await driver.aclose()
    assert driver.process is None
    assert lifecycle.backend_status == BackendStatusValue.STOPPED


async def test_list_models_passthrough(
    tmp_path, fake_llama_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_llama_binary, local_model_file)
    try:
        await driver.start()
        models = await driver.list_models()
        assert models == [
            {"id": "fake-llama-model", "object": "model", "owned_by": "fake"}
        ]
    finally:
        await driver.aclose()


async def test_stream_responses_yields_event_dicts_with_usage(
    tmp_path, fake_llama_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_llama_binary, local_model_file)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        assert types.count("response.output_text.delta") == 8
        usage = events[-1]["response"]["usage"]
        assert usage["total_tokens"] == 11
    finally:
        await driver.aclose()


async def test_stream_chat_completions_skips_done_sentinel(
    tmp_path, fake_llama_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_llama_binary, local_model_file)
    try:
        await driver.start()
        chunks = [c async for c in driver.stream_chat_completions({"messages": []})]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        # The [DONE] sentinel was not yielded as a chunk.
        assert len(chunks) == 9
    finally:
        await driver.aclose()


async def test_stop_terminates_process(
    tmp_path, fake_llama_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_llama_binary, local_model_file)
    await driver.start()
    proc = driver.process
    assert proc is not None and proc.poll() is None
    await driver.stop()
    assert proc.poll() is not None
    # Stop is idempotent.
    await driver.stop()


async def test_missing_local_model_file_raises(tmp_path, fake_llama_binary) -> None:
    driver = _backend(tmp_path, fake_llama_binary, "/definitely/not/here.gguf")
    with pytest.raises(Exception, match="not found"):
        await driver.start()


async def test_nonexistent_binary_error_message(tmp_path, local_model_file) -> None:
    settings = make_settings(tmp_path, "/nonexistent/llama-server-binary")
    driver = LlamaCppBackend(
        settings, {"artifacts": {"model": {"path": local_model_file}}}
    )
    with pytest.raises(RuntimeError, match="failed to spawn llama-server"):
        await driver.start()


async def test_early_exit_includes_stderr_tail(
    tmp_path, early_exit_binary, local_model_file
) -> None:
    settings = make_settings(tmp_path, early_exit_binary)
    settings.SERVER_START_HEALTH_TIMEOUT = 5
    driver = LlamaCppBackend(
        settings, {"artifacts": {"model": {"path": local_model_file}}}
    )
    with pytest.raises(RuntimeError) as exc_info:
        await driver.start()
    message = str(exc_info.value)
    assert "exited with code 3" in message
    # Legacy pattern: last stderr lines are included for diagnosis.
    assert "failed to initialize backend" in message
    assert "fake: loading model" in message


async def test_get_logs_ring(tmp_path, fake_llama_binary, local_model_file) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
    driver = LlamaCppBackend(
        settings, {"artifacts": {"model": {"path": local_model_file}}}
    )
    try:
        await driver.start()
        await asyncio.sleep(0.1)
        driver._append_log("stderr", "hello ring")
        logs = driver.get_logs(10)
        assert isinstance(logs, list)
        assert logs, "recent entries should be present"
        # Phase 13: ring entries carry ts/stream/text.
        assert {"stream", "text"} <= set(logs[-1])
        assert logs[-1]["text"] == "hello ring"
        assert driver.log_ring.last_cursor >= len(driver.log_ring)
    finally:
        await driver.aclose()
