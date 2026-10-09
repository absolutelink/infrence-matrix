"""Driver + lifecycle tests against the FAKE gufo subprocess.

The fake server (tests/conftest.py) is spawned exactly like a real
``gufo serve llm``: the driver Popen's the wrapper with the built argv,
polls /health, and proxies native OpenResponses SSE. The fake records
markers + the last request body in its state dir so tests can assert
multi-model forwarding and rate-gauge enrichment.
"""

import json
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


def _backend(tmp_path: Any, binary: str, model: str, **cfg_extra: Any) -> GufoBackend:
    settings = make_settings(tmp_path, binary)
    cfg: dict[str, Any] = {"artifacts": {"model": {"path": model}}}
    cfg.update(cfg_extra)
    return GufoBackend(settings, cfg)


async def test_start_healthy_running_via_lifecycle(
    tmp_path, fake_gufo_binary, local_model_file, state_dir
) -> None:
    driver = _backend(tmp_path, fake_gufo_binary, local_model_file)
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


async def test_list_models_multi_model(
    tmp_path, fake_gufo_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_gufo_binary, local_model_file)
    try:
        await driver.start()
        models = await driver.list_models()
        assert [m["id"] for m in models] == ["served-alpha", "served-beta"]
    finally:
        await driver.aclose()


async def test_per_request_model_forwarded_upstream(
    tmp_path, fake_gufo_binary, local_model_file, state_dir
) -> None:
    """gufo multi-model: the request's `model` must reach the backend."""
    driver = _backend(tmp_path, fake_gufo_binary, local_model_file)
    try:
        await driver.start()
        events = [
            e
            async for e in driver.stream_responses(
                {"model": "served-beta", "input": "go"}
            )
        ]
        assert events[-1]["type"] == "response.completed"
        recorded = json.loads((state_dir / "last_request.json").read_text())
        assert recorded["path"] == "/v1/responses"
        assert recorded["body"]["model"] == "served-beta"
        assert recorded["body"]["stream"] is True
    finally:
        await driver.aclose()


async def test_terminal_usage_enriched_with_rate_gauges(
    tmp_path, fake_gufo_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_gufo_binary, local_model_file)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
        usage = events[-1]["response"]["usage"]
        assert usage["input_tokens"] == 3
        assert usage["output_tokens"] == 8
        # The fake's /metrics gauges were scraped and merged before the
        # terminal event was yielded.
        assert usage["completion_tokens_details"]["prompt_per_second"] == 1412.21
        assert usage["completion_tokens_details"]["predicted_per_second"] == 36.2534
    finally:
        await driver.aclose()


async def test_terminal_usage_enriched_with_latency(
    tmp_path, fake_gufo_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_gufo_binary, local_model_file)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
        usage = events[-1]["response"]["usage"]
        # The fake's native `timings` (ms) were mapped to
        # completion_tokens_details latency (seconds) before the terminal
        # event was yielded.
        assert usage["completion_tokens_details"]["prompt_time"] == 0.1205
        assert usage["completion_tokens_details"]["prediction_time"] == 3.4
    finally:
        await driver.aclose()


async def test_terminal_usage_latency_from_usage_nested_timings(
    tmp_path, fake_gufo_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_gufo_binary, local_model_file)
    try:
        await driver.start()
        events = [
            e
            async for e in driver.stream_responses(
                {"input": "go", "__timings_in_usage__": True}
            )
        ]
        usage = events[-1]["response"]["usage"]
        # timings nested inside usage (not beside it) must still map to
        # completion_tokens_details latency via the driver's fallback.
        assert "timings" not in events[-1]["response"]
        assert usage["completion_tokens_details"]["prompt_time"] == 0.1205
        assert usage["completion_tokens_details"]["prediction_time"] == 3.4
    finally:
        await driver.aclose()


async def test_stop_terminates_process(
    tmp_path, fake_gufo_binary, local_model_file
) -> None:
    driver = _backend(tmp_path, fake_gufo_binary, local_model_file)
    await driver.start()
    proc = driver.process
    assert proc is not None and proc.poll() is None
    await driver.stop()
    assert proc.poll() is not None
    await driver.stop()  # idempotent


async def test_backend_port_none_until_start_and_after_stop(
    tmp_path, fake_gufo_binary, local_model_file
) -> None:
    """Port model overhaul: the engine port is OS-assigned at start and cleared
    on stop, so the "None until start" invariant holds across a boot cycle."""
    driver = _backend(tmp_path, fake_gufo_binary, local_model_file)
    assert driver.backend_port is None
    try:
        await driver.start()
        assert isinstance(driver.backend_port, int) and driver.backend_port > 0
    finally:
        await driver.stop()
    assert driver.backend_port is None
    await driver.aclose()


async def test_missing_local_model_file_raises(tmp_path, fake_gufo_binary) -> None:
    driver = _backend(tmp_path, fake_gufo_binary, "/definitely/not/here.gguf")
    with pytest.raises(Exception, match="not found"):
        await driver.start()


async def test_nonexistent_binary_error_message(tmp_path, local_model_file) -> None:
    settings = make_settings(tmp_path, "/nonexistent/gufo-binary")
    driver = GufoBackend(settings, {"model": {"path": local_model_file}})
    with pytest.raises(RuntimeError, match="failed to spawn gufo"):
        await driver.start()


async def test_early_exit_includes_stderr_tail(
    tmp_path, early_exit_binary, local_model_file
) -> None:
    settings = make_settings(tmp_path, early_exit_binary)
    settings.SERVER_START_HEALTH_TIMEOUT = 5
    driver = GufoBackend(settings, {"model": {"path": local_model_file}})
    with pytest.raises(RuntimeError) as exc_info:
        await driver.start()
    message = str(exc_info.value)
    assert "exited with code 3" in message
    assert "failed to initialize device" in message
    assert "fake-gufo: loading model" in message


async def test_aux_descriptor_resolved_to_local_path(
    tmp_path, fake_gufo_binary, local_model_file
) -> None:
    """Aux dict descriptors (mmproj etc.) download/resolve before spawn."""
    aux = tmp_path / "models" / "vision.gguf"
    aux.write_bytes(b"mmproj-fake")
    driver = _backend(
        tmp_path,
        fake_gufo_binary,
        local_model_file,
        artifacts={"model": {"path": local_model_file}, "mmproj": {"path": str(aux)}},
    )
    try:
        await driver.start()  # would raise if resolution failed
        assert await driver.health() is True
    finally:
        await driver.aclose()


async def test_effective_capacity_from_config(
    tmp_path, fake_gufo_binary, local_model_file
) -> None:
    driver = _backend(
        tmp_path, fake_gufo_binary, local_model_file, server={"sessions": 4}
    )
    assert driver.effective_capacity == 4
