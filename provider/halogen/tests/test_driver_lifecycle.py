"""Driver + lifecycle tests against the FAKE halogen subprocess.

The fake takes its port from HALOGEN_API_PORT (env-driven, like the real
engine) and records its HALOGEN_* environment, so these tests verify
the backend_config → env mapping end to end through a real spawn, plus
start→healthy, streaming, stop, early-exit, and missing-binary errors.
"""

import json
import time
from typing import Any

import pytest
from conftest import make_settings
from provider_lib.backend import BackendLifecycle
from provider_lib.wire import BackendStatusValue

from provider_halogen.driver import HalogenBackend


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition not met in time")


def _backend(
    tmp_path: Any, binary: str, artifacts: dict[str, str], **cfg_extra: Any
) -> HalogenBackend:
    settings = make_settings(tmp_path, binary)
    cfg: dict[str, Any] = {
        "model": {"path": artifacts["checkpoint"]},
        "tokenizer": {"path": artifacts["tokenizer"]},
    }
    cfg.update(cfg_extra)
    return HalogenBackend(settings, cfg)


async def test_start_healthy_running_via_lifecycle(
    tmp_path, fake_halogen_binary, local_artifacts, state_dir
) -> None:
    driver = _backend(tmp_path, fake_halogen_binary, local_artifacts)
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
        assert (state_dir / "server_started").exists()
        assert await driver.health() is True
        # Two ports: api = PROVIDER_PORT+1, engine = PROVIDER_PORT+2.
        assert driver.api_port == 8182
        assert driver.engine_port == 8183
    finally:
        await lifecycle.stop()
        await driver.aclose()
    assert lifecycle.backend_status == BackendStatusValue.STOPPED


async def test_env_mapping_through_real_spawn(
    tmp_path, fake_halogen_binary, local_artifacts, state_dir
) -> None:
    driver = _backend(
        tmp_path,
        fake_halogen_binary,
        local_artifacts,
        options={"kv_slots": 4, "drafter": "mtp", "cache_mb": 2048},
    )
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert recorded["HALOGEN_KV_SLOTS"] == "4"
        assert recorded["HALOGEN_DRAFTER"] == "mtp"
        assert recorded["HALOGEN_CACHE_MB"] == "2048"
        assert recorded["HALOGEN_API_PORT"] == "8182"
        assert recorded["HALOGEN_ENGINE"] == "127.0.0.1:8183"
        assert recorded["HALOGEN_CHECKPOINT"] == local_artifacts["checkpoint"]
        assert recorded["HALOGEN_TOKENIZER"] == local_artifacts["tokenizer"]
    finally:
        await driver.aclose()


async def test_explicit_ports_from_config(
    tmp_path, fake_halogen_binary, local_artifacts
) -> None:
    driver = _backend(
        tmp_path,
        fake_halogen_binary,
        local_artifacts,
        api_port=9401,
        engine_port=9402,
    )
    try:
        await driver.start()
        assert driver.api_port == 9401
        assert driver.engine_port == 9402
    finally:
        await driver.aclose()


async def test_stream_responses_passthrough(
    tmp_path, fake_halogen_binary, local_artifacts
) -> None:
    driver = _backend(tmp_path, fake_halogen_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        assert types.count("response.output_text.delta") == 8
        assert events[-1]["response"]["usage"]["total_tokens"] == 13
    finally:
        await driver.aclose()


async def test_stream_chat_completions(
    tmp_path, fake_halogen_binary, local_artifacts
) -> None:
    driver = _backend(tmp_path, fake_halogen_binary, local_artifacts)
    try:
        await driver.start()
        chunks = [c async for c in driver.stream_chat_completions({"messages": []})]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert len(chunks) == 9  # [DONE] sentinel not yielded
    finally:
        await driver.aclose()


async def test_list_models(tmp_path, fake_halogen_binary, local_artifacts) -> None:
    driver = _backend(tmp_path, fake_halogen_binary, local_artifacts)
    try:
        await driver.start()
        models = await driver.list_models()
        assert models[0]["id"] == "halogen-qwen3.8-27b"
    finally:
        await driver.aclose()


async def test_stop_terminates_process(
    tmp_path, fake_halogen_binary, local_artifacts
) -> None:
    driver = _backend(tmp_path, fake_halogen_binary, local_artifacts)
    await driver.start()
    proc = driver.process
    assert proc is not None and proc.poll() is None
    await driver.stop()
    assert proc.poll() is not None
    await driver.stop()  # idempotent


async def test_missing_checkpoint_raises(tmp_path, fake_halogen_binary) -> None:
    settings = make_settings(tmp_path, fake_halogen_binary)
    driver = HalogenBackend(settings, {"tokenizer": {"path": "/nope"}})
    with pytest.raises(Exception, match="missing 'model'"):
        await driver.start()


async def test_nonexistent_binary_error_message(tmp_path, local_artifacts) -> None:
    settings = make_settings(tmp_path, "/nonexistent/halogen-entrypoint")
    driver = HalogenBackend(
        settings,
        {
            "model": {"path": local_artifacts["checkpoint"]},
            "tokenizer": {"path": local_artifacts["tokenizer"]},
        },
    )
    with pytest.raises(RuntimeError, match="halogen entrypoint not found"):
        await driver.start()


async def test_early_exit_includes_stderr_tail(
    tmp_path, early_exit_binary, local_artifacts
) -> None:
    settings = make_settings(tmp_path, early_exit_binary)
    settings.SERVER_START_HEALTH_TIMEOUT = 5
    driver = HalogenBackend(
        settings,
        {
            "model": {"path": local_artifacts["checkpoint"]},
            "tokenizer": {"path": local_artifacts["tokenizer"]},
        },
    )
    with pytest.raises(RuntimeError) as exc_info:
        await driver.start()
    message = str(exc_info.value)
    assert "exited with code 4" in message
    # stderr is merged into stdout; the tail is still captured.
    assert "rocm init failed" in message


def test_effective_capacity(tmp_path, fake_halogen_binary, local_artifacts) -> None:
    driver = _backend(
        tmp_path, fake_halogen_binary, local_artifacts, options={"kv_slots": 6}
    )
    assert driver.effective_capacity == 6
