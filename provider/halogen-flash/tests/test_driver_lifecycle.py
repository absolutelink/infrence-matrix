"""Driver + lifecycle tests against the FAKE halogen-flash subprocess.

Covers: start→healthy, native OpenResponses SSE passthrough with the
usage-normalization override applied to the terminal event, static-port
env wiring, stop, early-exit stderr tail, missing entrypoint error, and
stream cancellation.
"""

import json
import time
from typing import Any

import pytest
from conftest import NPU_AVAILABLE, NPU_UNAVAILABLE, make_settings
from provider_lib.backend import BackendLifecycle
from provider_lib.wire import BackendStatusValue

from provider_halogen_flash.driver import HalogenFlashBackend
from provider_halogen_flash.env import derive_static_ports


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition not met in time")


def _backend(
    tmp_path: Any,
    binary: str,
    artifacts: dict[str, str],
    settings_kw: dict[str, Any] | None = None,
    **kw: Any,
) -> HalogenFlashBackend:
    npu_probe = kw.pop("npu_probe", lambda: NPU_UNAVAILABLE)
    settings = make_settings(tmp_path, binary, **(settings_kw or {}))
    cfg: dict[str, Any] = {
        "artifacts": {
            "model": {"path": artifacts["checkpoint"]},
            "tokenizer": {"path": artifacts["tokenizer"]},
        },
    }
    cfg.update(kw)
    return HalogenFlashBackend(settings, cfg, npu_probe=npu_probe)


NPU_PINS = """
model qwen3-embedding-0.6b repo=acme/embed revision=v1
file qwen3-embedding-0.6b embed.bin 10 aa
model qwen3.5-2b repo=acme/nano revision=v2
file qwen3.5-2b nano.bin 10 bb
"""


def _npu_settings(tmp_path: Any) -> dict[str, Any]:
    """A pins file plus pre-placed local NPU artifacts (no downloads)."""
    pins = tmp_path / "npu_pins.txt"
    pins.write_text(NPU_PINS)
    for dir_id, name in (
        ("qwen3-embedding-0.6b", "embed.bin"),
        ("qwen3.5-2b", "nano.bin"),
    ):
        p = tmp_path / "models" / "npu" / dir_id / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return {"NPU_PINS_FILE": str(pins)}


async def test_start_healthy_running_via_lifecycle(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _backend(tmp_path, fake_flash_binary, local_artifacts)
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
    finally:
        await lifecycle.stop()
        await driver.aclose()
    assert lifecycle.backend_status == BackendStatusValue.STOPPED


async def test_static_ports_are_pinned_in_env(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    """The disk-cache fingerprint needs stable ports: derived from the
    MACHINE UID ('flash-test'), not from PROVIDER_PORT."""
    driver = _backend(tmp_path, fake_flash_binary, local_artifacts)
    expected_api, expected_engine = derive_static_ports("flash-test")
    try:
        await driver.start()
        assert driver.api_port == expected_api
        assert driver.engine_port == expected_engine
        recorded = json.loads((state_dir / "env.json").read_text())
        assert recorded["HALOGEN_API_PORT"] == str(expected_api)
        assert recorded["HALOGEN_PORT"] == str(expected_engine)
        assert recorded["HALOGEN_ENGINE"] == f"127.0.0.1:{expected_engine}"
    finally:
        await driver.aclose()


async def test_explicit_ports_override_derivation(
    tmp_path, fake_flash_binary, local_artifacts
) -> None:
    driver = _backend(
        tmp_path,
        fake_flash_binary,
        local_artifacts,
        networking={"api_port": 8250, "engine_port": 8251},
    )
    try:
        await driver.start()
        assert (driver.api_port, driver.engine_port) == (8250, 8251)
    finally:
        await driver.aclose()


async def test_stream_passthrough_normalizes_usage(
    tmp_path, fake_flash_binary, local_artifacts
) -> None:
    """The fake emits a chat-style terminal usage blob; the override must
    deliver a spec-shaped usage dict before the event is yielded."""
    driver = _backend(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        assert types.count("response.output_text.delta") == 8
        usage = events[-1]["response"]["usage"]
        # Normalized spec shape (fake reported prompt_tokens=100,
        # completion_tokens=8, timings.cache_n=80).
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 8
        assert usage["total_tokens"] == 108
        assert usage["input_tokens_details"] == {"cached_tokens": 80}
        assert usage["output_tokens_details"] == {"reasoning_tokens": 0}
        # No raw chat-style keys survive.
        assert "prompt_tokens" not in usage
        assert "timings" not in usage
    finally:
        await driver.aclose()


async def test_npu_env_set_when_probe_available(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _backend(
        tmp_path,
        fake_flash_binary,
        local_artifacts,
        settings_kw=_npu_settings(tmp_path),
        npu_probe=lambda: NPU_AVAILABLE,
        npu={"npu_models": ["qwen3-embedding-0.6b", "qwen3.5-2b"]},
    )
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert recorded["HALOGEN_NPU_MODELS"] == "qwen3-embedding-0.6b,qwen3.5-2b"
        # Phase 9: the pinned files were pre-downloaded (resolved locally)
        # and tracked in the prune reference set.
        assert any("npu/qwen3.5-2b/nano.bin" in p for p in driver.resolved_artifacts)
    finally:
        await driver.aclose()


async def test_npu_env_omitted_when_no_npu(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    """Graceful degradation: NPU requested but probe fails → start works,
    npu env omitted."""
    driver = _backend(
        tmp_path,
        fake_flash_binary,
        local_artifacts,
        npu_probe=lambda: NPU_UNAVAILABLE,
        npu={"npu_models": ["qwen3-embedding-0.6b"]},
    )
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert "HALOGEN_NPU_MODELS" not in recorded
    finally:
        await driver.aclose()


async def test_npu_probe_crash_degrades_gracefully(
    tmp_path, fake_flash_binary, local_artifacts
) -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("probe exploded")

    driver = _backend(
        tmp_path,
        fake_flash_binary,
        local_artifacts,
        npu_probe=boom,
        npu={"npu_models": ["qwen3.5-2b"]},
    )
    try:
        await driver.start()
        assert await driver.health() is True
    finally:
        await driver.aclose()


async def test_cache_dir_env_when_enabled(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    driver = _backend(
        tmp_path,
        fake_flash_binary,
        local_artifacts,
        disk_cache={"cache_dir_enabled": True},
    )
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert recorded["HALOGEN_CACHE_DIR"].endswith("halogen-flash")
    finally:
        await driver.aclose()


async def test_stop_terminates_process(
    tmp_path, fake_flash_binary, local_artifacts
) -> None:
    driver = _backend(tmp_path, fake_flash_binary, local_artifacts)
    await driver.start()
    proc = driver.process
    assert proc is not None and proc.poll() is None
    await driver.stop()
    assert proc.poll() is not None
    await driver.stop()  # idempotent


async def test_nonexistent_entrypoint_error(tmp_path, local_artifacts) -> None:
    settings = make_settings(tmp_path, "/nonexistent/flash-entrypoint")
    driver = HalogenFlashBackend(
        settings,
        {
            "model": {"path": local_artifacts["checkpoint"]},
            "tokenizer": {"path": local_artifacts["tokenizer"]},
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )
    with pytest.raises(RuntimeError, match="halogen-flash entrypoint not found"):
        await driver.start()


async def test_early_exit_includes_stderr_tail(
    tmp_path, early_exit_binary, local_artifacts
) -> None:
    settings = make_settings(tmp_path, early_exit_binary)
    settings.SERVER_START_HEALTH_TIMEOUT = 5
    driver = HalogenFlashBackend(
        settings,
        {
            "model": {"path": local_artifacts["checkpoint"]},
            "tokenizer": {"path": local_artifacts["tokenizer"]},
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )
    with pytest.raises(RuntimeError) as exc_info:
        await driver.start()
    message = str(exc_info.value)
    assert "exited with code 5" in message
    assert "npu attach failed" in message


async def test_blank_artifacts_defer_to_image_defaults(
    tmp_path, fake_flash_binary, state_dir
) -> None:
    """No `artifacts.model` / `artifacts.tokenizer`: nothing is resolved and
    neither variable is exported, so the image picks its own checkpoint
    (`resolve_checkpoint` under MODELS_DIR) and sidecar tokenizer
    (`need_tokenizer`) instead of being pinned to a missing path."""
    settings = make_settings(tmp_path, fake_flash_binary)
    driver = HalogenFlashBackend(settings, {}, npu_probe=lambda: NPU_UNAVAILABLE)
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert "HALOGEN_CHECKPOINT" not in recorded
        assert "HALOGEN_TOKENIZER" not in recorded
        # Nothing resolved => storage.prune_unused refuses to run (by design).
        assert driver.resolved_artifacts == []
    finally:
        await driver.aclose()
