"""Driver + lifecycle tests against the FAKE halogen-flash subprocess.

Covers: start→healthy, native OpenResponses SSE passthrough with the
usage-normalization override applied to the terminal event, static-port
env wiring, stop, early-exit stderr tail, missing entrypoint error, and
stream cancellation.
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import NPU_AVAILABLE, NPU_UNAVAILABLE, make_settings
from provider_lib.backend import BackendLifecycle
from provider_lib.wire import BackendStatusValue

from provider_halogen_flash.driver import HalogenFlashBackend
from provider_halogen_flash.env import DEFAULT_DOWNLOAD_REPO, derive_static_ports


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
    """No `artifacts.model` / `artifacts.tokenizer`: the provider downloads
    NOTHING and exports neither variable, so the image picks its own
    checkpoint (`resolve_checkpoint`) and sidecar tokenizer
    (`need_tokenizer`) — and `HALOGEN_DOWNLOAD` is still set, which is what
    lets the engine fetch that default head into MODELS_DIR on a cold box."""
    settings = make_settings(tmp_path, fake_flash_binary)
    driver = HalogenFlashBackend(settings, {}, npu_probe=lambda: NPU_UNAVAILABLE)
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert "HALOGEN_CHECKPOINT" not in recorded
        assert "HALOGEN_TOKENIZER" not in recorded
        assert recorded["HALOGEN_DOWNLOAD"] == DEFAULT_DOWNLOAD_REPO
        # Nothing resolved => storage.prune_unused refuses to run (by design).
        assert driver.resolved_artifacts == []
    finally:
        await driver.aclose()


def test_download_repo_resolution_order(tmp_path, fake_flash_binary) -> None:
    """Explicit `troubleshooting.download` > the HF descriptor's repo > this
    image's weights repo; an explicit EMPTY string opts out entirely."""
    from provider_halogen_flash.env import flatten_options

    def repo_for(cfg: dict[str, Any]) -> str | None:
        driver = HalogenFlashBackend(
            make_settings(tmp_path, fake_flash_binary),
            cfg,
            npu_probe=lambda: NPU_UNAVAILABLE,
        )
        return driver._download_repo(flatten_options(cfg))

    assert repo_for({}) == DEFAULT_DOWNLOAD_REPO
    assert (
        repo_for({"artifacts": {"model": {"path": "/models/x.hgn"}}})
        == DEFAULT_DOWNLOAD_REPO
    )
    assert (
        repo_for(
            {
                "artifacts": {
                    "model": {"source": "hf", "repo": "someone/else", "file": "x.hgn"}
                }
            }
        )
        == "someone/else"
    )
    assert (
        repo_for({"troubleshooting": {"download": "explicit/repo"}}) == "explicit/repo"
    )
    # The explicit key wins over the descriptor repo.
    assert (
        repo_for(
            {
                "artifacts": {
                    "model": {"source": "hf", "repo": "someone/else", "file": "x.hgn"}
                },
                "troubleshooting": {"download": "explicit/repo"},
            }
        )
        == "explicit/repo"
    )
    # Empty string = air-gapped opt-out (never an accidental default).
    assert repo_for({"troubleshooting": {"download": ""}}) is None
    assert repo_for({"troubleshooting": {"download": "   "}}) is None


async def test_repo_layout_checkpoint_keeps_its_companion_dir(
    tmp_path, fake_flash_binary, state_dir
) -> None:
    """Engine-fetched companions (overlay, ngram table, tokenizer/) land
    BESIDE the checkpoint and the driver never sees those paths, so the
    directory joins `resolved_artifacts` — unless the checkpoint sits in the
    models root, where protecting "everything" would disable pruning."""
    models = tmp_path / "models"
    repo_dir = models / "peonist-ai" / "halogen-qwen3.8-flash-next"
    repo_dir.mkdir(parents=True)
    ckpt = repo_dir / "qwen38-flash-next-v2.hgn"
    ckpt.write_bytes(b"hgn-fake")
    settings = make_settings(tmp_path, fake_flash_binary)
    driver = HalogenFlashBackend(
        settings,
        {"artifacts": {"model": {"path": str(ckpt)}}},
        npu_probe=lambda: NPU_UNAVAILABLE,
    )
    try:
        await driver.start()
        assert str(repo_dir.resolve()) in [
            str(Path(p).resolve()) for p in driver.resolved_artifacts
        ]
        assert str(models.resolve()) not in [
            str(Path(p).resolve()) for p in driver.resolved_artifacts
        ]
        recorded = json.loads((state_dir / "env.json").read_text())
        assert recorded["HALOGEN_DOWNLOAD"] == DEFAULT_DOWNLOAD_REPO
    finally:
        await driver.aclose()


async def test_models_root_checkpoint_does_not_protect_the_root(
    tmp_path, fake_flash_binary, local_artifacts
) -> None:
    settings = make_settings(tmp_path, fake_flash_binary)
    driver = HalogenFlashBackend(
        settings,
        {
            "artifacts": {
                "model": {"path": local_artifacts["checkpoint"]},
                "tokenizer": {"path": local_artifacts["tokenizer"]},
            }
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )
    try:
        await driver.start()
        resolved = {str(Path(p).resolve()) for p in driver.resolved_artifacts}
        assert resolved == {
            str(local_artifacts["checkpoint"]),
            str(local_artifacts["tokenizer"]),
        }
    finally:
        await driver.aclose()


class _AliveProc:
    """Stand-in for a live Popen: never exits, so only health can end the wait."""

    def poll(self) -> None:
        return None


def _flaky_health(monkeypatch, *, succeed_after: int) -> dict[str, int]:
    """Patch the /health probe to fail `succeed_after - 1` times, then 200."""
    import httpx

    calls = {"n": 0}

    class _Resp:
        status_code = 200

    async def fake_get(self, url, **kwargs):  # noqa: ANN001, ARG001
        calls["n"] += 1
        if calls["n"] < succeed_after:
            raise httpx.ConnectError("engine not listening yet")
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    return calls


def _assert_boot_budget_default() -> None:
    """The shipped boot budget must dwarf llama.cpp's health timeout: these
    engines download before the API port answers. (Test settings pin both to
    15s, so compare the class defaults.)"""
    from provider_lib.config import ProviderSettings

    fields = ProviderSettings.model_fields
    boot = fields["ENGINE_BOOT_TIMEOUT"].default
    health = fields["SERVER_START_HEALTH_TIMEOUT"].default
    assert boot == 3600 and health == 120 and boot > health


async def test_long_boot_heartbeats_initializing_with_engine_line(
    tmp_path, fake_flash_binary, monkeypatch
) -> None:
    """The engine downloads its checkpoint + companions BEFORE binding the
    API port. The wait must report `initializing` (with the engine's newest
    line) rather than sit silent, and the budget is ENGINE_BOOT_TIMEOUT —
    not llama.cpp's 120s health timeout."""
    from provider_halogen_flash import driver as driver_module

    monkeypatch.setattr(driver_module, "_BOOT_HEARTBEAT_SECONDS", 0.0)
    seen: list[tuple[str, str | None]] = []

    async def cb(status: str, reason: str | None) -> None:
        seen.append((status, reason))

    settings = make_settings(tmp_path, fake_flash_binary)
    driver = HalogenFlashBackend(settings, {}, status_cb=cb)
    driver._proc = _AliveProc()  # type: ignore[assignment]
    driver._log_ring.append("stderr", "Downloading 100% of 48.0 GiB tokenizer.json")
    calls = _flaky_health(monkeypatch, succeed_after=3)

    await driver._wait_for_health()

    assert calls["n"] == 3
    assert seen, "expected at least one initializing heartbeat"
    assert {s for s, _ in seen} == {BackendStatusValue.INITIALIZING}
    assert "tokenizer.json" in seen[0][1]


async def test_warm_boot_emits_no_heartbeats(
    tmp_path, fake_flash_binary, monkeypatch
) -> None:
    """Files already on disk: the boot must be as quiet as it was before the
    heartbeat existed (the first beat only fires after the grace period)."""
    seen: list[tuple[str, str | None]] = []

    async def cb(status: str, reason: str | None) -> None:
        seen.append((status, reason))

    settings = make_settings(tmp_path, fake_flash_binary)
    driver = HalogenFlashBackend(settings, {}, status_cb=cb)
    driver._proc = _AliveProc()  # type: ignore[assignment]
    _flaky_health(monkeypatch, succeed_after=1)

    await driver._wait_for_health()
    assert seen == []


async def test_boot_budget_comes_from_engine_boot_timeout(
    tmp_path, fake_flash_binary
) -> None:
    settings = make_settings(tmp_path, fake_flash_binary)
    _assert_boot_budget_default()
    settings.ENGINE_BOOT_TIMEOUT = 0
    driver = HalogenFlashBackend(settings, {})
    driver._proc = _AliveProc()  # type: ignore[assignment]
    with pytest.raises(TimeoutError, match="halogen-flash failed to become healthy"):
        await driver._wait_for_health()


async def test_a_broken_status_callback_never_breaks_the_boot(
    tmp_path, fake_flash_binary, monkeypatch
) -> None:
    """Progress is cosmetic: if the WS is gone mid-boot, the wait still has
    to finish and the backend still has to come up."""
    from provider_halogen_flash import driver as driver_module

    monkeypatch.setattr(driver_module, "_BOOT_HEARTBEAT_SECONDS", 0.0)

    async def cb(_status: str, _reason: str | None) -> None:
        raise RuntimeError("websocket closed")

    settings = make_settings(tmp_path, fake_flash_binary)
    driver = HalogenFlashBackend(settings, {}, status_cb=cb)
    driver._proc = _AliveProc()  # type: ignore[assignment]
    driver._log_ring.append("stdout", "fetching qwen38-flash-next-v2.hgn")
    _flaky_health(monkeypatch, succeed_after=2)

    await driver._wait_for_health()  # must not raise
