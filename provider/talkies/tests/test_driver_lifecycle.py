"""Driver + lifecycle tests against the FAKE talkies python subprocess.

The fake (tests/conftest.py) is spawned exactly like the real engine
(``<python> -m uvicorn talkies.server:app --host 127.0.0.1 --port P``),
dumps the argv + TALKIES_* env it received into a state dir, and answers
``/healthz`` + ``/api/ps``. Covers: spawn + pin semantics, /api/ps health
gating, early-exit reporting, process-group stop, and the prefetch pass
(snapshot_download called only when the flat model dir is empty).
"""

import json
from typing import Any

import pytest
from conftest import ASR_SLUG, TTS_SLUG, make_settings
from provider_lib.backend import BackendLifecycle
from provider_lib.wire import BackendStatusValue

from provider_talkies.driver import TalkiesBackend


def _config(**over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {"model": {"slug": TTS_SLUG}}
    cfg.update(over)
    return cfg


def _driver(tmp_path, python, registry, **over: Any) -> TalkiesBackend:
    settings = make_settings(
        tmp_path, python, models_file=registry, **over.pop("settings_over", {})
    )
    return TalkiesBackend(settings, _config(**over))


async def test_start_spawn_env_health_stop_cycle(
    tmp_path, fake_talkies_python, talkies_registry, model_dir, spawn_state
) -> None:
    assert model_dir.is_dir()  # prefetch cache pre-populated (no network)
    driver = _driver(tmp_path, fake_talkies_python, talkies_registry)
    emitted: list[str] = []

    async def cb(status: str, _reason: str | None) -> None:
        emitted.append(status)

    lifecycle = BackendLifecycle(driver, capacity=1, status_callback=cb)
    try:
        await lifecycle.start()
        assert lifecycle.backend_status == BackendStatusValue.RUNNING
        # starting -> (initializing heartbeats during prefetch/load) -> running
        assert emitted[0] == BackendStatusValue.STARTING
        assert emitted[-1] == BackendStatusValue.RUNNING
        assert set(emitted) <= {
            BackendStatusValue.STARTING,
            BackendStatusValue.INITIALIZING,
            BackendStatusValue.RUNNING,
        }
        assert driver.process is not None and driver.process.poll() is None
        assert await driver.health() is True

        spawn = json.loads((spawn_state / "spawn.json").read_text())
        argv = spawn["argv"]
        # uvicorn launch (never `python -m talkies`), private loopback bind.
        assert argv[1:4] == ["-m", "uvicorn", "talkies.server:app"]
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert int(argv[argv.index("--port") + 1]) == driver.backend_port
        env = spawn["env"]
        assert env["TALKIES_ENABLED_MODELS"] == TTS_SLUG
        assert env["TALKIES_PRELOAD"] == TTS_SLUG
        assert env["TALKIES_MODEL_TTL"] == "0"
        assert env["TALKIES_MODEL_CONCURRENCY"] == f"{TTS_SLUG}=1"
        assert env["TALKIES_MODELS_FILE"] == str(talkies_registry)
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["HF_HOME"] == str(tmp_path / "models" / "hf")
        # Empty config token => a random one generated per start and handed
        # to the engine (never logged).
        assert len(env["TALKIES_AUTH_TOKEN"]) >= 16
    finally:
        await lifecycle.stop()
        await driver.aclose()
    assert driver.process is None
    assert driver.backend_port is None
    assert lifecycle.backend_status == BackendStatusValue.STOPPED


async def test_health_requires_slug_in_api_ps(
    tmp_path, fake_talkies_python, talkies_registry, model_dir, monkeypatch
) -> None:
    """Pin semantics: /healthz alone is NOT healthy — /api/ps must list the
    slug (running <=> resident)."""
    monkeypatch.setenv("FAKE_TALKIES_PS_EMPTY", "1")
    assert model_dir.is_dir()
    driver = _driver(
        tmp_path,
        fake_talkies_python,
        talkies_registry,
        settings_over={"TALKIES_BOOT_TIMEOUT": 2},
    )
    with pytest.raises((RuntimeError, TimeoutError), match="health"):
        await driver.start()
    assert driver.process is None  # failed boot tears the process down


async def test_unknown_slug_fails_before_spawn(
    tmp_path, fake_talkies_python, talkies_registry
) -> None:
    driver = _driver(
        tmp_path, fake_talkies_python, talkies_registry, model={"slug": "nope"}
    )
    with pytest.raises(RuntimeError, match="not in registry"):
        await driver.start()


async def test_early_exit_includes_stderr_tail(
    tmp_path, early_exit_python, talkies_registry, model_dir
) -> None:
    assert model_dir.is_dir()
    driver = _driver(
        tmp_path,
        early_exit_python,
        talkies_registry,
        settings_over={"TALKIES_BOOT_TIMEOUT": 5},
    )
    with pytest.raises(RuntimeError) as exc_info:
        await driver.start()
    message = str(exc_info.value)
    assert "exited with code 3" in message
    assert "CUDA out of memory" in message


async def test_stop_is_idempotent(
    tmp_path, fake_talkies_python, talkies_registry, model_dir
) -> None:
    assert model_dir.is_dir()
    driver = _driver(tmp_path, fake_talkies_python, talkies_registry)
    await driver.start()
    proc = driver.process
    assert proc is not None and proc.poll() is None
    await driver.stop()
    assert proc.poll() is not None
    await driver.stop()  # idempotent
    await driver.aclose()


async def test_prefetch_skipped_whenmodel_dir_non_empty(
    tmp_path, fake_talkies_python, talkies_registry, model_dir, monkeypatch
) -> None:
    """The flat $DATA/models/<slug> dir's non-emptiness is the cached signal
    (entrypoint parity): snapshot_download must NOT run."""

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("snapshot_download must not be called when cached")

    assert model_dir.is_dir() and any(model_dir.iterdir())
    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    driver = _driver(tmp_path, fake_talkies_python, talkies_registry)
    try:
        await driver.start()
        assert await driver.health() is True
        assert driver.resolved_artifacts == [
            str(model_dir),
            str(tmp_path / "models" / "custom-voices"),
        ]
    finally:
        await driver.aclose()


async def test_prefetch_downloads_when_dir_empty(
    tmp_path, fake_talkies_python, talkies_registry, monkeypatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(repo: str, **kwargs: Any) -> str:
        calls.append({"repo": repo, **kwargs})
        local_dir = kwargs.get("local_dir")
        if local_dir:
            from pathlib import Path

            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "weights.bin").write_bytes(b"w")
        return str(local_dir or repo)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    driver = _driver(
        tmp_path,
        fake_talkies_python,
        talkies_registry,
        model={"slug": ASR_SLUG},
    )
    try:
        await driver.start()
        # Registry repo + revision honored; dependency repos go to the HF
        # cache (no local_dir), entrypoint parity.
        main = [
            c for c in calls if c["repo"] == "deepdml/faster-whisper-large-v3-turbo-ct2"
        ]
        assert main and "local_dir" in main[0]
        assert {"repo": "some/dep-repo"} in [{"repo": c["repo"]} for c in calls]
        dep = [c for c in calls if c["repo"] == "some/dep-repo"][0]
        assert "local_dir" not in dep
        # NEW-1 discriminator: the dependency cache_dir the prefetch used MUST
        # equal the hub cache the engine actually reads (build_env's
        # HF_HUB_CACHE, which beats HF_HOME's implicit <HOME>/hub). Fails if
        # either side moves.
        from provider_talkies import env as tenv

        engine_env = tenv.build_env(
            {"model": {"slug": ASR_SLUG}},
            settings=driver._settings,
            data_dir=driver._data_dir,
            auth_token="t",
        )
        assert (
            dep["cache_dir"]
            == engine_env["HF_HUB_CACHE"]
            == str(driver._data_dir / "hf")
        )
    finally:
        await driver.aclose()


async def test_download_patterns_reach_snapshot_download(
    tmp_path, fake_talkies_python, talkies_registry, monkeypatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(repo: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        from pathlib import Path

        local_dir = kwargs.get("local_dir")
        if local_dir:
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "x").write_bytes(b"x")
        return str(local_dir or repo)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    driver = _driver(
        tmp_path,
        fake_talkies_python,
        talkies_registry,
        model={"slug": "sherpa-zipformer-en-left-64"},
    )
    try:
        await driver.start()
    finally:
        await driver.aclose()
    assert calls and calls[0]["allow_patterns"] == ["tokens.txt", "encoder*.onnx"]


def test_apply_config_resets_resolved_state(tmp_path, talkies_registry) -> None:
    driver = TalkiesBackend(make_settings(tmp_path, models_file=talkies_registry), {})
    driver.resolved_artifacts = ["/models/old"]
    driver.slug = "old-slug"
    driver.apply_config(_config())
    assert driver.resolved_artifacts == []
    assert driver.slug is None
    assert driver.backend_port is None


async def test_list_models_uses_alias_then_slug(tmp_path, talkies_registry) -> None:
    driver = _driver(tmp_path, "python", talkies_registry)
    driver.apply_config(_config())
    models = await driver.list_models()
    assert models[0]["id"] == TTS_SLUG
    driver.set_alias("my-tts")
    driver.modality = "tts"
    models = await driver.list_models()
    assert models[0]["id"] == "my-tts"
    assert models[0]["modality"] == "tts"
    assert models[0]["owned_by"] == "talkies"
