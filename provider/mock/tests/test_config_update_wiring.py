"""Phase 9 wiring test: the mock provider installs the shared
config.update / cache.clear / prune_unused handlers and the full flow
works against its real driver."""

import asyncio
from typing import Any

from provider_lib.admin_client import AdminClient
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState, prompt_cache_dir

from provider_mock.backend import MockBackend
from provider_mock.main import install_command_handlers, make_lifecycle


def _settings(tmp_path: Any) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="mock-cu",
        PROVIDER_REGISTRATION_TOKEN="tok",
        ADMIN_BASE_URL="http://localhost:9999",
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
    )


async def _dispatch(client: AdminClient, type_: str, payload: dict) -> dict[str, Any]:
    from provider_lib.wire import Frame

    await client._dispatch(Frame(type=type_, id="cmd-1", epoch=1, payload=payload))
    # Drain the outgoing queue until the ack shows up (status/progress
    # events share the queue and precede it).
    for _ in range(50):
        frame = await asyncio.wait_for(client._outgoing.get(), timeout=5.0)
        if frame.type == "ack" and frame.reply_to == "cmd-1":
            return frame.payload
    raise AssertionError("no ack produced")


def test_handlers_installed() -> None:
    settings = _settings(__import__("pathlib").Path("/tmp"))
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client)
    install_command_handlers(client, lifecycle, ConfigState("fp-0"))
    for name in (
        "backend.start",
        "backend.stop",
        "provider.config.update",
        "cache.clear",
        "storage.prune_unused",
    ):
        assert name in client._command_handlers


async def test_config_update_end_to_end_on_mock(tmp_path: Any) -> None:
    settings = _settings(tmp_path)
    client = AdminClient(settings)
    driver = MockBackend(model="mock-model", delta_count=3)
    lifecycle = BackendLifecycle(driver, capacity=1)
    state = ConfigState("fp-old")
    install_command_handlers(client, lifecycle, state)

    old_cache = prompt_cache_dir(settings, "fp-old")
    old_cache.mkdir(parents=True)
    (old_cache / "p.bin").write_bytes(b"x" * 64)
    model_file = settings.MODELS_DIR / "keep.gguf"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"m")

    ack = await _dispatch(
        client,
        "provider.config.update",
        {
            "backend_config": {"delta_count": 7},
            "config_fingerprint": "fp-new",
            "capacity": 2,
        },
    )
    assert ack["ok"] is True
    assert ack["detail"]["config_fingerprint"] == "fp-new"
    assert ack["detail"]["capacity"] == 2
    assert ack["detail"]["model_metadata"][0]["id"] == "mock-model"
    assert driver.delta_count == 7  # apply_config adopted the new knob
    assert lifecycle.capacity == 2
    assert lifecycle.backend_status == "running"
    assert not old_cache.exists()
    assert model_file.exists()
    assert state.applied_fingerprint == "fp-new"

    # Same fingerprint again -> noop.
    ack = await _dispatch(
        client,
        "provider.config.update",
        {"backend_config": {"delta_count": 7}, "config_fingerprint": "fp-new"},
    )
    assert ack["ok"] is True
    assert ack["detail"]["noop"] is True


async def test_cache_clear_and_prune_on_mock(tmp_path: Any) -> None:
    settings = _settings(tmp_path)
    client = AdminClient(settings)
    driver = MockBackend()
    lifecycle = BackendLifecycle(driver, capacity=1)
    install_command_handlers(client, lifecycle, ConfigState())

    cache_dir = prompt_cache_dir(settings, "fp-z")
    cache_dir.mkdir(parents=True)
    (cache_dir / "c.bin").write_bytes(b"c" * 33)
    orphan = settings.MODELS_DIR / "orphan.gguf"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"o" * 5)

    ack = await _dispatch(client, "cache.clear", {})
    assert ack["ok"] is True
    assert ack["detail"]["bytes_freed"] == 33
    assert not cache_dir.exists()

    # Prune with nothing resolved -> refuse (safety).
    ack = await _dispatch(client, "storage.prune_unused", {})
    assert ack["ok"] is False
    assert "no_resolved_artifacts" in ack["error"]

    driver.resolved_artifacts = [str(orphan)]
    ack = await _dispatch(client, "storage.prune_unused", {"dry_run": True})
    assert ack["ok"] is True
    assert ack["detail"]["deleted"] == []  # orphan is now referenced
    assert orphan.exists()
