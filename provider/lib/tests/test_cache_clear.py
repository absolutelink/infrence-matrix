"""cache.clear handler: deletes ONLY prompt-cache convention dirs."""

from typing import Any

import pytest
from config_fakes import RecordingClient, TrackDriver

from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import (
    ConfigState,
    install_config_handlers,
    prompt_cache_dir,
    prompt_cache_root,
)


@pytest.fixture
def settings(tmp_path: Any) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="cc-test",
        MACHINE_SECRET="tok",
        AGENT_ID="cc-test-agent",
        ADMIN_BASE_URL="http://localhost:9999",
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
    )


def _seed(settings: ProviderSettings) -> None:
    for fp in ("fp-a", "fp-b"):
        d = prompt_cache_dir(settings, fp)
        d.mkdir(parents=True, exist_ok=True)
        (d / "slot.bin").write_bytes(b"z" * 40)
    # A model file that must NEVER be deleted by cache.clear.
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.MODELS_DIR / "model.gguf").write_bytes(b"m" * 1000)
    # provider_config.json in CACHE_DIR must survive too.
    (settings.CACHE_DIR / "provider_config.json").write_text("{}")


async def test_cache_clear_deletes_prompt_cache_only(settings: Any) -> None:
    _seed(settings)
    client = RecordingClient(settings)
    lifecycle = BackendLifecycle(TrackDriver(), capacity=1)
    install_config_handlers(client, lifecycle, ConfigState("fp-a"), settings)

    ack = await client.dispatch("cache.clear", {})
    assert ack["ok"] is True
    assert ack["detail"]["bytes_freed"] == 80  # two 40-byte slot files
    assert not prompt_cache_root(settings).exists()
    # Model files and other cache content untouched.
    assert (settings.MODELS_DIR / "model.gguf").exists()
    assert (settings.CACHE_DIR / "provider_config.json").exists()


async def test_cache_clear_dry_run_deletes_nothing(settings: Any) -> None:
    _seed(settings)
    client = RecordingClient(settings)
    lifecycle = BackendLifecycle(TrackDriver(), capacity=1)
    install_config_handlers(client, lifecycle, ConfigState("fp-a"), settings)

    ack = await client.dispatch("cache.clear", {"dry_run": True})
    assert ack["ok"] is True
    assert ack["detail"]["dry_run"] is True
    assert ack["detail"]["bytes_freed"] == 80
    assert len(ack["detail"]["deleted"]) == 2
    assert prompt_cache_dir(settings, "fp-a").exists()
    assert prompt_cache_dir(settings, "fp-b").exists()


async def test_cache_clear_noop_when_no_cache(settings: Any) -> None:
    client = RecordingClient(settings)
    lifecycle = BackendLifecycle(TrackDriver(), capacity=1)
    install_config_handlers(client, lifecycle, ConfigState(), settings)
    ack = await client.dispatch("cache.clear", {})
    assert ack["ok"] is True
    assert ack["detail"]["bytes_freed"] == 0
    assert ack["detail"]["deleted"] == []


async def test_cache_clear_includes_extra_dirs(settings: Any) -> None:
    """Provider-specific engine cache dirs (extra_cache_dirs) are cleared."""
    extra = settings.CACHE_DIR / "halogen-flash"
    extra.mkdir(parents=True)
    (extra / "diskcache.bin").write_bytes(b"c" * 50)
    client = RecordingClient(settings)
    lifecycle = BackendLifecycle(TrackDriver(), capacity=1)
    install_config_handlers(
        client, lifecycle, ConfigState(), settings, extra_cache_dirs=[extra]
    )
    ack = await client.dispatch("cache.clear", {})
    assert ack["ok"] is True
    assert ack["detail"]["bytes_freed"] == 50
    assert not extra.exists()


async def test_cache_clear_refused_while_in_use(settings: Any) -> None:
    """N-4: clearing engine caches under load can break live streams ->
    refuse with backend_in_use unless force."""
    _seed(settings)
    client = RecordingClient(settings)
    lifecycle = BackendLifecycle(TrackDriver(), capacity=1)
    install_config_handlers(client, lifecycle, ConfigState("fp-a"), settings)
    await lifecycle.start()
    await lifecycle.acquire_slot()

    ack = await client.dispatch("cache.clear", {})
    assert ack["ok"] is False
    assert ack["error"] == "backend_in_use"
    assert ack["detail"]["step"] == "drain"
    assert ack["detail"]["in_flight"] == 1
    # Nothing deleted.
    assert prompt_cache_root(settings).exists()

    # dry_run is always allowed (touches nothing).
    ack = await client.dispatch("cache.clear", {"dry_run": True})
    assert ack["ok"] is True
    assert prompt_cache_root(settings).exists()

    # force: true overrides the refusal.
    ack = await client.dispatch("cache.clear", {"force": True})
    assert ack["ok"] is True
    assert ack["detail"]["bytes_freed"] == 80
    assert not prompt_cache_root(settings).exists()
    await lifecycle.release_slot()
