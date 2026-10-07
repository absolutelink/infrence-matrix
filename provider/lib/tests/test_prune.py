"""storage.prune_unused handler: keeps referenced artifacts, deletes
orphans; dry_run returns the plan without deleting."""

from pathlib import Path
from typing import Any

import pytest
from config_fakes import RecordingClient, TrackDriver

from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState, install_config_handlers


@pytest.fixture
def settings(tmp_path: Any) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="prune-test",
        MACHINE_SECRET="tok",
        AGENT_ID="prune-test-agent",
        ADMIN_BASE_URL="http://localhost:9999",
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
    )


def _seed_models(settings: ProviderSettings) -> dict[str, Path]:
    models = settings.MODELS_DIR
    files = {
        "main": models / "repo" / "main.gguf",
        "mmproj": models / "repo" / "mmproj.gguf",
        "orphan": models / "old" / "gone.gguf",
        "flat_orphan": models / "stray.gguf",
        "tokenizer_dir": models / "tok" / "tokenizer.json",
    }
    for p in files.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"a" * 10)
    return files


def _client(settings: ProviderSettings, driver: TrackDriver):
    client = RecordingClient(settings)
    lifecycle = BackendLifecycle(driver, capacity=1)
    install_config_handlers(client, lifecycle, ConfigState("fp"), settings)
    return client


async def test_prune_keeps_referenced_deletes_orphans(settings: Any) -> None:
    files = _seed_models(settings)
    driver = TrackDriver()
    driver.resolved_artifacts = [
        str(files["main"]),
        str(files["mmproj"]),
        str(files["tokenizer_dir"].parent),  # directory artifact
    ]
    client = _client(settings, driver)

    ack = await client.dispatch("storage.prune_unused", {})
    assert ack["ok"] is True
    deleted = set(ack["detail"]["deleted"])
    assert deleted == {str(files["orphan"]), str(files["flat_orphan"])}
    assert ack["detail"]["bytes_freed"] == 20
    assert files["main"].exists()
    assert files["mmproj"].exists()
    assert files["tokenizer_dir"].exists()  # under referenced dir
    assert not files["orphan"].exists()
    assert not files["flat_orphan"].exists()


async def test_prune_dry_run_plans_without_deleting(settings: Any) -> None:
    files = _seed_models(settings)
    driver = TrackDriver()
    driver.resolved_artifacts = [str(files["main"])]
    client = _client(settings, driver)

    ack = await client.dispatch("storage.prune_unused", {"dry_run": True})
    assert ack["ok"] is True
    assert ack["detail"]["dry_run"] is True
    planned = set(ack["detail"]["deleted"])
    assert planned == {
        str(files["mmproj"]),
        str(files["orphan"]),
        str(files["flat_orphan"]),
        str(files["tokenizer_dir"]),
    }
    for key in ("mmproj", "orphan", "flat_orphan", "tokenizer_dir"):
        assert files[key].exists()


async def test_prune_keeps_live_file_with_unnormalized_referenced_path(
    settings: Any,
) -> None:
    """N-5: a referenced path recorded with `..` / double slashes must
    still protect the real live file (realpath-normalized comparison)."""
    files = _seed_models(settings)
    models = settings.MODELS_DIR
    driver = TrackDriver()
    messy = [
        # `..` segment resolving to the live main.gguf
        str(models / "repo" / ".." / "repo" / "main.gguf"),
        # double slashes around the mmproj
        str(models) + "//" + "repo" + "//" + "mmproj.gguf",
    ]
    driver.resolved_artifacts = messy
    client = _client(settings, driver)

    ack = await client.dispatch("storage.prune_unused", {})
    assert ack["ok"] is True
    deleted = set(ack["detail"]["deleted"])
    assert str(files["main"]) not in deleted
    assert str(files["mmproj"]) not in deleted
    assert files["main"].exists()
    assert files["mmproj"].exists()
    # Orphans still pruned.
    assert not files["orphan"].exists()
    assert not files["flat_orphan"].exists()


async def test_prune_refused_without_resolved_artifacts(settings: Any) -> None:
    """No trustworthy reference set -> refuse rather than delete all."""
    _seed_models(settings)
    driver = TrackDriver()
    driver.resolved_artifacts = []
    client = _client(settings, driver)

    ack = await client.dispatch("storage.prune_unused", {})
    assert ack["ok"] is False
    assert "no_resolved_artifacts" in ack["error"]
    assert all(p.exists() for p in Path(settings.MODELS_DIR).rglob("*.gguf"))
