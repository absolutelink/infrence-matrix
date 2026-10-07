"""ensure_artifact: local paths, flat resolution, dedup, progress, guards."""

import asyncio
import time
from pathlib import Path

import pytest

from provider_lib import downloader
from provider_lib.downloader import DownloadError, ensure_artifact


@pytest.fixture(autouse=True)
def _clear_registry():
    downloader._DOWNLOAD_TASKS.clear()
    yield
    downloader._DOWNLOAD_TASKS.clear()


async def test_local_path_exists(tmp_path: Path) -> None:
    f = tmp_path / "local.gguf"
    f.write_bytes(b"x")
    assert await ensure_artifact({"path": str(f)}, models_dir=tmp_path) == str(f)


async def test_local_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="not found"):
        await ensure_artifact(
            {"path": str(tmp_path / "nope.gguf")}, models_dir=tmp_path
        )


async def test_repo_layout_found_without_download(tmp_path: Path, monkeypatch) -> None:
    mirrored = tmp_path / "ggml-org" / "models" / "gemma.gguf"
    mirrored.parent.mkdir(parents=True)
    mirrored.write_bytes(b"x")

    def boom(*_a, **_k):  # pragma: no cover - must not be called
        raise AssertionError("download must not happen when file exists")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom, raising=True)
    got = await ensure_artifact(
        {"source": "hf", "repo": "ggml-org/models", "file": "gemma.gguf"},
        models_dir=tmp_path,
    )
    assert got == str(mirrored)


async def test_flat_resolution_without_download(tmp_path: Path) -> None:
    flat = tmp_path / "gemma.gguf"
    flat.write_bytes(b"x")
    got = await ensure_artifact(
        {"source": "hf", "repo": "some/repo", "file": "gemma.gguf"},
        models_dir=tmp_path,
    )
    assert got == str(flat)


async def test_offline_env_does_not_block_an_operator_download(
    tmp_path, monkeypatch
) -> None:
    """Engine images bake HF_HUB_OFFLINE=1 so the served backend never
    reaches for the Hub mid-request — but that env must not silence the
    provider's own artifact fetch, which only runs when the operator picked
    an HF file. The hub's in-process switch is cleared; os.environ is left
    exactly as the image set it (backend children still inherit it)."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    hf_constants.HF_HUB_OFFLINE = True

    seen: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs):
        seen["flag_during_call"] = hf_constants.HF_HUB_OFFLINE
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"downloaded")
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    got = await ensure_artifact(
        {"source": "hf", "repo": "some/repo", "file": "model.gguf"},
        models_dir=tmp_path,
    )

    assert seen["flag_during_call"] is False
    assert got == str(tmp_path / "some" / "repo" / "model.gguf")
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"


async def test_download_progress_events_and_returns_path(tmp_path, monkeypatch) -> None:
    """Progress (0.36+ reality): hf_hub_download exposes NO bar hook and
    hf_xet bypasses http_get's tqdm entirely — _download_hf therefore polls
    the incomplete file and publishes throttled events + a terminal one."""
    calls: list[dict] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        # Simulate the hub writing under the incomplete candidates the
        # poller watches, then finishing.
        incomplete = (
            Path(kwargs["local_dir"])
            / ".cache"
            / "huggingface"
            / "download"
            / "m.gguf.incomplete"
        )
        incomplete.parent.mkdir(parents=True, exist_ok=True)
        incomplete.write_bytes(b"x" * 40)
        time.sleep(1.2)  # let one poll tick observe bytes
        incomplete.write_bytes(b"x" * 80)
        time.sleep(1.2)  # second observed tick
        incomplete.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"downloaded")
        return str(target)

    # No real metadata HEAD in the unit test (total unknown OK).
    async def fake_size(*a, **k):  # noqa: ARG001
        return None

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download, raising=True
    )
    monkeypatch.setattr(
        "provider_lib.downloader._hf_file_size", fake_size, raising=True
    )

    events: list[dict] = []

    async def progress_cb(payload: dict) -> None:
        events.append(payload)

    got = await ensure_artifact(
        {"source": "huggingface", "repo": "my/repo", "file": "m.gguf"},
        models_dir=tmp_path,
        progress_cb=progress_cb,
    )
    assert got.endswith("my/repo/m.gguf")
    assert calls and calls[0]["repo_id"] == "my/repo"
    assert len(events) >= 3  # start + intermediate + terminal
    # Intermediate events carry observed bytes; terminal marks finished.
    intermediates = [e for e in events if not e["finished"]]
    assert intermediates and max(e["bytes_downloaded"] for e in intermediates) >= 40
    last = events[-1]
    assert last["finished"] is True
    assert last["file"] == "m.gguf"
    assert last["bytes_downloaded"] == len(b"downloaded")
    assert last["progress_percent"] == 100.0


async def test_inflight_dedup_single_download(tmp_path, monkeypatch) -> None:
    downloads: list[str] = []
    gate = asyncio.Event()

    def fake_hf_hub_download(**kwargs):
        downloads.append(kwargs["filename"])
        # Block (in-thread) until released so both callers overlap.
        while not gate.is_set():
            time.sleep(0.01)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return str(target)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download, raising=True
    )
    art = {"source": "hf", "repo": "r/r1", "file": "same.gguf"}
    t1 = asyncio.create_task(ensure_artifact(art, models_dir=tmp_path))
    while not downloads:
        await asyncio.sleep(0.01)
    t2 = asyncio.create_task(ensure_artifact(art, models_dir=tmp_path))
    await asyncio.sleep(0.05)
    gate.set()
    p1, p2 = await asyncio.wait_for(asyncio.gather(t1, t2), timeout=10)
    assert p1 == p2
    assert len(downloads) == 1  # second call awaited the in-flight task


async def test_traversal_guards(tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        await ensure_artifact(
            {"source": "hf", "repo": "../evil", "file": "x.gguf"},
            models_dir=tmp_path,
        )
    with pytest.raises(DownloadError):
        await ensure_artifact(
            {"source": "hf", "repo": "ok/repo", "file": "../../x.gguf"},
            models_dir=tmp_path,
        )
    with pytest.raises(DownloadError):
        await ensure_artifact(
            {"source": "hf", "repo": "/abs/repo", "file": "x.gguf"},
            models_dir=tmp_path,
        )


async def test_unsupported_source(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="unsupported"):
        await ensure_artifact(
            {"source": "modelscope", "repo": "a/b", "file": "x.gguf"},
            models_dir=tmp_path,
        )


async def test_missing_repo_or_file(tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        await ensure_artifact({"source": "hf", "file": "x.gguf"}, models_dir=tmp_path)
    with pytest.raises(DownloadError):
        await ensure_artifact({"source": "hf", "repo": "a/b"}, models_dir=tmp_path)
