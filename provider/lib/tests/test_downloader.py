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


async def test_download_called_with_tqdm_and_returns_path(
    tmp_path, monkeypatch
) -> None:
    calls: list[dict] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"downloaded")
        # Drive the tqdm_class like huggingface_hub does.
        bar = kwargs["tqdm_class"](total=100, initial=0)
        bar.update(50)
        time.sleep(1.05)  # exceed the throttle window
        bar.update(50)
        bar.close()
        return str(target)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download, raising=True
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
    # Let the run_coroutine_threadsafe callbacks land.
    for _ in range(200):
        if len(events) >= 2:
            break
        await asyncio.sleep(0.01)
    assert len(events) >= 2
    assert events[-1]["finished"] is True
    assert events[-1]["bytes_downloaded"] == 100
    assert events[-1]["file"] == "m.gguf"


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
