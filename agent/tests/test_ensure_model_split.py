"""Tests for the agent _ensure_model split-GGUF handling (no network)."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api.routes import servers
from app.services.model_manager import model_manager


@pytest.fixture
def tmp_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the model_manager at a temp MODELS_PATH."""
    monkeypatch.setattr(model_manager, "models_path", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_ensure_model_split_all_present_returns_primary(tmp_models: Path) -> None:
    repo = "test-org/split"
    repo_dir = tmp_models / repo
    repo_dir.mkdir(parents=True)
    parts = [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]
    for p in parts:
        (repo_dir / p).write_bytes(b"x")

    source = {
        "source": "huggingface",
        "repo_id": repo,
        "filename": "model-00001-of-00002.gguf",
        "filenames": list(parts),
        "job_id": "job-1",
    }
    # No download should occur when all parts are present.
    model_manager.download_model_parts = AsyncMock(side_effect=AssertionError(
        "should not download when all parts present"
    ))

    result = await servers._ensure_model(f"/models/{repo}/model.gguf", source)
    assert result == str(repo_dir / "model-00001-of-00002.gguf")


@pytest.mark.asyncio
async def test_ensure_model_split_missing_downloads_all_parts(tmp_models: Path) -> None:
    repo = "test-org/split2"
    source = {
        "source": "huggingface",
        "repo_id": repo,
        "filename": "model-00001-of-00003.gguf",
        "filenames": [
            "model-00002-of-00003.gguf",
            "model-00001-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ],
        "job_id": "job-2",
    }
    expected = str(tmp_models / repo / "model-00001-of-00003.gguf")
    mock = AsyncMock(return_value=expected)
    model_manager.download_model_parts = mock

    result = await servers._ensure_model(f"/models/{repo}/model.gguf", source)
    assert result == expected
    mock.assert_awaited_once()
    called_filenames = mock.await_args.kwargs["filenames"]
    assert called_filenames == source["filenames"]


@pytest.mark.asyncio
async def test_ensure_model_single_file_present(tmp_models: Path) -> None:
    (tmp_models / "solo.gguf").write_bytes(b"x")
    source = {
        "source": "huggingface",
        "repo_id": "test-org/solo",
        "filename": "solo.gguf",
        "job_id": "job-3",
    }
    result = await servers._ensure_model("solo.gguf", source)
    assert result == str(tmp_models / "solo.gguf")


@pytest.mark.asyncio
async def test_ensure_model_missing_no_source_raises(
    tmp_models: Path,  # noqa: ARG001
) -> None:
    with pytest.raises(HTTPException) as exc:
        await servers._ensure_model("/models/absent.gguf", None)
    assert exc.value.status_code == 404
