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
    model_manager.download_model_parts = AsyncMock(
        side_effect=AssertionError("should not download when all parts present")
    )

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


@pytest.mark.asyncio
async def test_ensure_model_gufo_aux_downloads_when_missing(
    tmp_models: Path,
) -> None:
    """A Gufo aux source dict (path + repo_id + filenames + server_id)
    downloads the file to ``MODELS_PATH/repo/file`` when it is missing and
    returns that path (== the engine_options path the binary uses)."""
    repo = "aux-org/mmproj"
    aux_path = f"/models/{repo}/vision.gguf"
    aux = {
        "key": "mmproj",
        "path": aux_path,
        "source": "huggingface",
        "repo_id": repo,
        "filename": "vision.gguf",
        "filenames": ["vision.gguf"],
        "job_id": "server-abc-mmproj",
        "server_id": "abc",
    }
    expected = str(tmp_models / repo / "vision.gguf")
    mock = AsyncMock(return_value=expected)
    model_manager.download_model_parts = mock

    result = await servers._ensure_model(aux_path, aux)

    assert result == expected
    mock.assert_awaited_once()
    assert mock.await_args.kwargs["repo_id"] == repo
    assert mock.await_args.kwargs["filenames"] == ["vision.gguf"]
    assert mock.await_args.kwargs["server_id"] == "abc"


@pytest.mark.asyncio
async def test_ensure_model_gufo_aux_split_uses_primary(
    tmp_models: Path,  # noqa: ARG001
) -> None:
    repo = "aux-org/mtp"
    aux_path = f"/models/{repo}/mtp.gguf"
    aux = {
        "key": "mtp_model",
        "path": aux_path,
        "source": "huggingface",
        "repo_id": repo,
        "filename": "mtp-00001-of-00002.gguf",
        "filenames": [
            "mtp-00001-of-00002.gguf",
            "mtp-00002-of-00002.gguf",
        ],
        "job_id": "server-xyz-mtp_model",
        "server_id": "xyz",
    }
    expected = str(tmp_models / repo / "mtp-00001-of-00002.gguf")
    mock = AsyncMock(return_value=expected)
    model_manager.download_model_parts = mock

    result = await servers._ensure_model(aux_path, aux)

    assert result == expected
    assert mock.await_args.kwargs["filenames"] == aux["filenames"]


@pytest.mark.asyncio
async def test_ensure_model_gufo_aux_already_present_skips_download(
    tmp_models: Path,
) -> None:
    repo = "aux-org/present"
    aux_path = f"/models/{repo}/mm.gguf"
    (tmp_models / repo).mkdir(parents=True)
    (tmp_models / repo / "mm.gguf").write_bytes(b"x")
    aux = {
        "key": "mmproj",
        "path": aux_path,
        "source": "huggingface",
        "repo_id": repo,
        "filename": "mm.gguf",
        "filenames": ["mm.gguf"],
        "job_id": "server-1-mmproj",
        "server_id": "1",
    }
    mock = AsyncMock(side_effect=AssertionError("should not download when present"))
    model_manager.download_model_parts = mock

    result = await servers._ensure_model(aux_path, aux)

    assert result == str(tmp_models / repo / "mm.gguf")


@pytest.mark.asyncio
async def test_prepare_rewrites_gufo_aux_engine_option_to_subfolder_path(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: Qwen3.8-Flash-Next MTP downloads into a repo SUBFOLDER, but
    engine_options recorded the library Model.path (repo root). prepare_server
    must rewrite engine_options[key] to the actual on-disk (subfolder) path so
    the gufo binary reads the file where it landed."""
    from app.core.config import settings

    # Allow the gufo engine on this test agent platform.
    monkeypatch.setattr(settings, "AGENT_PLATFORM", "gufo")

    repo = "org/qwen3.8-flash-next"
    # Library path points at the repo root; the file actually lives in mtp/.
    stale_root_path = f"/models/{repo}/mtp.gguf"
    subfolder_path = str(tmp_models / repo / "mtp" / "mtp.gguf")
    aux = {
        "key": "mtp_model",
        "path": stale_root_path,
        "source": "huggingface",
        "repo_id": repo,
        "filename": "mtp/mtp.gguf",
        "filenames": ["mtp/mtp.gguf"],
        "job_id": "server-flash-mtp_model",
        "server_id": "flash",
    }
    # Simulate the download landing in the subfolder.
    (tmp_models / repo / "mtp").mkdir(parents=True)
    (tmp_models / repo / "mtp" / "mtp.gguf").write_bytes(b"x")

    request = servers.ServerStartRequest(
        config=servers.ServerSpec(
            id="flash",
            model_path=f"/models/{repo}/base.gguf",
            engine="gufo",
            engine_options={"mtp_model": stale_root_path},
        ),
        source={
            "source": "huggingface",
            "repo_id": repo,
            "filename": "base.gguf",
            "filenames": ["base.gguf"],
            "job_id": "server-flash",
            "server_id": "flash",
        },
        aux_sources=[aux],
    )
    # Avoid downloading the main base model; it is not the focus here.
    (tmp_models / repo / "base.gguf").write_bytes(b"b")

    result = await servers.prepare_server(request)

    assert result["status"] == "prepared"
    # The stale root path must have been replaced by the real subfolder path.
    assert request.config.engine_options["mtp_model"] == subfolder_path
    assert request.config.engine_options["mtp_model"] != stale_root_path
