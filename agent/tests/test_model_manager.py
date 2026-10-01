"""Tests for split GGUF model download helpers in the agent ModelManager."""

from unittest.mock import patch

import pytest

from app.services.model_manager import (
    ModelManager,
    _make_tqdm_class,
    pick_primary_filename,
)


def test_pick_primary_filename_prefers_first_part() -> None:
    names = [
        "model-00003-of-00003.gguf",
        "model-00001-of-00003.gguf",
        "model-00002-of-00003.gguf",
    ]
    assert pick_primary_filename(names) == "model-00001-of-00003.gguf"


def test_pick_primary_filename_single() -> None:
    assert pick_primary_filename(["solo.gguf"]) == "solo.gguf"


def test_pick_primary_filename_no_index_returns_first() -> None:
    assert pick_primary_filename(["a.gguf", "b.gguf"]) == "a.gguf"


@pytest.mark.asyncio
async def test_download_model_parts_downloads_all_and_returns_primary() -> None:
    mgr = ModelManager()
    calls: list[str] = []

    async def fake_download(
        repo_id,
        filename,
        source="huggingface",  # noqa: ARG001
        job_id=None,  # noqa: ARG001
        server_id=None,  # noqa: ARG001
    ):
        calls.append(filename)
        return f"/tmp/models/{repo_id}/{filename}"

    with patch.object(mgr, "download_model", side_effect=fake_download):
        result = await mgr.download_model_parts(
            "test-org/split",
            [
                "model-00002-of-00002.gguf",
                "model-00001-of-00002.gguf",
            ],
            job_id="job-1",
        )

    # Every part downloaded exactly once.
    assert sorted(calls) == [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]
    # Returned path is the primary (-00001-of-) part.
    assert result == "/tmp/models/test-org/split/model-00001-of-00002.gguf"


@pytest.mark.asyncio
async def test_download_model_parts_empty_raises() -> None:
    mgr = ModelManager()
    with pytest.raises(ValueError):
        await mgr.download_model_parts("test-org/empty", [])


def test_make_tqdm_class_binds_server_id() -> None:
    cls = _make_tqdm_class("job-1", "model.gguf", "srv-uuid")
    assert cls.server_id == "srv-uuid"
    assert cls.job_id == "job-1"
    assert cls.filename == "model.gguf"


def test_make_tqdm_class_server_id_defaults_empty() -> None:
    cls = _make_tqdm_class("job-1", "model.gguf")
    assert cls.server_id == ""


def test_event_publishing_tqdm_includes_server_id() -> None:
    cls = _make_tqdm_class("job-1", "model.gguf", "srv-uuid")
    bar = cls(total=1000)
    bar.n = 500
    with patch("app.services.model_manager.publish_event") as mock_publish:
        bar._publish(now=bar._last_ts + 1.0)
    event_name, payload = mock_publish.call_args.args
    assert event_name == "download.progress"
    assert payload["server_id"] == "srv-uuid"
    assert payload["job_id"] == "job-1"
    assert payload["filename"] == "model.gguf"
    assert payload["progress_percent"] == 50.0


@pytest.mark.asyncio
async def test_download_model_parts_threads_server_id() -> None:
    mgr = ModelManager()
    seen: list[str | None] = []

    async def fake_download(
        repo_id,
        filename,
        source="huggingface",  # noqa: ARG001
        job_id=None,  # noqa: ARG001
        server_id=None,
    ):
        seen.append(server_id)
        return f"/tmp/models/{repo_id}/{filename}"

    with patch.object(mgr, "download_model", side_effect=fake_download):
        await mgr.download_model_parts(
            "test-org/split",
            ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"],
            job_id="server-abc",
            server_id="srv-uuid",
        )

    assert seen == ["srv-uuid", "srv-uuid"]
