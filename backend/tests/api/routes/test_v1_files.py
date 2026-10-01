"""Tests for V1 Files API endpoint (OpenAI-compatible file management)."""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.routes.v1 import v1_files
from app.models import File as FileModel


@pytest.fixture
def files_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the upload directory at a per-test temp path."""
    target = tmp_path / "files"
    monkeypatch.setattr(v1_files.settings, "FILES_PATH", str(target), raising=False)
    return target


def _upload(client: TestClient, purpose: str, filename: str = "payload.txt"):
    return client.post(
        "/v1/files",
        files={"file": (filename, b"hello world", "text/plain")},
        data={"purpose": purpose},
    )


class TestAllowedPurposes:
    """Upload accepts the full OpenAI purpose set."""

    @pytest.mark.parametrize(
        "purpose",
        [
            "assistants",
            "assistants_output",
            "batch",
            "batch_output",
            "fine-tune",
            "fine-tune-results",
            "user_data",
            "vision",
        ],
    )
    def test_valid_purpose_accepted(
        self,
        client: TestClient,
        db: Session,
        files_dir: Path,
        purpose: str,
    ) -> None:
        response = _upload(client, purpose)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["purpose"] == purpose
        assert data["object"] == "file"
        assert data["bytes"] == len(b"hello world")
        assert data["status"] == "uploaded"
        # OpenAI-style id with the `file-` prefix and a 32-char hex body.
        assert data["id"].startswith("file-")
        assert len(data["id"]) == len("file-") + 32
        int(data["id"].removeprefix("file-"), 16)  # hex-parseable

        stored = db.exec(select(FileModel)).all()
        assert len(stored) == 1
        assert stored[0].purpose == purpose
        assert Path(stored[0].path).exists()
        assert len(stored[0].checksum_sha256) == 64
        # Round-trip: prefixed id resolves to the same DB row via retrieve.
        got = client.get(f"/v1/files/{data['id']}")
        assert got.status_code == 200
        assert got.json()["id"] == data["id"]

    def test_previously_rejected_purpose_now_works(
        self, client: TestClient, db: Session, files_dir: Path
    ) -> None:
        """`vision` was rejected under the old 3-purpose allowlist; now accepted."""
        assert "vision" not in {"batch", "retrieval", "assistants"}
        response = _upload(client, "vision", filename="image.png")
        assert response.status_code == 200, response.text
        assert response.json()["purpose"] == "vision"

    def test_retrieval_purpose_now_rejected(
        self, client: TestClient, db: Session, files_dir: Path
    ) -> None:
        """`retrieval` is not part of the OpenAI set and is rejected."""
        response = _upload(client, "retrieval")
        assert response.status_code == 400
        assert "Invalid purpose" in response.json()["detail"]

    def test_invalid_purpose_rejected(
        self, client: TestClient, db: Session, files_dir: Path
    ) -> None:
        response = _upload(client, "bogus-purpose")
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Invalid purpose" in detail
        # The error message advertises the full allowed set.
        assert "fine-tune" in detail
        assert "assistants_output" in detail


class TestListFiles:
    def test_list_empty(self, client: TestClient, db: Session) -> None:
        response = client.get("/v1/files")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_filters_by_purpose(
        self, client: TestClient, db: Session, files_dir: Path
    ) -> None:
        _upload(client, "batch", filename="a.jsonl")
        _upload(client, "vision", filename="b.png")

        response = client.get("/v1/files", params={"purpose": "vision"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["purpose"] == "vision"
        assert data["data"][0]["filename"] == "b.png"


class TestRetrieveAndDelete:
    def test_retrieve_and_delete(
        self, client: TestClient, db: Session, files_dir: Path
    ) -> None:
        uploaded = _upload(client, "user_data", filename="doc.txt").json()
        file_id = uploaded["id"]

        got = client.get(f"/v1/files/{file_id}")
        assert got.status_code == 200
        assert got.json()["filename"] == "doc.txt"

        content = client.get(f"/v1/files/{file_id}/content")
        assert content.status_code == 200
        assert content.content == b"hello world"

        deleted = client.delete(f"/v1/files/{file_id}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

        assert client.get(f"/v1/files/{file_id}").status_code == 404
        assert db.exec(select(FileModel)).all() == []
        # On-disk file removed.
        assert not any(files_dir.glob("file-*"))

    def test_retrieve_missing_returns_404(
        self, client: TestClient, db: Session
    ) -> None:
        response = client.get(f"/v1/files/{uuid.uuid4()}")
        assert response.status_code == 404
        # A prefixed unknown id also 404s (not a parse error).
        response = client.get(f"/v1/files/file-{uuid.uuid4().hex}")
        assert response.status_code == 404

    def test_malformed_id_returns_404(self, client: TestClient, db: Session) -> None:
        response = client.get("/v1/files/not-a-uuid")
        assert response.status_code == 404

    def test_filename_is_sanitized_to_basename(
        self, client: TestClient, db: Session, files_dir: Path
    ) -> None:
        """A path-traversal-style filename is stored as its basename only."""
        response = _upload(client, "user_data", filename="../../etc/passwd.txt")
        assert response.status_code == 200, response.text
        assert response.json()["filename"] == "passwd.txt"

        stored = db.exec(select(FileModel)).one()
        assert stored.filename == "passwd.txt"
        # The written file lives inside files_dir, not escaped upward.
        assert stored.path.startswith(str(files_dir))
        assert Path(stored.path).name.startswith("file-")
        assert Path(stored.path).exists()
