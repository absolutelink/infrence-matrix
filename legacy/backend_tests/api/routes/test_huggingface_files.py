"""Route-level test for /huggingface/models/files enrichment (httpx mocked)."""

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("GET", "http://x"),
                response=self,  # type: ignore[arg-type]
            )

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._payload = {
            "siblings": [
                {"rfilename": "UD-Q4_K_XL/model-00001-of-00002.gguf", "size": 100},
                {"rfilename": "UD-Q4_K_XL/model-00002-of-00002.gguf", "size": 100},
                {"rfilename": "Q8_0/model.gguf", "size": 300},
                {"rfilename": "mmproj-F16.gguf", "size": 50},
                {"rfilename": "loose.gguf", "size": 10},
                {"rfilename": "README.md", "size": 1},
            ],
            "cardData": {"gguf": {"Q8_0/model.gguf": "Q8_0"}},
        }

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def get(self, url: str, params: Any = None, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._payload)


@pytest.fixture
def mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)


@pytest.mark.usefixtures("mocked_client")
def test_list_model_files_enriches_quantization_and_aux() -> None:
    client = TestClient(app)
    resp = client.get("/api/v1/huggingface/models/files", params={"repo_id": "foo/bar"})
    assert resp.status_code == 200
    files = {f["path"]: f for f in resp.json()}

    # Split files under a quantization subfolder.
    q4a = files["UD-Q4_K_XL/model-00001-of-00002.gguf"]
    assert q4a["quantization"] == "UD-Q4_K_XL"
    assert q4a["is_aux"] is False
    assert q4a["model_type"] == "llm"

    # Q8 tag from cardData mapping (matches dir too).
    q8 = files["Q8_0/model.gguf"]
    assert q8["quantization"] == "Q8_0"
    assert q8["is_aux"] is False

    # mmproj is auxiliary (F16 tag may be present but it is still aux).
    mm = files["mmproj-F16.gguf"]
    assert mm["is_aux"] is True
    assert mm["model_type"] == "mmproj"

    # Loose untagged file.
    loose = files["loose.gguf"]
    assert loose["quantization"] is None
    assert loose["is_aux"] is False

    # Non-gguf filtered out.
    assert "README.md" not in files
