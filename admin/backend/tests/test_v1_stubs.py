"""501 stubs for dropped public endpoints (Phase 7)."""

import pytest
from fastapi.testclient import TestClient

from app.api.v1.stubs import STUBBED_ENDPOINTS


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("method,path", STUBBED_ENDPOINTS)
def test_stub_returns_501_envelope(client: TestClient, method: str, path: str) -> None:
    request_path = path.replace("{file_id}", "file_123").replace(
        "{batch_id}", "batch_123"
    )
    resp = client.request(method, request_path, json={})
    assert resp.status_code == 501, f"{method} {request_path}"
    body = resp.json()
    assert body["error"]["type"] == "not_supported_error"
    assert body["error"]["code"] == "endpoint_not_supported"
    assert request_path in body["error"]["param"]
    assert "not supported" in body["error"]["message"]


def test_embeddings_no_longer_stubbed(client: TestClient) -> None:
    """Phase 18: /v1/embeddings is a real endpoint, not a 501 stub. A request
    missing 'model' now reaches the route and answers 400 (not 501)."""
    assert ("POST", "/v1/embeddings") not in STUBBED_ENDPOINTS
    resp = client.post("/v1/embeddings", json={})
    assert resp.status_code == 400
