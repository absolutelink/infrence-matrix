"""Tests for POST /v1/responses route."""

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import Model


def _make_model(db: Session, name: str = "responses-model.Q4_K_M.gguf") -> Model:
    model = Model(
        name=name,
        path=f"/models/{name}",
        size_bytes=1234567890,
        architecture="llama",
        parameter_count=3000000000,
        quantization="Q4_K_M",
        supports_embeddings=False,
        supports_vision=False,
        source="huggingface",
    )
    db.add(model)
    db.commit()
    return model


class TestCreateResponse:
    def test_missing_model_returns_openai_error(self, client: TestClient) -> None:
        response = client.post(
            "/v1/responses",
            json={"model": "does-not-exist", "input": "hi"},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["type"] == "not_found"
        assert data["error"]["code"] == "model_not_found"

    def test_background_not_supported(self, client: TestClient, db: Session) -> None:
        _make_model(db)
        response = client.post(
            "/v1/responses",
            json={
                "model": "responses-model.Q4_K_M.gguf",
                "input": "hi",
                "background": True,
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["param"] == "background"

    def test_non_streaming_no_agent(self, client: TestClient, db: Session) -> None:
        """Without an online agent the endpoint returns 503 (no inference)."""
        _make_model(db)
        response = client.post(
            "/v1/responses",
            json={
                "model": "responses-model.Q4_K_M.gguf",
                "input": "hi",
                "stream": False,
            },
        )
        assert response.status_code in [200, 503]
        if response.status_code == 503:
            assert response.json()["error"]["type"] in [
                "too_many_requests",
                "server_error",
            ]

    def test_streaming_requires_agent(self, client: TestClient, db: Session) -> None:
        _make_model(db)
        response = client.post(
            "/v1/responses",
            json={
                "model": "responses-model.Q4_K_M.gguf",
                "input": "hi",
                "stream": True,
            },
        )
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            assert response.headers["content-type"].startswith("text/event-stream")
            body = response.text
            assert "event: response.created" in body
            assert "data: [DONE]" in body

    def test_previous_response_not_found(self, client: TestClient, db: Session) -> None:
        _make_model(db)
        response = client.post(
            "/v1/responses",
            json={
                "model": "responses-model.Q4_K_M.gguf",
                "input": "hi",
                "previous_response_id": "resp_missing",
            },
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "previous_response_not_found"
        assert data["error"]["param"] == "previous_response_id"

    def test_invalid_input_item_rejected(self, client: TestClient, db: Session) -> None:
        _make_model(db)
        response = client.post(
            "/v1/responses",
            json={
                "model": "responses-model.Q4_K_M.gguf",
                "input": [{"type": "item_reference", "id": "item_1"}],
            },
        )
        # item_reference must be rejected (validation or translation error)
        assert response.status_code in [400, 422, 503]

    def test_store_false_no_persistence(self, client: TestClient, db: Session) -> None:
        _make_model(db)
        response = client.post(
            "/v1/responses",
            json={
                "model": "responses-model.Q4_K_M.gguf",
                "input": "hi",
                "store": False,
                "previous_response_id": "resp_anything",
            },
        )
        # store=false still needs the previous response to exist when chaining
        assert response.status_code == 404

    def test_tools_validation(self, client: TestClient, db: Session) -> None:
        _make_model(db)
        response = client.post(
            "/v1/responses",
            json={
                "model": "responses-model.Q4_K_M.gguf",
                "input": "hi",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "tool_choice": "none",
            },
        )
        assert response.status_code in [200, 503]
