"""Tests for V1 Models API endpoint."""

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import Model


class TestListModels:
    """Test GET /v1/models endpoint."""

    def test_list_models_empty(self, client: TestClient, db: Session) -> None:
        """Test listing models when no models exist."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_models_with_data(self, client: TestClient, db: Session) -> None:
        """Test listing models with existing models."""
        model = Model(
            name="test-model.Q4_K_M.gguf",
            path="/models/test-model.Q4_K_M.gguf",
            size_bytes=1234567890,
            architecture="llama",
            parameter_count=3000000000,
            quantization="Q4_K_M",
            supports_embeddings=False,
            supports_vision=False,
            context_length=4096,
            source="huggingface",
            source_repo_id="test/repo",
        )
        db.add(model)
        db.commit()

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model.Q4_K_M.gguf"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "inference-matrix"

    def test_list_models_multiple(self, client: TestClient, db: Session) -> None:
        """Test listing multiple models."""
        models = [
            Model(
                name="model1.Q4_K_M.gguf",
                path="/models/model1.gguf",
                size_bytes=1000000000,
                architecture="llama",
                parameter_count=1000000000,
                quantization="Q4_K_M",
                supports_embeddings=False,
                supports_vision=False,
                context_length=4096,
                source="huggingface",
            ),
            Model(
                name="model2.Q5_K_M.gguf",
                path="/models/model2.gguf",
                size_bytes=2000000000,
                architecture="llama",
                parameter_count=2000000000,
                quantization="Q5_K_M",
                supports_embeddings=True,
                supports_vision=False,
                context_length=8192,
                source="huggingface",
            ),
        ]
        db.add_all(models)
        db.commit()

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2


class TestRetrieveModel:
    """Test GET /v1/models/{model_id} endpoint."""

    def test_retrieve_model_exists(self, client: TestClient, db: Session) -> None:
        """Test retrieving an existing model."""
        model = Model(
            name="test-model.Q4_K_M.gguf",
            path="/models/test-model.Q4_K_M.gguf",
            size_bytes=1234567890,
            architecture="llama",
            parameter_count=3000000000,
            quantization="Q4_K_M",
            supports_embeddings=False,
            supports_vision=False,
            context_length=4096,
            source="huggingface",
        )
        db.add(model)
        db.commit()

        response = client.get("/v1/models/test-model.Q4_K_M.gguf")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-model.Q4_K_M.gguf"
        assert data["object"] == "model"
        assert data["owned_by"] == "inference-matrix"

    def test_retrieve_model_not_found(self, client: TestClient, db: Session) -> None:
        """Test retrieving a non-existent model."""
        response = client.get("/v1/models/nonexistent.gguf")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
