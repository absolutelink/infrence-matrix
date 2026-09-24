"""Tests for V1 Models API endpoint (OpenAI-compatible server aliases)."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import Agent, Model, ServerInstance


def _make_model(db: Session, name: str) -> Model:
    """Create a Model row (required parent of a ServerInstance)."""
    model = Model(
        name=name,
        path=f"/models/{name}",
        size_bytes=1234567890,
        architecture="llama",
        model_type="llm",
        parameter_count=3000000000,
        quantization="Q4_K_M",
        supports_embeddings=False,
        supports_vision=False,
        source="huggingface",
        source_repo_id="test/repo",
    )
    db.add(model)
    return model


def _make_agent(db: Session, name: str) -> Agent:
    """Create an Agent row (required parent of a ServerInstance)."""
    agent = Agent(
        name=name,
        host="localhost",
        port=8080,
        status="online",
    )
    db.add(agent)
    return agent


def _make_instance(
    db: Session, model: Model, agent: Agent, alias: str, status: str
) -> ServerInstance:
    """Create a ServerInstance row."""
    instance = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias=alias,
        process_command="llama-server",
        status=status,
        started_at=datetime.now(UTC),
    )
    db.add(instance)
    return instance


class TestListModels:
    """Test GET /v1/models endpoint."""

    def test_list_models_empty(self, client: TestClient, db: Session) -> None:
        """Test listing models when no server instances exist."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_models_with_running_instance(
        self, client: TestClient, db: Session
    ) -> None:
        """Running instances with an alias are listed."""
        model = _make_model(db, "test-model.Q4_K_M.gguf")
        agent = _make_agent(db, "agent-1")
        _make_instance(db, model, agent, "test-alias", "running")
        db.commit()

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-alias"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "test-model.Q4_K_M.gguf"

    def test_list_models_multiple(self, client: TestClient, db: Session) -> None:
        """Multiple running instances are all listed."""
        model1 = _make_model(db, "model1.Q4_K_M.gguf")
        model2 = _make_model(db, "model2.Q5_K_M.gguf")
        agent = _make_agent(db, "agent-1")
        _make_instance(db, model1, agent, "alias1", "running")
        _make_instance(db, model2, agent, "alias2", "starting")
        db.commit()

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert {m["id"] for m in data["data"]} == {"alias1", "alias2"}

    def test_list_models_skips_offline_instances(
        self, client: TestClient, db: Session
    ) -> None:
        """Stopped instances are not listed."""
        model = _make_model(db, "test-model.Q4_K_M.gguf")
        agent = _make_agent(db, "agent-1")
        _make_instance(db, model, agent, "alias-running", "running")
        _make_instance(db, model, agent, "alias-stopped", "stopped")
        db.commit()

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert [m["id"] for m in data["data"]] == ["alias-running"]

    def test_list_models_skips_instances_without_alias(
        self, client: TestClient, db: Session
    ) -> None:
        """Instances without an alias are skipped."""
        model = _make_model(db, "test-model.Q4_K_M.gguf")
        agent = _make_agent(db, "agent-1")
        instance = _make_instance(db, model, agent, "has-alias", "running")
        instance.alias = ""
        db.add(instance)
        db.commit()

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []


class TestRetrieveModel:
    """Test GET /v1/models/{model_id} endpoint."""

    def test_retrieve_model_exists(self, client: TestClient, db: Session) -> None:
        """Test retrieving an existing model alias."""
        model = _make_model(db, "test-model.Q4_K_M.gguf")
        agent = _make_agent(db, "agent-1")
        _make_instance(db, model, agent, "test-alias", "running")
        db.commit()

        response = client.get("/v1/models/test-alias")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-alias"
        assert data["object"] == "model"
        assert data["owned_by"] == "test-model.Q4_K_M.gguf"

    def test_retrieve_model_not_found(self, client: TestClient, db: Session) -> None:
        """Test retrieving a non-existent model alias."""
        response = client.get("/v1/models/nonexistent.gguf")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
