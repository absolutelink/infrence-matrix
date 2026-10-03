"""Tests for the server-instance model metadata endpoint (reasoning block)."""

import uuid as uuid_module
from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import Agent, Model, ServerInstance


def _make_instance(db: Session) -> ServerInstance:
    model = Model(
        name=f"meta-{uuid_module.uuid4().hex[:8]}.gguf",
        path="/models/meta.gguf",
        size_bytes=1,
        architecture="llama",
        quantization="Q4_K_M",
        source="local",
    )
    db.add(model)
    agent = Agent(name=f"meta-agent-{uuid_module.uuid4().hex[:8]}", host="h", port=1)
    db.add(agent)
    instance = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias="meta-alias",
        process_command="llama-server",
        status="stopped",
        started_at=datetime.now(UTC),
    )
    db.add(instance)
    db.commit()
    db.refresh(instance)
    return instance


class TestUpdateServerMetadataReasoning:
    def test_saves_reasoning_block(self, client: TestClient, db: Session) -> None:
        instance = _make_instance(db)
        response = client.put(
            f"/api/v1/server-instances/{instance.id}/metadata",
            json={
                "reasoning": {
                    "supported": True,
                    "efforts": ["low", "medium", "high"],
                    "default": "medium",
                }
            },
        )
        assert response.status_code == 200
        saved = response.json()["model_metadata"]["reasoning"]
        assert saved["efforts"] == ["low", "medium", "high"]
        assert saved["default"] == "medium"

    def test_rejects_default_outside_efforts(
        self, client: TestClient, db: Session
    ) -> None:
        instance = _make_instance(db)
        response = client.put(
            f"/api/v1/server-instances/{instance.id}/metadata",
            json={"reasoning": {"supported": True, "efforts": ["low"], "default": "high"}},
        )
        assert response.status_code == 422

    def test_rejects_unknown_effort(self, client: TestClient, db: Session) -> None:
        instance = _make_instance(db)
        response = client.put(
            f"/api/v1/server-instances/{instance.id}/metadata",
            json={"reasoning": {"supported": True, "efforts": ["turbo"]}},
        )
        assert response.status_code == 422

    def test_clearing_reasoning_removes_block(
        self, client: TestClient, db: Session
    ) -> None:
        instance = _make_instance(db)
        client.put(
            f"/api/v1/server-instances/{instance.id}/metadata",
            json={"reasoning": {"supported": True, "efforts": ["low"]}},
        )
        response = client.put(
            f"/api/v1/server-instances/{instance.id}/metadata",
            json={"reasoning": None},
        )
        assert response.status_code == 200
        assert "reasoning" not in response.json()["model_metadata"]
