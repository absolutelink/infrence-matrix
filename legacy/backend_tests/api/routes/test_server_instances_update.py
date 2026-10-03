"""Regression tests for the active-lease restart guard on PUT /server-instances/{id}."""

import uuid as uuid_module
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import Agent, InferenceLease, Model, ServerInstance


def _make_running_instance(db: Session) -> ServerInstance:
    model = Model(
        name=f"lease-{uuid_module.uuid4().hex[:8]}.gguf",
        path="/models/lease.gguf",
        size_bytes=1,
        architecture="llama",
        quantization="Q4_K_M",
        source="local",
    )
    db.add(model)
    agent = Agent(name=f"lease-agent-{uuid_module.uuid4().hex[:8]}", host="h", port=1)
    db.add(agent)
    instance = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias=f"lease-alias-{uuid_module.uuid4().hex[:8]}",
        process_command="llama-server",
        status="running",
        started_at=datetime.now(UTC),
    )
    db.add(instance)
    db.commit()
    db.refresh(instance)
    return instance


def _add_active_lease(db: Session, instance: ServerInstance) -> None:
    lease = InferenceLease(
        request_id=f"chatcmpl-{uuid_module.uuid4()}",
        model_id=instance.model_id,
        server_instance_id=instance.id,
        status="active",
    )
    db.add(lease)
    db.commit()


class TestUpdateServerActiveLeaseGuard:
    @pytest.mark.parametrize("lease_count", [1, 2])
    def test_restart_blocked_by_active_leases(
        self, client: TestClient, db: Session, lease_count: int
    ) -> None:
        instance = _make_running_instance(db)
        for _ in range(lease_count):
            _add_active_lease(db, instance)

        response = client.put(
            f"/api/v1/server-instances/{instance.id}", json={"restart": True}
        )
        assert response.status_code == 409
        assert "active inference requests" in response.json()["detail"]
