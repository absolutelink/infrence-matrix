"""Tests for persisted benchmark definitions and queue endpoints."""

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import Agent, Model, ServerInstance


def _source(db: Session) -> ServerInstance:
    model = Model(
        name="bench-model.gguf",
        path="/models/bench-model.gguf",
        size_bytes=100,
        architecture="llama",
        quantization="Q4_K_M",
        source="huggingface",
        source_repo_id="test/repo",
        source_file="bench-model.gguf",
    )
    agent = Agent(name="bench-agent", host="localhost", port=8080, status="online")
    db.add(model)
    db.add(agent)
    db.flush()
    server = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias="bench-server",
        process_command="llama-server",
        gpu_layers=42,
        context_size=8192,
        flash_attn=False,
        status="stopped",
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def test_definition_crud_and_snapshot(client: TestClient, db: Session) -> None:
    server = _source(db)
    response = client.post(
        "/api/v1/benchmarks/definitions",
        json={
            "name": "Throughput",
            "source_server_instance_id": str(server.id),
            "config": {"prompt_sizes": [512, 2048], "repetitions": 5},
        },
    )
    assert response.status_code == 200
    definition = response.json()
    assert definition["source_server_instance_id"] == str(server.id)
    assert definition["config"]["gpu_layers"] == 42
    assert definition["config"]["context_size"] == 8192
    assert definition["config"]["prompt_sizes"] == [512, 2048]

    listed = client.get("/api/v1/benchmarks/definitions")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "Throughput"

    updated = client.patch(
        f"/api/v1/benchmarks/definitions/{definition['id']}",
        json={
            "name": "Updated throughput",
            "source_server_instance_id": str(server.id),
            "config": {"repetitions": 2},
        },
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Updated throughput"

    deleted = client.delete(f"/api/v1/benchmarks/definitions/{definition['id']}")
    assert deleted.status_code == 204
    assert client.get("/api/v1/benchmarks/definitions").json() == []


def test_run_requires_existing_definition(client: TestClient) -> None:
    response = client.post(
        "/api/v1/benchmarks/runs",
        json={"definition_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert response.status_code == 404
