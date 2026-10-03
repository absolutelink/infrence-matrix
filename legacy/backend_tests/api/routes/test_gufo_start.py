"""Route-level tests for starting gufo server instances."""

import uuid as uuid_module

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.routes import server_instances as route_module
from app.models import Agent, Model


@pytest.fixture(autouse=True)
def _no_background_init(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*args: object, **kwargs: object) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(route_module, "_initialize_server", _noop)


def _add_gufo_agent(
    db: Session, *, platform: str = "gufo", type_: str = "gufo"
) -> Agent:
    agent = Agent(
        name=f"gufo-agent-{uuid_module.uuid4().hex[:8]}",
        platform=platform,
        type=type_,
        host="h",
        port=8080,
        status="online",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _add_model(db: Session) -> Model:
    model = Model(
        name=f"gufo-{uuid_module.uuid4().hex[:8]}.gguf",
        path="/models/gufo.gguf",
        size_bytes=1,
        architecture="qwen3",
        model_type="llm",
        quantization="Q8_K_XL",
        source="local",
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def test_start_gufo_server_accepts_gufo_options(
    client: TestClient, db: Session
) -> None:
    agent = _add_gufo_agent(db)
    model = _add_model(db)

    response = client.post(
        "/api/v1/server-instances/start",
        json={
            "alias": f"gufo-{uuid_module.uuid4().hex[:8]}",
            "engine": "gufo",
            "model_id": str(model.id),
            "agent_id": str(agent.id),
            "engine_options": {
                "context": 131072,
                "speculative": "dflash2",
                "think": "on",
                "sessions": 2,
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "preparing"

    from app.models import ServerInstance

    instance = db.get(ServerInstance, uuid_module.UUID(body["server_id"]))
    assert instance is not None
    assert instance.engine == "gufo"
    assert instance.engine_options["speculative"] == "dflash2"
    assert instance.engine_options["sessions"] == 2
    assert instance.model_id == model.id


def test_start_gufo_rejects_llama_server_options(
    client: TestClient, db: Session
) -> None:
    agent = _add_gufo_agent(db)
    model = _add_model(db)

    response = client.post(
        "/api/v1/server-instances/start",
        json={
            "alias": f"gufo-{uuid_module.uuid4().hex[:8]}",
            "engine": "gufo",
            "model_id": str(model.id),
            "agent_id": str(agent.id),
            "server_options": {"parallel": 4},
        },
    )
    assert response.status_code == 400
    assert "not valid for Gufo" in response.json()["detail"]


def test_start_gufo_requires_matching_agent_type(
    client: TestClient, db: Session
) -> None:
    agent = _add_gufo_agent(db, type_="rocm")
    model = _add_model(db)

    response = client.post(
        "/api/v1/server-instances/start",
        json={
            "alias": f"gufo-{uuid_module.uuid4().hex[:8]}",
            "engine": "gufo",
            "model_id": str(model.id),
            "agent_id": str(agent.id),
        },
    )
    assert response.status_code == 400
    assert "platform=gufo and type=gufo" in response.json()["detail"]


def test_start_llamacpp_rejects_gufo_agent(client: TestClient, db: Session) -> None:
    agent = _add_gufo_agent(db)
    model = _add_model(db)

    response = client.post(
        "/api/v1/server-instances/start",
        json={
            "alias": f"llama-{uuid_module.uuid4().hex[:8]}",
            "engine": "llamacpp",
            "model_id": str(model.id),
            "agent_id": str(agent.id),
        },
    )
    assert response.status_code == 400
    assert "Gufo agents require engine=gufo" in response.json()["detail"]


def test_start_gufo_rejects_unknown_option_key(client: TestClient, db: Session) -> None:
    agent = _add_gufo_agent(db)
    model = _add_model(db)

    response = client.post(
        "/api/v1/server-instances/start",
        json={
            "alias": f"gufo-{uuid_module.uuid4().hex[:8]}",
            "engine": "gufo",
            "model_id": str(model.id),
            "agent_id": str(agent.id),
            "engine_options": {"parallel": 4},
        },
    )
    assert response.status_code == 422
