"""Phase 16 admin provider-agent reads (additive)."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import DefinitionAgent, Machine, ProviderAgent, ProviderDefinition


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _machine(session: Session) -> Machine:
    m = Machine(
        uid="m-" + uuid.uuid4().hex[:8],
        name="machine-" + uuid.uuid4().hex[:8],
        dns="host.example",
    )
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def _definition(session: Session) -> ProviderDefinition:
    d = ProviderDefinition(
        alias="model-" + uuid.uuid4().hex[:8],
        provider_type="llama-cpp",
        registration_token="tok-" + uuid.uuid4().hex[:12],
        backend_config={"model": {"file": "x.gguf"}},
    )
    session.add(d)
    session.commit()
    session.refresh(d)
    return d


def _agent(session: Session, machine: Machine, agent_id: str = "a1") -> ProviderAgent:
    a = ProviderAgent(
        machine_id=machine.id,
        provider_type="llama-cpp",
        agent_id=agent_id,
        base_port=8081,
    )
    session.add(a)
    session.commit()
    session.refresh(a)
    return a


def test_list_agents_empty(client: TestClient) -> None:
    resp = client.get("/admin/api/agents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_agents_shows_created(client: TestClient, session: Session) -> None:
    m = _machine(session)
    a = _agent(session, m, "agent-x")
    resp = client.get("/admin/api/agents")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == str(a.id)
    assert row["agent_id"] == "agent-x"
    assert row["machine_uid"] == m.uid
    assert row["provider_type"] == "llama-cpp"
    assert row["base_port"] == 8081
    assert row["agent_status"] == "registering"
    assert row["definition_count"] == 0


def test_get_agent_detail_with_definition_count(
    client: TestClient, session: Session
) -> None:
    m = _machine(session)
    a = _agent(session, m, "agent-y")
    d = _definition(session)
    session.add(DefinitionAgent(provider_definition_id=d.id, agent_id=a.id))
    session.commit()

    resp = client.get(f"/admin/api/agents/{a.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == "agent-y"
    assert body["definition_count"] == 1


def test_get_unknown_agent_404(client: TestClient) -> None:
    assert client.get("/admin/api/agents/nope").status_code == 404
    assert (
        client.get("/admin/api/agents/11111111-1111-1111-1111-111111111111").status_code
        == 404
    )
