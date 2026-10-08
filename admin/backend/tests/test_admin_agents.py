"""Phase 16 admin provider-agent reads + delete (additive)."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.models import (
    DefinitionAgent,
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
)
from app.services import redis_keys


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


def _instance(
    session: Session,
    agent: ProviderAgent,
    definition: ProviderDefinition,
    backend_status: str = "stopped",
) -> ProviderInstance:
    i = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=definition.id,
        backend_status=backend_status,
    )
    session.add(i)
    session.commit()
    session.refresh(i)
    return i


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


# ---------------------------------------------------------------------------
# DELETE /admin/api/agents/{agent_id}
# ---------------------------------------------------------------------------


def test_delete_agent_404(client: TestClient) -> None:
    assert client.delete("/admin/api/agents/nope").status_code == 404
    assert (
        client.delete(
            "/admin/api/agents/11111111-1111-1111-1111-111111111111"
        ).status_code
        == 404
    )


def test_delete_agent_409_when_connected(
    client: TestClient, session: Session
) -> None:
    m = _machine(session)
    a = _agent(session, m, "live-agent")
    a.websocket_connected = True
    session.add(a)
    session.commit()

    resp = client.delete(f"/admin/api/agents/{a.id}")
    assert resp.status_code == 409
    assert "connected" in resp.json()["detail"]
    # Row untouched.
    assert session.get(ProviderAgent, a.id) is not None


def test_delete_agent_disconnected_removes_cascade_and_redis(
    client: TestClient, session: Session, clean_redis
) -> None:
    m = _machine(session)
    d = _definition(session)
    a = _agent(session, m, "ghost-agent")
    inst = _instance(session, a, d, backend_status="stopped")
    session.add(DefinitionAgent(provider_definition_id=d.id, agent_id=a.id))
    session.commit()

    # Seed the agent's Redis keys (keyed by the ProviderAgent PK uuid).
    pk = str(a.id)
    for key in (
        redis_keys.secret_key(pk),
        redis_keys.epoch_key(pk),
        redis_keys.owner_key(pk),
        redis_keys.presence_key(pk),
        redis_keys.metrics_cats_key(pk),
    ):
        clean_redis.set(key, "x")

    # Seed an in-process scheduler VRAM hold so we can assert it is released,
    # plus the Redis im:vram:used mirror entry _mark_stopped is expected to hdel.
    scheduler = client.app.state.scheduler
    scheduler._booted[str(inst.id)] = (m.uid, 1234)
    vram_key = redis_keys.vram_used_key(m.uid)
    clean_redis.hset(vram_key, str(inst.id), "1234")

    agent_pk, inst_pk, def_pk = a.id, inst.id, d.id
    resp = client.delete(f"/admin/api/agents/{a.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"ok": True, "agent_id": str(agent_pk), "deleted_instances": 1}

    # The endpoint commits in its own session; drop this test session's
    # identity-map cache so the reads below hit the DB (ids captured above so
    # we never touch an expired attribute on a deleted row).
    session.expire_all()
    # Agent + children gone (explicit cascade in the endpoint).
    assert session.get(ProviderAgent, agent_pk) is None
    assert session.get(ProviderInstance, inst_pk) is None
    assert (
        session.exec(
            select(DefinitionAgent).where(DefinitionAgent.agent_id == agent_pk)
        ).first()
        is None
    )
    # The definition itself survives (only the link is dropped).
    assert session.get(ProviderDefinition, def_pk) is not None

    # Redis keys cleared.
    for key in (
        redis_keys.secret_key(pk),
        redis_keys.epoch_key(pk),
        redis_keys.owner_key(pk),
        redis_keys.presence_key(pk),
        redis_keys.metrics_cats_key(pk),
    ):
        assert clean_redis.get(key) is None

    # Scheduler ledger no longer holds the removed backend.
    assert str(inst_pk) not in scheduler._booted
    # ...and the Redis im:vram:used mirror entry was hdeld by _mark_stopped.
    assert clean_redis.hget(vram_key, str(inst_pk)) is None


def test_delete_agent_running_ghost_disconnected(
    client: TestClient, session: Session
) -> None:
    # Real-world case: a renamed container left a `running` backend row but its
    # socket is gone (websocket_connected False). It must still be deletable.
    m = _machine(session)
    d = _definition(session)
    a = _agent(session, m, "renamed-agent")
    a.websocket_connected = False
    session.add(a)
    session.commit()
    inst = _instance(session, a, d, backend_status="running")

    agent_pk, inst_pk = a.id, inst.id
    resp = client.delete(f"/admin/api/agents/{a.id}")
    assert resp.status_code == 200
    assert resp.json()["deleted_instances"] == 1
    session.expire_all()
    assert session.get(ProviderAgent, agent_pk) is None
    assert session.get(ProviderInstance, inst_pk) is None
