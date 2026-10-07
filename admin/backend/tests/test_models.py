"""Model round-trip and constraint tests against the live test database."""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.models import (
    DefinitionAgent,
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
    ResponseRecord,
    TokenUsageSample,
)


def _machine(session: Session, **kw) -> Machine:
    data = {
        "uid": "m-" + uuid.uuid4().hex[:8],
        "name": "machine-" + uuid.uuid4().hex[:8],
        "dns": "host.example",
        "total_vram_bytes": 24 * 1024**3,
    }
    data.update(kw)
    m = Machine(**data)
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def _definition(session: Session, **kw) -> ProviderDefinition:
    data = {
        "alias": "model-" + uuid.uuid4().hex[:8],
        "provider_type": "llama-cpp",
        "backend_config": {"model": {"file": "x.gguf"}},
        "vram_required_bytes": 8 * 1024**3,
    }
    data.update(kw)
    d = ProviderDefinition(**data)
    session.add(d)
    session.commit()
    session.refresh(d)
    return d


def _agent(session: Session, m: Machine, **kw) -> ProviderAgent:
    data = {
        "machine_id": m.id,
        "provider_type": "llama-cpp",
        "agent_id": "agent-" + uuid.uuid4().hex[:8],
        "base_port": 8081,
        "version": "dev",
    }
    data.update(kw)
    a = ProviderAgent(**data)
    session.add(a)
    session.commit()
    session.refresh(a)
    return a


def test_machine_reachable_address_prefers_dns() -> None:
    m = Machine(uid="u", name="n", host="h", dns="d", ip="i")
    assert m.reachable_address() == "d"
    m.dns = None
    assert m.reachable_address() == "h"
    m.host = None
    assert m.reachable_address() == "i"


def test_provider_instance_roundtrip(session: Session) -> None:
    m = _machine(session)
    a = _agent(session, m)
    d = _definition(session)
    inst = ProviderInstance(
        agent_id=a.id,
        provider_definition_id=d.id,
        port=8081,
        backend_status="running",
        config_fingerprint="abc123",
        assigned_gpus=["gpu-uuid-1"],
    )
    session.add(inst)
    session.commit()
    session.refresh(inst)

    assert inst.machine.uid == m.uid
    assert inst.agent.id == a.id
    assert inst.provider_definition.alias == d.alias
    assert inst.assigned_gpus == ["gpu-uuid-1"]


def test_duplicate_agent_definition_instance_rejected(session: Session) -> None:
    m = _machine(session)
    a = _agent(session, m)
    d = _definition(session)
    session.add(ProviderInstance(agent_id=a.id, provider_definition_id=d.id))
    session.commit()
    with pytest.raises(IntegrityError):
        session.add(ProviderInstance(agent_id=a.id, provider_definition_id=d.id))
        session.commit()


def test_response_record_chain(session: Session) -> None:
    d = _definition(session)
    parent = ResponseRecord(
        response_id="resp_parent_" + uuid.uuid4().hex[:8],
        provider_definition_id=d.id,
        status="completed",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
    )
    session.add(parent)
    session.commit()
    child = ResponseRecord(
        response_id="resp_child_" + uuid.uuid4().hex[:8],
        previous_response_id=parent.response_id,
        provider_definition_id=d.id,
        status="completed",
    )
    session.add(child)
    session.commit()
    session.refresh(child)
    assert child.previous_response_id == parent.response_id
    assert child.provider_definition.id == d.id


def test_token_usage_sample_defaults(session: Session) -> None:
    s = TokenUsageSample(prompt_tokens=5, completion_tokens=7)
    session.add(s)
    session.commit()
    session.refresh(s)
    assert s.cached_tokens == 0
    assert s.created_at is not None


def test_unique_constraints(session: Session) -> None:
    with pytest.raises(IntegrityError):
        _machine(session, uid="dup")
        _machine(session, uid="dup")
        session.commit()


# --- Phase 16: machine-scoped provider agents (additive foundation) ---------


def _ptype(session: Session, name: str = "llama-cpp", **kw) -> ProviderType:
    data = {
        "name": name,
        "schema": {"type": "object"},
        "schema_fingerprint": "fp-" + name,
        "status": "active",
    }
    data.update(kw)
    pt = ProviderType(**data)
    session.add(pt)
    session.commit()
    session.refresh(pt)
    return pt


def test_machine_has_generated_registration_secret(session: Session) -> None:
    m = _machine(session)
    assert m.registration_secret  # default_factory minted a non-empty secret


def test_provider_type_max_running_backends_defaults_zero(session: Session) -> None:
    pt = _ptype(session)
    assert pt.max_running_backends == 0  # 0 = unlimited


def test_apply_max_running_from_schema_reads_x_key(session: Session) -> None:
    """Phase 16 slice 6: the admin extracts the per-agent running cap from the
    committed schema's top-level ``x-max-running-backends``. halogen-flash ships
    ``1`` (single NPU); an absent key defaults to unlimited (0); a non-int is
    coerced to 0 rather than crashing registration."""
    from app.api.admin.providers import _apply_max_running_from_schema

    pt = _ptype(session, max_running_backends=0)
    _apply_max_running_from_schema(pt, {"type": "object", "x-max-running-backends": 1})
    assert pt.max_running_backends == 1

    # Absent key -> unlimited.
    _apply_max_running_from_schema(pt, {"type": "object"})
    assert pt.max_running_backends == 0

    # Non-integer value -> coerced to 0 (never raises).
    _apply_max_running_from_schema(pt, {"x-max-running-backends": "many"})
    assert pt.max_running_backends == 0


def test_provider_definition_placement_defaults_any_of_type(
    session: Session,
) -> None:
    d = _definition(session)
    assert d.agent_placement == "any_of_type"


def test_provider_agent_roundtrip(session: Session) -> None:
    m = _machine(session)
    agent = ProviderAgent(
        machine_id=m.id,
        provider_type="llama-cpp",
        agent_id="agent-a",
        base_port=8081,
        version="dev",
    )
    session.add(agent)
    session.commit()
    session.refresh(agent)

    assert agent.machine.uid == m.uid
    assert agent.agent_status == "registering"
    assert agent.websocket_connected is False
    assert agent.epoch == 0
    assert agent.assigned_gpus == []


def test_provider_agent_unique_on_machine_type_agent(session: Session) -> None:
    m = _machine(session)
    session.add(
        ProviderAgent(machine_id=m.id, provider_type="llama-cpp", agent_id="dup")
    )
    session.commit()
    with pytest.raises(IntegrityError):
        session.add(
            ProviderAgent(machine_id=m.id, provider_type="llama-cpp", agent_id="dup")
        )
        session.commit()


def test_two_agents_share_machine_and_type_via_agent_id(session: Session) -> None:
    m = _machine(session)
    a = ProviderAgent(machine_id=m.id, provider_type="llama-cpp", agent_id="a")
    b = ProviderAgent(machine_id=m.id, provider_type="llama-cpp", agent_id="b")
    session.add_all([a, b])
    session.commit()
    session.refresh(m)
    assert {agent.agent_id for agent in m.agents} == {"a", "b"}


def test_definition_specific_placement_link(session: Session) -> None:
    m = _machine(session)
    agent = ProviderAgent(
        machine_id=m.id, provider_type="llama-cpp", agent_id="pick-me"
    )
    session.add(agent)
    session.commit()
    d = _definition(session, agent_placement="specific")
    session.add(DefinitionAgent(provider_definition_id=d.id, agent_id=agent.id))
    session.commit()
    session.refresh(d)
    session.refresh(agent)
    assert [a.id for a in d.agents] == [agent.id]
    assert [defn.id for defn in agent.definitions] == [d.id]
