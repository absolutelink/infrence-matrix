"""Model round-trip and constraint tests against the live test database."""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.models import (
    Machine,
    ProviderDefinition,
    ProviderInstance,
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
        "registration_token": "tok-" + uuid.uuid4().hex[:12],
        "backend_config": {"model": {"file": "x.gguf"}},
        "vram_required_bytes": 8 * 1024**3,
    }
    data.update(kw)
    d = ProviderDefinition(**data)
    session.add(d)
    session.commit()
    session.refresh(d)
    return d


def test_machine_reachable_address_prefers_dns() -> None:
    m = Machine(uid="u", name="n", host="h", dns="d", ip="i")
    assert m.reachable_address() == "d"
    m.dns = None
    assert m.reachable_address() == "h"
    m.host = None
    assert m.reachable_address() == "i"


def test_provider_instance_roundtrip(session: Session) -> None:
    m = _machine(session)
    d = _definition(session)
    inst = ProviderInstance(
        machine_id=m.id,
        provider_definition_id=d.id,
        port=8081,
        instance_status="running",
        backend_status="running",
        config_fingerprint="abc123",
        assigned_gpus=["gpu-uuid-1"],
    )
    session.add(inst)
    session.commit()
    session.refresh(inst)

    assert inst.machine.uid == m.uid
    assert inst.provider_definition.alias == d.alias
    assert inst.assigned_gpus == ["gpu-uuid-1"]
    assert inst.epoch == 0
    assert inst.websocket_connected is False


def test_duplicate_machine_definition_instance_rejected(session: Session) -> None:
    m = _machine(session)
    d = _definition(session)
    session.add(ProviderInstance(machine_id=m.id, provider_definition_id=d.id))
    session.commit()
    with pytest.raises(IntegrityError):
        session.add(ProviderInstance(machine_id=m.id, provider_definition_id=d.id))
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
