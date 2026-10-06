"""Registration endpoint validation and upsert behavior."""

import hashlib
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.admin.providers import compute_config_fingerprint
from app.core.config import settings
from app.models import Machine, ProviderDefinition, ProviderInstance
from app.services import redis_keys


def _machine(session: Session, uid: str = "mach-1") -> Machine:
    m = Machine(uid=uid, name=f"name-{uid}", host="10.0.0.5", dns="box.local")
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def _definition(
    session: Session,
    *,
    token: str = "reg-token-1",
    provider_type: str = "mock",
    enabled: bool = True,
    backend_config: dict | None = None,
) -> ProviderDefinition:
    d = ProviderDefinition(
        alias=f"alias-{uuid.uuid4().hex[:8]}",
        provider_type=provider_type,
        registration_token=token,
        backend_config=backend_config
        or {"model": {"file": "m.gguf"}, "args": {"ctx": 4096}},
        vram_required_bytes=8 * 1024**3,
        idle_timeout_seconds=123,
        capacity=2,
        enabled=enabled,
    )
    session.add(d)
    session.commit()
    session.refresh(d)
    return d


def _body(**overrides) -> dict:
    body = {
        "machine_uid": "mach-1",
        "registration_token": "reg-token-1",
        "provider_type": "mock",
        "schema": {"type": "object"},
        "version": settings.VERSION,
        "port": 8081,
        "hardware": {
            "gpus": [{"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3}],
            "total_vram_bytes": 24 * 1024**3,
        },
        "metrics_categories": ["vram"],
        "registered_at": "2026-01-01T00:00:00+00:00",
    }
    body.update(overrides)
    return body


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_register_success(client: TestClient, session: Session, clean_redis) -> None:
    machine = _machine(session)
    definition = _definition(session)

    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["instance_id"]
    assert len(data["instance_secret"]) >= 32
    assert data["machine"]["uid"] == machine.uid
    assert data["provider_definition"]["alias"] == definition.alias
    assert data["provider_definition"]["provider_type"] == "mock"
    assert data["provider_definition"]["idle_timeout_seconds"] == 123
    assert data["provider_definition"]["capacity"] == 2
    assert data["provider_definition"]["vram_required_bytes"] == 8 * 1024**3

    expected_fp = hashlib.sha256(
        json.dumps(
            definition.backend_config, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert data["provider_definition"]["config_fingerprint"] == expected_fp
    assert compute_config_fingerprint(definition.backend_config) == expected_fp

    inst = session.exec(select(ProviderInstance)).one()
    assert str(inst.id) == data["instance_id"]
    assert inst.machine_id == machine.id
    assert inst.provider_definition_id == definition.id
    assert inst.port == 8081
    assert inst.version == settings.VERSION
    assert inst.instance_status == "registering"
    assert inst.config_fingerprint == expected_fp

    # Secret lives in Redis, not Postgres.
    stored = clean_redis.get(redis_keys.secret_key(data["instance_id"]))
    assert stored == data["instance_secret"]

    session.refresh(machine)
    assert machine.hardware == _body()["hardware"]
    assert machine.total_vram_bytes == 24 * 1024**3


def test_register_reregister_reuses_instance_row(
    client: TestClient, session: Session
) -> None:
    _machine(session)
    _definition(session)

    first = client.post("/admin/api/providers/register", json=_body(port=9000)).json()
    second = client.post("/admin/api/providers/register", json=_body(port=9100)).json()

    assert first["instance_id"] == second["instance_id"]
    assert first["instance_secret"] != second["instance_secret"]
    assert len(session.exec(select(ProviderInstance)).all()) == 1
    inst = session.exec(select(ProviderInstance)).one()
    assert inst.port == 9100


def test_register_unknown_token(client: TestClient, session: Session) -> None:
    _machine(session)
    _definition(session, token="other-token")
    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 401


def test_register_disabled_definition(client: TestClient, session: Session) -> None:
    _machine(session)
    _definition(session, enabled=False)
    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 403


def test_register_provider_type_mismatch(client: TestClient, session: Session) -> None:
    _machine(session)
    _definition(session, provider_type="llama-cpp")
    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 409
    assert "provider type mismatch" in resp.json()["detail"]


def test_register_version_mismatch(client: TestClient, session: Session) -> None:
    _machine(session)
    _definition(session)
    resp = client.post(
        "/admin/api/providers/register",
        json=_body(version="definitely-not-" + settings.VERSION),
    )
    assert resp.status_code == 409
    assert "version mismatch" in resp.json()["detail"]


def test_register_unknown_machine(client: TestClient, session: Session) -> None:
    _definition(session)
    resp = client.post(
        "/admin/api/providers/register", json=_body(machine_uid="ghost-machine")
    )
    assert resp.status_code == 404


def test_register_hardware_without_total_keeps_machine_vram(
    client: TestClient, session: Session
) -> None:
    machine = _machine(session)
    machine.total_vram_bytes = 99 * 1024**3
    session.add(machine)
    session.commit()
    _definition(session)

    body = _body(hardware={"gpus": []})
    resp = client.post("/admin/api/providers/register", json=body)
    assert resp.status_code == 200
    session.refresh(machine)
    assert machine.total_vram_bytes == 99 * 1024**3
    assert machine.hardware == {"gpus": []}
