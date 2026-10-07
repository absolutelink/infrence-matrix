"""Phase 9 admin provider-definitions CRUD + config-change push.

Phase 12: ``provider_type`` must reference a registered ``ProviderType``
row and ``backend_config`` is validated against that type's committed
JSON Schema. The autouse fixture below seeds the known types with a
permissive schema so the Phase 9 push/CRUD semantics stay the focus of
these tests; the schema-validation specifics are covered in
``test_provider_types.py``.

Phase 16: shells and ``registration_token`` are gone — ``provider_type`` +
``backend_config`` are required at create, and a config push addresses the
owning **agent** socket carrying the target ``instance_id``.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.admin.providers import compute_config_fingerprint
from app.models import (
    DefinitionAgent,
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
)
from app.services.hashing import canonical_json_sha256


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _seed_provider_types(session: Session) -> None:
    """Register the known types with a permissive (any-object) schema."""
    for name in ("mock", "llama-cpp", "halogen", "halogen-flash", "gufo"):
        if (
            session.exec(select(ProviderType).where(ProviderType.name == name)).first()
            is None
        ):
            session.add(
                ProviderType(
                    name=name,
                    schema={"type": "object"},
                    schema_fingerprint=canonical_json_sha256({"type": "object"}),
                    status="active",
                )
            )
    session.commit()


def _create(client: TestClient, **overrides) -> dict:
    body = {
        "alias": "def-model",
        "provider_type": "mock",
        "backend_config": {"delta_count": 3},
        "capacity": 1,
        "idle_timeout_seconds": 300,
        "vram_required_bytes": 0,
    }
    body.update(overrides)
    resp = client.post("/admin/api/definitions", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _connected_backend(
    session: Session, definition_id: str, *, connected: bool = True
) -> tuple[str, str]:
    """Attach an agent + one backend to a definition; returns
    (agent_id, instance_id)."""
    machine = Machine(
        uid=f"m-{uuid.uuid4().hex[:8]}",
        name=f"m-{uuid.uuid4().hex[:8]}",
        registration_secret="s",
    )
    session.add(machine)
    session.commit()
    agent = ProviderAgent(
        machine_id=machine.id,
        provider_type="mock",
        agent_id=f"ag-{uuid.uuid4().hex[:8]}",
        websocket_connected=connected,
    )
    session.add(agent)
    session.commit()
    instance = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=uuid.UUID(definition_id),
        config_fingerprint=compute_config_fingerprint({"delta_count": 3}),
    )
    session.add(instance)
    session.commit()
    return str(agent.id), str(instance.id)


def test_create_returns_fingerprint(client: TestClient) -> None:
    created = _create(client)
    assert created["alias"] == "def-model"
    assert created["config_fingerprint"] == compute_config_fingerprint(
        {"delta_count": 3}
    )
    assert created["provider_type"] == "mock"


def test_create_requires_provider_type(client: TestClient) -> None:
    """Phase 16: no shells — provider_type is required at create."""
    resp = client.post("/admin/api/definitions", json={"alias": "x"})
    assert resp.status_code == 422


def test_create_rejects_unknown_provider_type(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={"alias": "x", "provider_type": "not-a-type"},
    )
    assert resp.status_code == 422


def test_create_rejects_capacity_below_one(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={"alias": "x", "provider_type": "mock", "capacity": 0},
    )
    assert resp.status_code == 422


def test_create_rejects_negative_vram(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={"alias": "x", "provider_type": "mock", "vram_required_bytes": -5},
    )
    assert resp.status_code == 422


def test_create_alias_conflict(client: TestClient) -> None:
    _create(client, alias="taken")
    resp = client.post(
        "/admin/api/definitions",
        json={"alias": "taken", "provider_type": "mock", "backend_config": {}},
    )
    assert resp.status_code == 409


def test_create_warms_alias_registry(client: TestClient, monkeypatch) -> None:
    from app.api.admin import definitions as definitions_mod

    warmed: list[str] = []
    monkeypatch.setattr(
        definitions_mod.alias_registry,
        "ensure_registered",
        lambda alias: warmed.append(alias),
    )
    _create(client, alias="warm-me")
    assert "warm-me" in warmed


def test_get_list_and_delete(client: TestClient, session: Session) -> None:
    created = _create(client, alias="crud-model")
    got = client.get(f"/admin/api/definitions/{created['id']}")
    assert got.status_code == 200
    assert got.json()["alias"] == "crud-model"
    assert got.json()["instances"] == []

    lst = client.get("/admin/api/definitions").json()
    assert [d["alias"] for d in lst] == ["crud-model"]

    resp = client.delete(f"/admin/api/definitions/{created['id']}")
    assert resp.status_code == 200
    session.expunge_all()
    assert session.exec(select(ProviderDefinition)).first() is None


def test_patch_backend_config_triggers_push(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """backend_config change pushes provider.config.update to the connected
    backend (addressed to its agent) and the DB fingerprint updates on the
    ok ack."""
    from app.services import config_update as cu
    from app.services.wire import Frame

    created = _create(client, alias="push-model")
    definition_id = created["id"]
    agent_id, instance_id = _connected_backend(session, definition_id)

    sent: list[tuple[str, dict]] = []

    async def fake_send_command(aid, type_, payload, timeout=30.0):  # noqa: ARG001
        sent.append((aid, dict(payload)))
        return Frame(
            type="ack",
            payload={
                "ok": True,
                "detail": {
                    "config_fingerprint": payload["config_fingerprint"],
                    "capacity": payload["capacity"],
                    "model_metadata": [{"id": "push-model", "object": "model"}],
                },
            },
        )

    monkeypatch.setattr(cu.manager, "send_command", fake_send_command)

    resp = client.patch(
        f"/admin/api/definitions/{definition_id}",
        json={"backend_config": {"delta_count": 9}},
    )
    assert resp.status_code == 200
    body = resp.json()
    new_fp = compute_config_fingerprint({"delta_count": 9})
    assert body["config_fingerprint"] == new_fp
    assert len(sent) == 1
    assert sent[0][0] == agent_id
    assert sent[0][1]["instance_id"] == instance_id
    assert sent[0][1]["backend_config"] == {"delta_count": 9}
    assert sent[0][1]["config_fingerprint"] == new_fp
    assert sent[0][1]["capacity"] == 1
    assert sent[0][1]["idle_timeout_seconds"] == 300
    results = body["config_update_results"]
    assert results[0]["ok"] is True
    assert results[0]["config_fingerprint"] == new_fp

    session.expunge_all()
    row = session.get(ProviderInstance, uuid.UUID(instance_id))
    assert row.config_fingerprint == new_fp
    # Discovered metadata persisted onto the definition.
    definition = session.get(ProviderDefinition, uuid.UUID(definition_id))
    assert definition.model_metadata == {
        "models": [{"id": "push-model", "object": "model"}]
    }


def test_patch_non_config_fields_do_not_push(
    client: TestClient, session: Session, monkeypatch
) -> None:
    from app.services import config_update as cu

    created = _create(client, alias="nc-model")
    _connected_backend(session, created["id"])

    calls: list = []

    async def no_send(*args, **kwargs):  # noqa: ARG001
        calls.append(args)
        raise AssertionError("must not be called")

    monkeypatch.setattr(cu.manager, "send_command", no_send)

    # idle_timeout + enabled only: admin-side fields, no provider push
    # (the idle reaper is admin-side; the provider consumes no idle).
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"idle_timeout_seconds": 60, "enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json()["config_update_results"] == []
    assert calls == []
    assert resp.json()["idle_timeout_seconds"] == 60
    assert resp.json()["enabled"] is False


def test_patch_capacity_only_triggers_push(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """SF-2: capacity is enforced at the provider, so a capacity-only
    PATCH (fingerprint unchanged) must push config.update."""
    from app.services import config_update as cu
    from app.services.wire import Frame

    created = _create(client, alias="cap-model", capacity=1)
    agent_id, instance_id = _connected_backend(session, created["id"])

    sent: list[tuple[str, dict]] = []

    async def fake_send_command(aid, type_, payload, timeout=30.0):  # noqa: ARG001
        sent.append((aid, dict(payload)))
        return Frame(
            type="ack",
            payload={
                "ok": True,
                "detail": {
                    "noop": True,
                    "capacity_adopted": True,
                    "config_fingerprint": payload["config_fingerprint"],
                    "capacity": payload["capacity"],
                },
            },
        )

    monkeypatch.setattr(cu.manager, "send_command", fake_send_command)

    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"capacity": 4},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["capacity"] == 4
    assert len(sent) == 1
    assert sent[0][0] == agent_id
    assert sent[0][1]["instance_id"] == instance_id
    # Same fingerprint, new capacity in the pushed payload.
    assert sent[0][1]["config_fingerprint"] == created["config_fingerprint"]
    assert sent[0][1]["capacity"] == 4
    results = body["config_update_results"]
    assert results[0]["ok"] is True


def test_patch_explicit_null_on_required_fields_is_422(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """SF-4 + Phase 16: explicit null on a NOT NULL column -> clear 422,
    never a 409 IntegrityError or a crash."""
    from app.services import config_update as cu

    created = _create(client, alias="null-model")

    async def no_send(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("must not be called")

    monkeypatch.setattr(cu.manager, "send_command", no_send)

    for field in ("alias", "capacity"):
        resp = client.patch(
            f"/admin/api/definitions/{created['id']}", json={field: None}
        )
        assert resp.status_code == 422, (field, resp.text)
        assert "cannot be null" in str(resp.json()["detail"]), field

    # Phase 16: provider_type/backend_config are NOT NULL (no shells) → the
    # same 422.
    for field in ("provider_type", "backend_config"):
        resp = client.patch(
            f"/admin/api/definitions/{created['id']}", json={field: None}
        )
        assert resp.status_code == 422, (field, resp.text)
        assert "cannot be" in str(resp.json()["detail"]), field

    # Row untouched.
    session.expunge_all()
    row = session.get(ProviderDefinition, uuid.UUID(created["id"]))
    assert row.alias == "null-model"
    assert row.capacity == 1
    assert row.provider_type == "mock"


def test_patch_provider_type_refused_with_instances_attached(
    client: TestClient, session: Session
) -> None:
    """N-3: changing provider_type while instances are attached breaks
    their binding -> 409."""
    created = _create(client, alias="pt-model", provider_type="mock")
    _connected_backend(session, created["id"], connected=False)

    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"provider_type": "llama-cpp"},
    )
    assert resp.status_code == 409
    assert "attached" in resp.json()["detail"]

    # Same-type PATCH is a no-op change and stays allowed.
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"provider_type": "mock"},
    )
    assert resp.status_code == 200


def test_patch_provider_type_allowed_without_instances(client: TestClient) -> None:
    created = _create(client, alias="pt-free", provider_type="mock")
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"provider_type": "llama-cpp"},
    )
    assert resp.status_code == 200
    assert resp.json()["provider_type"] == "llama-cpp"


def test_patch_same_backend_config_no_push(client: TestClient, monkeypatch) -> None:
    from app.services import config_update as cu

    created = _create(client, alias="same-model")

    async def no_send(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("must not be called")

    monkeypatch.setattr(cu.manager, "send_command", no_send)
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"backend_config": {"delta_count": 3}},
    )
    assert resp.status_code == 200
    assert resp.json()["config_update_results"] == []


def test_backend_config_serialization_validation(session: Session) -> None:
    # A set is accepted by Pydantic's dict[str, Any] but cannot survive
    # the canonical fingerprint serializer -> 422 from the admin's own
    # validation (unit-level: not expressible over a JSON transport).
    from fastapi import HTTPException

    from app.api.admin.definitions import _get_provider_type, _validate_backend_config

    ptype = _get_provider_type(session, "mock")
    _validate_backend_config(ptype, {"ok": [1, {"a": 2}]})  # must not raise
    with pytest.raises(HTTPException) as exc:
        _validate_backend_config(ptype, {"bad": {1, 2, 3}})
    assert exc.value.status_code == 422


def test_delete_refused_when_connected_instance(
    client: TestClient, session: Session
) -> None:
    created = _create(client, alias="dl-model")
    agent_id, instance_id = _connected_backend(session, created["id"], connected=True)

    resp = client.delete(f"/admin/api/definitions/{created['id']}")
    assert resp.status_code == 409
    assert "enabled=false" in resp.json()["detail"]

    # Disconnect the agent -> delete allowed.
    agent = session.get(ProviderAgent, uuid.UUID(agent_id))
    agent.websocket_connected = False
    session.add(agent)
    session.commit()
    resp = client.delete(f"/admin/api/definitions/{created['id']}")
    assert resp.status_code == 200
    session.expunge_all()
    assert session.get(ProviderInstance, uuid.UUID(instance_id)) is None


# --- Phase 16: definition placement (additive) -----------------------------


def _seed_agent(session: Session, provider_type: str = "mock") -> ProviderAgent:
    m = Machine(
        uid="pm-" + uuid.uuid4().hex[:8],
        name="pmachine-" + uuid.uuid4().hex[:8],
        dns="h",
    )
    session.add(m)
    session.commit()
    a = ProviderAgent(
        machine_id=m.id,
        provider_type=provider_type,
        agent_id="ag-" + uuid.uuid4().hex[:6],
    )
    session.add(a)
    session.commit()
    session.refresh(a)
    return a


def test_placement_defaults_any_of_type(client: TestClient) -> None:
    created = _create(client)
    assert created["agent_placement"] == "any_of_type"
    assert created["agents"] == []


def test_create_specific_placement_links_agents(
    client: TestClient, session: Session
) -> None:
    a = _seed_agent(session)
    created = _create(client, agent_placement="specific", agents=[str(a.id)])
    assert created["agent_placement"] == "specific"
    assert created["agents"] == [str(a.id)]
    links = session.exec(
        select(DefinitionAgent).where(
            DefinitionAgent.provider_definition_id == uuid.UUID(created["id"])
        )
    ).all()
    assert [str(link.agent_id) for link in links] == [str(a.id)]


def test_create_specific_requires_agents(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "spec-no-agents",
            "provider_type": "mock",
            "backend_config": {"delta_count": 1},
            "agent_placement": "specific",
        },
    )
    assert resp.status_code == 422


def test_create_specific_unknown_agent_422(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "spec-bad-agent",
            "provider_type": "mock",
            "backend_config": {"delta_count": 1},
            "agent_placement": "specific",
            "agents": ["11111111-1111-1111-1111-111111111111"],
        },
    )
    assert resp.status_code == 422
    assert "unknown agent" in resp.json()["detail"]


def test_create_invalid_placement_value_422(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "bad-placement",
            "provider_type": "mock",
            "backend_config": {"delta_count": 1},
            "agent_placement": "bogus",
        },
    )
    assert resp.status_code == 422


def test_create_malformed_agent_id_422(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "bad-agent-id",
            "provider_type": "mock",
            "backend_config": {"delta_count": 1},
            "agent_placement": "specific",
            "agents": ["not-a-uuid"],
        },
    )
    assert resp.status_code == 422
    assert "invalid agent id" in resp.json()["detail"]


def test_create_agents_without_specific_422(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "agents-no-specific",
            "provider_type": "mock",
            "backend_config": {"delta_count": 1},
            "agent_placement": "any_of_type",
            "agents": ["11111111-1111-1111-1111-111111111111"],
        },
    )
    assert resp.status_code == 422
    assert "only valid with" in resp.json()["detail"]


def test_patch_placement_to_specific_then_back(
    client: TestClient, session: Session
) -> None:
    created = _create(client)
    a = _seed_agent(session)
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"agent_placement": "specific", "agents": [str(a.id)]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agents"] == [str(a.id)]
    # Back to any_of_type clears the links.
    resp2 = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"agent_placement": "any_of_type"},
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["agents"] == []
    links = session.exec(
        select(DefinitionAgent).where(
            DefinitionAgent.provider_definition_id == uuid.UUID(created["id"])
        )
    ).all()
    assert links == []


# --- Phase 16 review: placement type-match (M1) + stale prune (M2) ----------


def test_create_specific_agent_type_mismatch_422(
    client: TestClient, session: Session
) -> None:
    """M1: a llama-cpp agent cannot host a mock definition (create)."""
    gufo_agent = _seed_agent(session, provider_type="gufo")
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "mismatch-create",
            "provider_type": "mock",
            "backend_config": {"delta_count": 1},
            "agent_placement": "specific",
            "agents": [str(gufo_agent.id)],
        },
    )
    assert resp.status_code == 422
    assert "does not match" in resp.json()["detail"]


def test_patch_specific_agent_type_mismatch_422(
    client: TestClient, session: Session
) -> None:
    """M1: the same guard applies on PATCH."""
    created = _create(client, alias="mismatch-patch")
    gufo_agent = _seed_agent(session, provider_type="gufo")
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"agent_placement": "specific", "agents": [str(gufo_agent.id)]},
    )
    assert resp.status_code == 422
    assert "does not match" in resp.json()["detail"]


def test_patch_placement_prunes_stale_backend(
    client: TestClient, session: Session
) -> None:
    """M2: moving a definition's specific placement off agent A deletes A's
    now-ghost backend row."""
    created = _create(client, alias="prune-me")
    def_id = uuid.UUID(created["id"])
    agent_a = _seed_agent(session)
    agent_b = _seed_agent(session)
    ghost = ProviderInstance(
        agent_id=agent_a.id,
        provider_definition_id=def_id,
        port=agent_a.base_port,
    )
    session.add(ghost)
    session.commit()
    ghost_id = ghost.id

    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"agent_placement": "specific", "agents": [str(agent_b.id)]},
    )
    assert resp.status_code == 200, resp.text
    session.expire_all()
    assert session.get(ProviderInstance, ghost_id) is None
    # agent B has no instance yet (it never registered) — nothing created.
    remaining = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.provider_definition_id == def_id
        )
    ).all()
    assert remaining == []
