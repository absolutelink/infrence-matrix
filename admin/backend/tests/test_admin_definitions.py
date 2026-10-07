"""Phase 9 admin provider-definitions CRUD + config-change push.

Phase 12: ``provider_type`` must reference a registered ``ProviderType``
row and ``backend_config`` is validated against that type's committed
JSON Schema. The autouse fixture below seeds the known types with a
permissive schema so the Phase 9 push/CRUD semantics stay the focus of
these tests; the schema-validation specifics are covered in
``test_provider_types.py``.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.admin.providers import compute_config_fingerprint
from app.models import ProviderDefinition, ProviderType
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


def test_create_returns_fingerprint_and_token(client: TestClient) -> None:
    created = _create(client)
    assert created["alias"] == "def-model"
    assert created["config_fingerprint"] == compute_config_fingerprint(
        {"delta_count": 3}
    )
    # Auto-generated registration token (copyable into provider env).
    assert len(created["registration_token"]) >= 16


def test_create_explicit_registration_token(client: TestClient) -> None:
    created = _create(client, registration_token="my-token")
    assert created["registration_token"] == "my-token"


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
        "/admin/api/definitions", json={"alias": "taken", "provider_type": "mock"}
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
    """backend_config change pushes provider.config.update to connected
    instances and the DB fingerprint updates on the ok ack."""
    from app.models import Machine, ProviderInstance
    from app.services import config_update as cu
    from app.services.wire import Frame

    machine = Machine(uid="cu-mach", name="cu-machine")
    session.add(machine)
    created = _create(client, alias="push-model")
    definition_id = created["id"]
    instance = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=uuid.UUID(definition_id),
        websocket_connected=True,
        config_fingerprint=created["config_fingerprint"],
    )
    session.add(instance)
    session.commit()
    instance_id = str(instance.id)

    sent: list[tuple[str, dict]] = []

    async def fake_send_command(inst_id, type_, payload, timeout=30.0):  # noqa: ARG001
        sent.append((inst_id, dict(payload)))
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
    assert sent[0][0] == instance_id
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
    from app.models import Machine, ProviderInstance
    from app.services import config_update as cu

    machine = Machine(uid="nc-mach", name="nc-machine")
    session.add(machine)
    created = _create(client, alias="nc-model")
    instance = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=uuid.UUID(created["id"]),
        websocket_connected=True,
        config_fingerprint=created["config_fingerprint"],
    )
    session.add(instance)
    session.commit()

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
    from app.models import Machine, ProviderInstance
    from app.services import config_update as cu
    from app.services.wire import Frame

    machine = Machine(uid="cap-mach", name="cap-machine")
    session.add(machine)
    created = _create(client, alias="cap-model", capacity=1)
    instance = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=uuid.UUID(created["id"]),
        websocket_connected=True,
        config_fingerprint=created["config_fingerprint"],
    )
    session.add(instance)
    session.commit()
    instance_id = str(instance.id)

    sent: list[tuple[str, dict]] = []

    async def fake_send_command(inst_id, type_, payload, timeout=30.0):  # noqa: ARG001
        sent.append((inst_id, dict(payload)))
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
    assert sent[0][0] == instance_id
    # Same fingerprint, new capacity in the pushed payload.
    assert sent[0][1]["config_fingerprint"] == created["config_fingerprint"]
    assert sent[0][1]["capacity"] == 4
    results = body["config_update_results"]
    assert results[0]["ok"] is True


def test_patch_explicit_null_on_required_fields_is_422(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """SF-4 + Phase 14: explicit null on a NOT NULL column or on the
    shell-nullable fields (once set) -> clear 422, never a 409
    IntegrityError or a crash."""
    from app.services import config_update as cu

    created = _create(client, alias="null-model")

    async def no_send(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("must not be called")

    monkeypatch.setattr(cu.manager, "send_command", no_send)

    for field in ("alias", "registration_token", "capacity"):
        resp = client.patch(
            f"/admin/api/definitions/{created['id']}", json={field: None}
        )
        assert resp.status_code == 422, (field, resp.text)
        assert "cannot be null" in str(resp.json()["detail"]), field

    # Phase 14: provider_type/backend_config are nullable columns, but a
    # definition never returns to the shell state → the same 422.
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
    from app.models import Machine, ProviderInstance

    machine = Machine(uid="pt-mach", name="pt-machine")
    session.add(machine)
    created = _create(client, alias="pt-model", provider_type="mock")
    instance = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=uuid.UUID(created["id"]),
        websocket_connected=False,
    )
    session.add(instance)
    session.commit()

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
    from app.models import Machine, ProviderInstance

    machine = Machine(uid="dl-mach", name="dl-machine")
    session.add(machine)
    created = _create(client, alias="dl-model")
    instance = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=uuid.UUID(created["id"]),
        websocket_connected=True,
    )
    session.add(instance)
    session.commit()

    resp = client.delete(f"/admin/api/definitions/{created['id']}")
    assert resp.status_code == 409
    assert "enabled=false" in resp.json()["detail"]

    # Disconnect the instance -> delete allowed.
    instance.websocket_connected = False
    session.add(instance)
    session.commit()
    resp = client.delete(f"/admin/api/definitions/{created['id']}")
    assert resp.status_code == 200


# ============================================================================
# Phase 14 — shell definitions
# ============================================================================


def test_create_shell_definition(client: TestClient) -> None:
    """A definition with no provider_type/backend_config is a shell: null
    type/config/fingerprint, schedulable machinery untouched."""
    resp = client.post(
        "/admin/api/definitions", json={"alias": "shell-model", "capacity": 2}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider_type"] is None
    assert body["backend_config"] is None
    assert body["config_fingerprint"] is None
    assert len(body["registration_token"]) >= 16


def test_create_shell_with_backend_config_is_422(client: TestClient) -> None:
    """No type → no schema to validate against → refuse config (never
    store unvalidatable config)."""
    resp = client.post(
        "/admin/api/definitions",
        json={"alias": "sh-cfg", "backend_config": {"delta_count": 3}},
    )
    assert resp.status_code == 422
    assert "backend_config" in str(resp.json()["detail"])


def test_create_shell_with_provider_type_wants_type_registered(
    client: TestClient,
) -> None:
    resp = client.post(
        "/admin/api/definitions", json={"alias": "sh-bad", "provider_type": "ghost"}
    )
    assert resp.status_code == 422
    assert "unknown provider_type" in str(resp.json()["detail"])


def test_patch_config_onto_shell_validates_and_pushes(
    client: TestClient, monkeypatch
) -> None:
    """shell → configured: the PATCH validates against the (adopted or
    set) type's committed schema and fires the standard push; config may
    not be written while the definition is still untyped."""
    from app.services import config_update as cu

    created = client.post("/admin/api/definitions", json={"alias": "shell-push"}).json()

    async def no_send(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("must not push without a type")

    monkeypatch.setattr(cu.manager, "send_command", no_send)
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"backend_config": {"delta_count": 3}},
    )
    assert resp.status_code == 422

    # Set the type first (operator path; no instances attached), then
    # config validates + pushes (no connected instances → empty results).
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"provider_type": "mock"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["provider_type"] == "mock"

    async def ok_send(instance_id, kind, payload, timeout=None):  # noqa: ARG001
        from app.services.wire import Frame

        return Frame(
            type="ack",
            payload={
                "ok": True,
                "detail": {
                    "config_fingerprint": compute_config_fingerprint({"delta_count": 3})
                },
            },
        )

    monkeypatch.setattr(cu.manager, "send_command", ok_send)
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"backend_config": {"delta_count": 3}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backend_config"] == {"delta_count": 3}
    assert body["config_fingerprint"] == compute_config_fingerprint({"delta_count": 3})

    # No going back: explicit nulls are refused once set.
    for field in ("provider_type", "backend_config"):
        resp = client.patch(
            f"/admin/api/definitions/{created['id']}", json={field: None}
        )
        assert resp.status_code == 422


def test_shell_type_only_patch_never_pushes(
    client: TestClient, monkeypatch, session: Session
) -> None:
    """A capacity-only PATCH on a still-shell definition (type set, config
    still null) must NOT fabricate a push with the hash of {}."""
    from app.services import config_update as cu

    created = client.post("/admin/api/definitions", json={"alias": "shell-cap"}).json()
    session.expunge_all()
    row = session.get(ProviderDefinition, uuid.UUID(created["id"]))
    row.provider_type = "mock"
    session.add(row)
    session.commit()

    async def no_send(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("shell must not be pushed")

    monkeypatch.setattr(cu.manager, "send_command", no_send)
    resp = client.patch(f"/admin/api/definitions/{created['id']}", json={"capacity": 3})
    assert resp.status_code == 200
    assert resp.json()["config_fingerprint"] is None


def test_patch_shell_type_and_config_same_request(
    client: TestClient,
) -> None:
    """Phase 14 spec item B: a single PATCH may both type the shell and
    author the config (validated against that newly-set type's schema,
    then pushed)."""
    created = client.post(
        "/admin/api/definitions", json={"alias": "shell-one-shot"}
    ).json()

    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"provider_type": "mock", "backend_config": {"delta_count": 3}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider_type"] == "mock"
    assert body["backend_config"] == {"delta_count": 3}
    assert body["config_fingerprint"] == compute_config_fingerprint({"delta_count": 3})
