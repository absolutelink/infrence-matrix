"""Phase 9 admin machines CRUD."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.models import Machine


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _create(client: TestClient, uid: str = "m1", name: str = "machine-one") -> dict:
    resp = client.post(
        "/admin/api/machines",
        json={
            "uid": uid,
            "name": name,
            "host": "10.0.0.5",
            "total_vram_bytes": 12345,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_create_get_list_patch_delete(client: TestClient, session: Session) -> None:
    created = _create(client)
    assert created["uid"] == "m1"
    assert created["total_vram_bytes"] == 12345
    assert created["instance_count"] == 0

    got = client.get(f"/admin/api/machines/{created['id']}")
    assert got.status_code == 200
    assert got.json()["host"] == "10.0.0.5"

    lst = client.get("/admin/api/machines").json()
    assert [m["id"] for m in lst] == [created["id"]]

    patched = client.patch(
        f"/admin/api/machines/{created['id']}",
        json={"dns": "box.local", "total_vram_bytes": 999},
    )
    assert patched.status_code == 200
    assert patched.json()["dns"] == "box.local"
    assert patched.json()["total_vram_bytes"] == 999
    session.expunge_all()
    row = session.exec(select(Machine)).one()
    assert row.dns == "box.local"

    deleted = client.delete(f"/admin/api/machines/{created['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True
    assert client.get(f"/admin/api/machines/{created['id']}").status_code == 404


def test_uid_uniqueness_conflict(client: TestClient) -> None:
    _create(client, uid="dup", name="first")
    resp = client.post("/admin/api/machines", json={"uid": "dup", "name": "second"})
    assert resp.status_code == 409


def test_name_uniqueness_conflict(client: TestClient) -> None:
    _create(client, uid="u1", name="dup-name")
    resp = client.patch(
        "/admin/api/machines/" + _create(client, uid="u2", name="other")["id"],
        json={"name": "dup-name"},
    )
    assert resp.status_code == 409


def test_delete_refused_with_instances_attached(
    client: TestClient, session: Session
) -> None:
    from app.models import ProviderDefinition, ProviderInstance

    created = _create(client, uid="busy", name="busy-machine")
    definition = ProviderDefinition(
        alias="busy-model",
        provider_type="mock",
        registration_token="busy-tok",
    )
    session.add(definition)
    session.commit()
    instance = ProviderInstance(
        machine_id=uuid.UUID(created["id"]),
        provider_definition_id=definition.id,
    )
    session.add(instance)
    session.commit()

    resp = client.delete(f"/admin/api/machines/{created['id']}")
    assert resp.status_code == 409
    assert "instance" in resp.json()["detail"]
    # Machine still there.
    assert client.get(f"/admin/api/machines/{created['id']}").status_code == 200


def test_get_unknown_404(client: TestClient) -> None:
    assert client.get("/admin/api/machines/nope-not-a-uuid").status_code == 404
    assert (
        client.get(
            "/admin/api/machines/11111111-1111-1111-1111-111111111111"
        ).status_code
        == 404
    )


def test_patch_rejects_negative_vram(client: TestClient) -> None:
    created = _create(client)
    resp = client.patch(
        f"/admin/api/machines/{created['id']}", json={"total_vram_bytes": -1}
    )
    assert resp.status_code == 422
