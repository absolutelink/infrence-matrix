"""Phase 12 — schema-driven backend config: ProviderType registry.

Covers the registration consensus gate (docs/ws-protocol.md §2 table),
definition validation against the committed schema, the
/admin/api/provider-types endpoints (list/get/commit/dismiss), and the
report-only prestart schema check.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.models import (
    Machine,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
)
from app.services.hashing import canonical_json_sha256

# Committed schema for the test type: requires an integer `delta_count`.
SCHEMA_V1 = {
    "type": "object",
    "properties": {"delta_count": {"type": "integer"}},
    "required": ["delta_count"],
}
# A new schema the fleet must agree on before it commits.
SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "delta_count": {"type": "integer"},
        "speculative": {"type": "boolean"},
    },
    "required": ["delta_count"],
}
# A third, conflicting schema.
SCHEMA_V3 = {"type": "object", "properties": {"other": {"type": "string"}}}

FP_V1 = canonical_json_sha256(SCHEMA_V1)
FP_V2 = canonical_json_sha256(SCHEMA_V2)
FP_V3 = canonical_json_sha256(SCHEMA_V3)

TYPE_NAME = "mock"


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _machine(session: Session, uid: str) -> Machine:
    m = Machine(uid=uid, name=f"name-{uid}", host="10.0.0.5")
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def _definition(
    session: Session, *, token: str, alias: str | None = None
) -> ProviderDefinition:
    d = ProviderDefinition(
        alias=alias or f"alias-{uuid.uuid4().hex[:8]}",
        provider_type=TYPE_NAME,
        registration_token=token,
        backend_config={"delta_count": 3},
    )
    session.add(d)
    session.commit()
    session.refresh(d)
    return d


def _register(
    client: TestClient,
    machine_uid: str,
    token: str,
    schema: dict | None,
    **overrides,
):
    body = {
        "machine_uid": machine_uid,
        "registration_token": token,
        "provider_type": TYPE_NAME,
        "schema": schema,
        "version": settings.VERSION,
        "port": 8081,
        "hardware": {"gpus": [], "total_vram_bytes": 1},
        "metrics_categories": [],
    }
    body.update(overrides)
    if schema is None:
        body.pop("schema")  # omitted entirely (old provider package)
    return client.post("/admin/api/providers/register", json=body)


def _get_type(session: Session, name: str = TYPE_NAME) -> ProviderType:
    # Expire first: the admin request runs in its own session and this
    # test session may hold a stale identity-map copy from an earlier read.
    session.expire_all()
    return session.exec(select(ProviderType).where(ProviderType.name == name)).one()


# ---------------------------------------------------------------------------
# Registration: bootstrap + consensus matrix
# ---------------------------------------------------------------------------


def test_unknown_type_bootstraps_and_commits(
    client: TestClient, session: Session
) -> None:
    _machine(session, "boot-m")
    _definition(session, token="boot-tok")

    resp = _register(client, "boot-m", "boot-tok", SCHEMA_V1)
    assert resp.status_code == 200, resp.text
    instance_id = resp.json()["instance_id"]

    ptype = _get_type(session)
    assert ptype.name == TYPE_NAME
    assert ptype.schema == SCHEMA_V1
    assert ptype.schema_fingerprint == FP_V1
    assert ptype.status == "active"
    assert ptype.pending_fingerprint is None

    inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    assert inst.reported_schema_fingerprint == FP_V1


def test_matching_committed_fingerprint_registers(
    client: TestClient, session: Session
) -> None:
    _machine(session, "match-m")
    _definition(session, token="match-tok")
    assert _register(client, "match-m", "match-tok", SCHEMA_V1).status_code == 200

    # A second instance of the same type on the committed schema.
    _machine(session, "match-m2")
    _definition(session, token="match-tok2")
    resp = _register(client, "match-m2", "match-tok2", SCHEMA_V1)
    assert resp.status_code == 200
    assert _get_type(session).status == "active"


def test_mismatch_stages_pending_and_refuses(
    client: TestClient, session: Session
) -> None:
    _machine(session, "stage-a")
    _definition(session, token="stage-ta")
    a_id = _register(client, "stage-a", "stage-ta", SCHEMA_V1).json()["instance_id"]

    _machine(session, "stage-b")
    _definition(session, token="stage-tb")
    resp = _register(client, "stage-b", "stage-tb", SCHEMA_V2)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "schema_pending"
    assert detail["provider_type"] == TYPE_NAME
    assert detail["committed_fingerprint"] == FP_V1
    assert detail["pending_fingerprint"] == FP_V2
    b_id = _register_type_instance_id(session, "stage-b")
    assert detail["voted"] == [b_id]
    assert detail["waiting_on"] == [a_id]

    ptype = _get_type(session)
    assert ptype.status == "consensus_pending"
    assert ptype.pending_schema == SCHEMA_V2
    assert ptype.pending_fingerprint == FP_V2
    assert ptype.pending_voters == [b_id]
    # Committed schema untouched.
    assert ptype.schema == SCHEMA_V1

    # The refusal must NOT mint a secret for the new instance.
    # (Redis check via the sync client fixture.)
    import os

    import redis as redis_sync

    r = redis_sync.Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    try:
        assert r.get(f"im:ws:secret:{b_id}") is None
    finally:
        r.close()

    # ... but the reported fingerprint IS persisted on the 409 path.
    inst = session.get(ProviderInstance, uuid.UUID(b_id))
    session.refresh(inst)
    assert inst.reported_schema_fingerprint == FP_V2


def _register_type_instance_id(session: Session, machine_uid: str) -> str:
    inst = session.exec(
        select(ProviderInstance)
        .join(Machine, ProviderInstance.machine_id == Machine.id)
        .where(Machine.uid == machine_uid)
    ).one()
    return str(inst.id)


def test_pending_vote_incomplete_keeps_refusing(
    client: TestClient, session: Session
) -> None:
    _machine(session, "inc-a")
    _definition(session, token="inc-ta")
    a_id = _register(client, "inc-a", "inc-ta", SCHEMA_V1).json()["instance_id"]

    _machine(session, "inc-b")
    _definition(session, token="inc-tb")
    _register(client, "inc-b", "inc-tb", SCHEMA_V2)  # stage, 409
    b_id = _register_type_instance_id(session, "inc-b")

    _machine(session, "inc-c")
    _definition(session, token="inc-tc")
    resp = _register(
        client, "inc-c", "inc-tc", SCHEMA_V2
    )  # second vote, still incomplete
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    c_id = _register_type_instance_id(session, "inc-c")
    assert set(detail["voted"]) == {b_id, c_id}
    assert detail["waiting_on"] == [a_id]

    # Repeat vote from B is deduped (voter list unchanged in size).
    resp = _register(client, "inc-b", "inc-tb", SCHEMA_V2)
    assert resp.status_code == 409
    assert len(resp.json()["detail"]["voted"]) == 2
    assert _get_type(session).pending_voters.count(b_id) == 1


def test_last_vote_commits_pending(client: TestClient, session: Session) -> None:
    _machine(session, "fin-a")
    _definition(session, token="fin-ta")
    _register(client, "fin-a", "fin-ta", SCHEMA_V1)

    _machine(session, "fin-b")
    _definition(session, token="fin-tb")
    assert _register(client, "fin-b", "fin-tb", SCHEMA_V2).status_code == 409

    # A (the last outstanding voter) now presents the pending schema.
    resp = _register(client, "fin-a", "fin-ta", SCHEMA_V2)
    assert resp.status_code == 200, resp.text
    assert resp.json()["instance_secret"]  # registration completed

    ptype = _get_type(session)
    assert ptype.schema == SCHEMA_V2
    assert ptype.schema_fingerprint == FP_V2
    assert ptype.pending_schema is None
    assert ptype.pending_fingerprint is None
    assert ptype.pending_voters == []
    assert ptype.status == "active"


def test_third_fingerprint_conflicts(client: TestClient, session: Session) -> None:
    _machine(session, "cf-a")
    _definition(session, token="cf-ta")
    _register(client, "cf-a", "cf-ta", SCHEMA_V1)

    _machine(session, "cf-b")
    _definition(session, token="cf-tb")
    assert _register(client, "cf-b", "cf-tb", SCHEMA_V2).status_code == 409

    resp = _register(client, "cf-a", "cf-ta", SCHEMA_V3)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "schema_conflict"
    assert detail["committed_fingerprint"] == FP_V1
    assert detail["pending_fingerprint"] == FP_V2

    ptype = _get_type(session)
    assert ptype.status == "conflict"
    # Pending state preserved for the operator (not overwritten by V3).
    assert ptype.pending_fingerprint == FP_V2
    assert ptype.schema == SCHEMA_V1
    # Reported fingerprint written on the conflict path too.
    a = session.exec(
        select(ProviderInstance)
        .join(Machine, ProviderInstance.machine_id == Machine.id)
        .where(Machine.uid == "cf-a")
    ).one()
    assert a.reported_schema_fingerprint == FP_V3
    b = session.exec(
        select(ProviderInstance)
        .join(Machine, ProviderInstance.machine_id == Machine.id)
        .where(Machine.uid == "cf-b")
    ).one()
    assert b.reported_schema_fingerprint == FP_V2


def test_malformed_schema_rejected_422(client: TestClient, session: Session) -> None:
    _machine(session, "bad-m")
    _definition(session, token="bad-tok")
    resp = _register(client, "bad-m", "bad-tok", {"type": "not-a-real-schema-type"})
    assert resp.status_code == 422
    assert "invalid JSON Schema" in resp.json()["detail"]
    # No type row created for a malformed schema.
    assert session.exec(select(ProviderType)).first() is None


# ---------------------------------------------------------------------------
# provider-types endpoints
# ---------------------------------------------------------------------------


def test_list_provider_types(client: TestClient, session: Session) -> None:
    _machine(session, "lt-m")
    _definition(session, token="lt-tok")
    _register(client, "lt-m", "lt-tok", SCHEMA_V1)

    resp = client.get("/admin/api/provider-types")
    assert resp.status_code == 200
    items = resp.json()
    assert [t["name"] for t in items] == [TYPE_NAME]
    assert items[0]["status"] == "active"
    assert items[0]["schema_fingerprint"] == FP_V1
    assert items[0]["consensus"]["pending_fingerprint"] is None
    assert items[0]["consensus"]["universe_count"] == 1
    assert items[0]["consensus"]["on_committed"] == 1


def test_get_provider_type_full_schema(client: TestClient, session: Session) -> None:
    _machine(session, "gt-m")
    _definition(session, token="gt-tok")
    _register(client, "gt-m", "gt-tok", SCHEMA_V1)

    resp = client.get(f"/admin/api/provider-types/{TYPE_NAME}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema"] == SCHEMA_V1
    assert body["schema_fingerprint"] == FP_V1
    assert body["pending_schema"] is None
    assert body["status"] == "active"

    assert client.get("/admin/api/provider-types/nope").status_code == 404


def test_force_commit_pending(client: TestClient, session: Session) -> None:
    _machine(session, "fc-a")
    _definition(session, token="fc-ta")
    _register(client, "fc-a", "fc-ta", SCHEMA_V1)
    _machine(session, "fc-b")
    _definition(session, token="fc-tb")
    assert _register(client, "fc-b", "fc-tb", SCHEMA_V2).status_code == 409

    resp = client.post(f"/admin/api/provider-types/{TYPE_NAME}/pending/commit")
    assert resp.status_code == 200, resp.text
    assert resp.json()["schema_fingerprint"] == FP_V2

    ptype = _get_type(session)
    assert ptype.schema == SCHEMA_V2
    assert ptype.status == "active"
    assert ptype.pending_fingerprint is None

    # Future registrations on V2 now succeed.
    _machine(session, "fc-c")
    _definition(session, token="fc-tc")
    assert _register(client, "fc-c", "fc-tc", SCHEMA_V2).status_code == 200


def test_dismiss_pending(client: TestClient, session: Session) -> None:
    _machine(session, "dp-a")
    _definition(session, token="dp-ta")
    _register(client, "dp-a", "dp-ta", SCHEMA_V1)
    _machine(session, "dp-b")
    _definition(session, token="dp-tb")
    assert _register(client, "dp-b", "dp-tb", SCHEMA_V2).status_code == 409

    resp = client.post(f"/admin/api/provider-types/{TYPE_NAME}/pending/dismiss")
    assert resp.status_code == 200
    assert resp.json()["dismissed_fingerprint"] == FP_V2

    ptype = _get_type(session)
    assert ptype.schema == SCHEMA_V1  # committed unchanged
    assert ptype.pending_fingerprint is None
    assert ptype.status == "active"


def test_commit_and_dismiss_without_pending_409(
    client: TestClient, session: Session
) -> None:
    _machine(session, "np-m")
    _definition(session, token="np-tok")
    _register(client, "np-m", "np-tok", SCHEMA_V1)

    for action in ("commit", "dismiss"):
        resp = client.post(f"/admin/api/provider-types/{TYPE_NAME}/pending/{action}")
        assert resp.status_code == 409
        assert "nothing pending" in resp.json()["detail"]

    assert (
        client.post("/admin/api/provider-types/ghost/pending/commit").status_code == 404
    )
    assert (
        client.post("/admin/api/provider-types/ghost/pending/dismiss").status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Definition validation against the committed schema
# ---------------------------------------------------------------------------


def test_definition_valid_config_passes(client: TestClient, session: Session) -> None:
    _machine(session, "dv-m")
    _definition(session, token="dv-tok")
    _register(client, "dv-m", "dv-tok", SCHEMA_V1)  # commits SCHEMA_V1

    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "valid-cfg",
            "provider_type": TYPE_NAME,
            "backend_config": {"delta_count": 7},
        },
    )
    assert resp.status_code == 200, resp.text


def test_definition_invalid_config_rejected_with_field_errors(
    client: TestClient, session: Session
) -> None:
    _machine(session, "di-m")
    _definition(session, token="di-tok")
    _register(client, "di-m", "di-tok", SCHEMA_V1)

    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "invalid-cfg",
            "provider_type": TYPE_NAME,
            "backend_config": {"delta_count": "not-an-int"},
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "backend_config_schema_validation_failed"
    assert detail["provider_type"] == TYPE_NAME
    assert detail["schema_fingerprint"] == FP_V1
    paths = [e["path"] for e in detail["errors"]]
    assert "$.delta_count" in paths

    # Missing required key also surfaces.
    resp = client.post(
        "/admin/api/definitions",
        json={"alias": "missing-key", "provider_type": TYPE_NAME, "backend_config": {}},
    )
    assert resp.status_code == 422
    assert any("$" == e["path"] for e in resp.json()["detail"]["errors"])


def test_definition_unknown_provider_type_422(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={"alias": "x", "provider_type": "not-a-type"},
    )
    assert resp.status_code == 422
    assert "unknown provider_type" in resp.json()["detail"]
    assert "registered types" in resp.json()["detail"]


def test_definition_patch_validates_against_committed_schema(
    client: TestClient, session: Session
) -> None:
    _machine(session, "pv-m")
    _definition(session, token="pv-tok")
    _register(client, "pv-m", "pv-tok", SCHEMA_V1)
    created = client.post(
        "/admin/api/definitions",
        json={
            "alias": "patch-me",
            "provider_type": TYPE_NAME,
            "backend_config": {"delta_count": 1},
        },
    ).json()

    bad = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"backend_config": {"delta_count": [1, 2]}},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["error"] == "backend_config_schema_validation_failed"

    good = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"backend_config": {"delta_count": 42, "speculative": True}},
    )
    # SCHEMA_V1 allows additional properties; only delta_count is typed.
    assert good.status_code == 200, good.text

    # After V2 is committed, a config violating V2 is rejected.
    ptype = _get_type(session)
    ptype.pending_schema = SCHEMA_V2
    ptype.pending_fingerprint = FP_V2
    session.add(ptype)
    session.commit()
    client.post(f"/admin/api/provider-types/{TYPE_NAME}/pending/commit")
    rejected = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={"backend_config": {"delta_count": "nope"}},
    )
    assert rejected.status_code == 422


# ---------------------------------------------------------------------------
# B5: report-only prestart schema check
# ---------------------------------------------------------------------------


def test_schema_report_flags_bad_existing_config(session: Session, caplog) -> None:
    """A definition whose backend_config violates the committed schema is
    logged as a WARNING but never mutated and never raises (report-only)."""
    from scripts.schema_report import main as schema_report_main

    # Strict committed schema + a violating definition row (inserted
    # directly, bypassing the API's create-time validation).
    strict = {
        "type": "object",
        "properties": {"delta_count": {"type": "integer"}},
        "required": ["delta_count"],
    }
    session.add(
        ProviderType(
            name="report-type",
            schema=strict,
            schema_fingerprint=canonical_json_sha256(strict),
            status="active",
        )
    )
    bad = ProviderDefinition(
        alias="report-bad",
        provider_type="report-type",
        registration_token="report-bad-tok",
        backend_config={"delta_count": "wrong"},
    )
    session.add(bad)
    session.commit()

    with caplog.at_level("WARNING"):
        rc = schema_report_main()
    assert rc == 0  # never fatal
    assert "report-bad" in caplog.text
    assert "does not validate" in caplog.text

    # Row untouched.
    session.expunge_all()
    row = session.exec(
        select(ProviderDefinition).where(ProviderDefinition.alias == "report-bad")
    ).one()
    assert row.backend_config == {"delta_count": "wrong"}


def test_schema_report_skips_when_no_types(caplog) -> None:
    from scripts.schema_report import main as schema_report_main

    with caplog.at_level("INFO"):
        rc = schema_report_main()
    assert rc == 0
    assert "no provider types registered" in caplog.text


def test_schema_report_honors_sqlalchemy_url(monkeypatch, caplog) -> None:
    """M2: SQLALCHEMY_URL (exported by prestart.sh) selects the engine;
    a broken URL there is caught and never fatal."""
    from scripts.schema_report import main as schema_report_main

    monkeypatch.setenv("SQLALCHEMY_URL", "postgresql+psycopg://nobody@127.0.0.1:1/nope")
    with caplog.at_level("WARNING"):
        rc = schema_report_main()
    assert rc == 0
    assert "skipped due to error" in caplog.text


# ---------------------------------------------------------------------------
# Reviewer additions
# ---------------------------------------------------------------------------


def _redis_get(key: str) -> str | None:
    import os

    import redis as redis_sync

    r = redis_sync.Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    try:
        return r.get(key)
    finally:
        r.close()


def test_conflict_path_mints_no_secret(client: TestClient, session: Session) -> None:
    """Reviewer: no-secret-mint on the schema_conflict path."""
    _machine(session, "cs-a")
    _definition(session, token="cs-ta")
    a_id = _register(client, "cs-a", "cs-ta", SCHEMA_V1).json()["instance_id"]
    _machine(session, "cs-b")
    _definition(session, token="cs-tb")
    assert _register(client, "cs-b", "cs-tb", SCHEMA_V2).status_code == 409

    # Drop A's secret from its successful first registration so the
    # conflict attempt's mint (or lack thereof) is observable.
    import os

    import redis as redis_sync

    r = redis_sync.Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    r.delete(f"im:ws:secret:{a_id}")
    r.close()

    resp = _register(client, "cs-a", "cs-ta", SCHEMA_V3)
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "schema_conflict"
    assert _redis_get(f"im:ws:secret:{a_id}") is None
    b_id = _register_type_instance_id(session, "cs-b")
    assert _redis_get(f"im:ws:secret:{b_id}") is None


def test_hardware_merged_on_schema_pending_409(
    client: TestClient, session: Session
) -> None:
    """M1: a schema-refused registration still refreshes the machine's
    hardware report (the box is real regardless of schema state)."""
    _machine(session, "hw-a")
    _definition(session, token="hw-ta")
    _register(
        client,
        "hw-a",
        "hw-ta",
        SCHEMA_V1,
        hardware={"gpus": [{"uuid": "g1"}], "total_vram_bytes": 111},
    )

    machine_b = _machine(session, "hw-b")
    _definition(session, token="hw-tb")
    new_hw = {"gpus": [{"uuid": "g2"}], "total_vram_bytes": 222}
    resp = _register(client, "hw-b", "hw-tb", SCHEMA_V2, hardware=new_hw)
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "schema_pending"

    session.expire_all()
    machine_b = session.get(Machine, machine_b.id)
    assert machine_b.hardware == new_hw
    assert machine_b.total_vram_bytes == 222


def test_second_vote_409_carries_committed_fingerprint(
    client: TestClient, session: Session
) -> None:
    """Reviewer: committed_fingerprint present in the second-vote detail."""
    _machine(session, "sf-a")
    _definition(session, token="sf-ta")
    _register(client, "sf-a", "sf-ta", SCHEMA_V1)
    _machine(session, "sf-b")
    _definition(session, token="sf-tb")
    assert _register(client, "sf-b", "sf-tb", SCHEMA_V2).status_code == 409

    _machine(session, "sf-c")
    _definition(session, token="sf-tc")
    resp = _register(client, "sf-c", "sf-tc", SCHEMA_V2)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "schema_pending"
    assert detail["committed_fingerprint"] == FP_V1
    assert detail["pending_fingerprint"] == FP_V2
    assert len(detail["voted"]) == 2
    assert detail["waiting_on"] == [_register_type_instance_id(session, "sf-a")]


def test_solo_first_vote_of_new_schema_commits_immediately(
    client: TestClient, session: Session
) -> None:
    """L1: when the voter universe is just this instance, a new-schema
    registration is unanimous -> commit, no 409."""
    _machine(session, "solo-m")
    _definition(session, token="solo-tok")
    assert _register(client, "solo-m", "solo-tok", SCHEMA_V1).status_code == 200

    resp = _register(client, "solo-m", "solo-tok", SCHEMA_V2)
    assert resp.status_code == 200, resp.text
    ptype = _get_type(session)
    assert ptype.schema == SCHEMA_V2
    assert ptype.schema_fingerprint == FP_V2
    assert ptype.status == "active"
    assert ptype.pending_fingerprint is None


def test_committed_match_clears_stale_conflict(
    client: TestClient, session: Session
) -> None:
    """L2: a registration matching the committed fingerprint resets a
    stale conflict (only when nothing is pending — a live pending vote is
    never cleared by it)."""
    _machine(session, "cl-a")
    _definition(session, token="cl-ta")
    _register(client, "cl-a", "cl-ta", SCHEMA_V1)
    _machine(session, "cl-b")
    _definition(session, token="cl-tb")
    assert _register(client, "cl-b", "cl-tb", SCHEMA_V2).status_code == 409
    assert _register(client, "cl-a", "cl-ta", SCHEMA_V3).status_code == 409
    assert _get_type(session).status == "conflict"

    # Committed-match with pending still staged: allowed, but the pending
    # vote (and the conflict flag) survive.
    assert _register(client, "cl-a", "cl-ta", SCHEMA_V1).status_code == 200
    ptype = _get_type(session)
    assert ptype.pending_fingerprint == FP_V2
    assert ptype.status == "conflict"

    # Defensive L2 case: conflict with nothing pending resets to active
    # on a committed-match registration.
    ptype = _get_type(session)
    ptype.pending_schema = None
    ptype.pending_fingerprint = None
    ptype.pending_voters = []
    ptype.status = "conflict"
    session.add(ptype)
    session.commit()
    assert _register(client, "cl-a", "cl-ta", SCHEMA_V1).status_code == 200
    assert _get_type(session).status == "active"


# ---------------------------------------------------------------------------
# H2: TRANSITION — schema omitted (old provider packages, pre Phase D)
# ---------------------------------------------------------------------------


def test_omitted_schema_unknown_type_bootstraps_permissive(
    client: TestClient, session: Session
) -> None:
    _machine(session, "om-a")
    _definition(session, token="om-ta")
    resp = _register(client, "om-a", "om-ta", None)
    assert resp.status_code == 200, resp.text
    instance_id = resp.json()["instance_id"]

    ptype = _get_type(session)
    assert ptype.schema == {"type": "object"}
    assert ptype.schema_fingerprint == canonical_json_sha256({"type": "object"})
    assert ptype.status == "active"

    inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    assert inst.reported_schema_fingerprint == canonical_json_sha256({"type": "object"})


def test_omitted_schema_known_type_skips_gate(
    client: TestClient, session: Session
) -> None:
    _machine(session, "om2-a")
    _definition(session, token="om2-ta")
    assert _register(client, "om2-a", "om2-ta", SCHEMA_V1).status_code == 200

    # Old provider (no schema) against a type with a strict committed
    # schema: gate skipped, registration proceeds; reported fp is None
    # (the instance never proved which schema it runs).
    _machine(session, "om2-b")
    _definition(session, token="om2-tb")
    resp = _register(client, "om2-b", "om2-tb", None)
    assert resp.status_code == 200, resp.text
    b_id = _register_type_instance_id(session, "om2-b")
    inst = session.get(ProviderInstance, uuid.UUID(b_id))
    session.refresh(inst)
    assert inst.reported_schema_fingerprint is None

    # Consensus state untouched.
    ptype = _get_type(session)
    assert ptype.schema == SCHEMA_V1
    assert ptype.pending_fingerprint is None
    assert ptype.status == "active"


def test_omitted_schema_does_not_disturb_pending_vote(
    client: TestClient, session: Session
) -> None:
    _machine(session, "om3-a")
    _definition(session, token="om3-ta")
    _register(client, "om3-a", "om3-ta", SCHEMA_V1)
    _machine(session, "om3-b")
    _definition(session, token="om3-tb")
    assert _register(client, "om3-b", "om3-tb", SCHEMA_V2).status_code == 409

    # An old-schema-less provider registers fine mid-consensus...
    _machine(session, "om3-c")
    _definition(session, token="om3-tc")
    assert _register(client, "om3-c", "om3-tc", None).status_code == 200

    # ... but does NOT vote: the pending roster still holds only B.
    ptype = _get_type(session)
    assert ptype.status == "consensus_pending"
    assert ptype.pending_fingerprint == FP_V2
    assert ptype.pending_voters == [_register_type_instance_id(session, "om3-b")]

    # A completing its V2 vote now also needs C? No — C is in the
    # universe, so consensus is NOT complete until C presents V2 too.
    resp = _register(client, "om3-a", "om3-ta", SCHEMA_V2)
    assert resp.status_code == 409  # still waiting on C
    assert _register(client, "om3-c", "om3-tc", SCHEMA_V2).status_code == 200
    assert _get_type(session).schema == SCHEMA_V2


# ---------------------------------------------------------------------------
# H1: mixed str/int absolute_path must not 500 the error sorter
# ---------------------------------------------------------------------------

MIXED_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "args": {
            "type": "object",
            "properties": {"ctx": {"type": "integer"}},
            "required": ["ctx"],
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["args", "tags"],
}


def test_mixed_path_schema_errors_do_not_500(
    client: TestClient, session: Session
) -> None:
    """H1: an object-key error and an array-index error together used to
    crash sorted() with TypeError (str vs int comparison)."""
    strict = {
        "type": "object",
        "properties": {
            "args": {"type": "object", "properties": {"ctx": {"type": "integer"}}},
            "list": {"type": "array", "items": {"type": "string"}},
        },
    }
    session.add(
        ProviderType(
            name="mixed-type",
            schema=strict,
            schema_fingerprint=canonical_json_sha256(strict),
            status="active",
        )
    )
    session.commit()

    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "mixed-bad",
            "provider_type": "mixed-type",
            "backend_config": {
                "args": {"ctx": "not-int"},  # str path part: args.ctx
                "list": [123],  # int path part: list[0]
            },
        },
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "backend_config_schema_validation_failed"
    paths = {e["path"] for e in detail["errors"]}
    assert "$.args.ctx" in paths
    assert any(p.startswith("$.list") for p in paths)


def test_mixed_path_sorting_unit(session: Session) -> None:
    """Direct unit coverage of the sorted() key with mixed path types."""
    from app.api.admin.definitions import _get_provider_type, _validate_backend_config

    strict = {
        "type": "object",
        "properties": {
            "b": {"type": "array", "items": {"type": "string"}},
            "a": {"type": "integer"},
        },
    }
    session.add(
        ProviderType(
            name="mixed-unit",
            schema=strict,
            schema_fingerprint=canonical_json_sha256(strict),
            status="active",
        )
    )
    session.commit()
    ptype = _get_provider_type(session, "mixed-unit")
    with pytest.raises(Exception) as excinfo:
        _validate_backend_config(ptype, {"a": "x", "b": [1]})
    # Must be the clean 422 HTTPException, never a TypeError.
    from fastapi import HTTPException

    assert isinstance(excinfo.value, HTTPException)
    assert excinfo.value.status_code == 422
    paths = [e["path"] for e in excinfo.value.detail["errors"]]
    assert sorted(paths) == paths  # deterministic ordering


# ---------------------------------------------------------------------------
# M3: retype without config change validates against the NEW schema
# ---------------------------------------------------------------------------


def test_patch_retype_validates_existing_config(
    client: TestClient, session: Session
) -> None:
    strict = {
        "type": "object",
        "properties": {"only_key": {"type": "string"}},
        "required": ["only_key"],
    }
    # Bootstrap the permissive "mock" committed schema via a registration.
    _machine(session, "rt-m")
    _definition(session, token="rt-tok")
    assert _register(client, "rt-m", "rt-tok", SCHEMA_V1).status_code == 200

    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "retype-src",
            "provider_type": TYPE_NAME,
            "backend_config": {"delta_count": 3},
        },
    )
    assert resp.status_code == 200, resp.text
    def_id = resp.json()["id"]

    # Register the strict type out-of-band (as a provider bootstrap would).
    from app.core.db import engine

    with Session(engine) as s:
        s.add(
            ProviderType(
                name="strict-type",
                schema=strict,
                schema_fingerprint=canonical_json_sha256(strict),
                status="active",
            )
        )
        s.commit()

    # Retype without touching backend_config: {"delta_count": 3} violates
    # the new schema's required only_key -> 422 (was silently allowed).
    resp = client.patch(
        f"/admin/api/definitions/{def_id}", json={"provider_type": "strict-type"}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "backend_config_schema_validation_failed"

    # Retype + a valid new config in the same PATCH -> allowed.
    resp = client.patch(
        f"/admin/api/definitions/{def_id}",
        json={
            "provider_type": "strict-type",
            "backend_config": {"only_key": "ok"},
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["backend_config"] == {"only_key": "ok"}


# ---------------------------------------------------------------------------
# L1 (Phase 12 E review): the frontend hardcodes the permissive-schema
# fingerprint literal in admin/frontend/src/routes/_layout/instances.tsx
# (PERMISSIVE_SCHEMA_FP) to avoid badging bootstrap-schema instances as
# waiting_schema. Pin the backend value to that exact literal so a change
# here fails loudly and the frontend constant gets updated in lockstep.
# ---------------------------------------------------------------------------

FRONTEND_PERMISSIVE_FP = (
    "a2c799262a3ce3c19ef5cdd983bf3d12b43ab3c426227091b909dcb7054738c0"
)


def test_permissive_fingerprint_pinned_to_frontend_constant() -> None:
    from app.api.admin.providers import PERMISSIVE_FINGERPRINT, PERMISSIVE_SCHEMA

    assert PERMISSIVE_SCHEMA == {"type": "object"}
    assert PERMISSIVE_FINGERPRINT == canonical_json_sha256(PERMISSIVE_SCHEMA)
    assert PERMISSIVE_FINGERPRINT == FRONTEND_PERMISSIVE_FP
