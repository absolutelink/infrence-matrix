"""Phase 10 admin read endpoints.

Covers the UI-facing reads added with the admin overhaul:

- ``GET /admin/api/instances`` / ``GET /admin/api/instances/{id}``
- ``GET /admin/api/responses``
- ``GET /admin/api/stats/usage``
- ``GET /admin/api/stats/overview``

Includes the timezone-serialization fix (FIX 1): every emitted
timestamp must carry an explicit UTC offset so non-UTC browsers parse
it correctly.
"""

import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.redis import get_redis
from app.models import (
    Machine,
    ProviderDefinition,
    ProviderInstance,
    ResponseRecord,
    TokenUsageSample,
)
from app.services import redis_keys
from app.services.wire import Frame


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


BASE = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


def _assert_utc_iso(value: str) -> datetime:
    """Timestamps serialize with an explicit UTC offset (FIX 1)."""
    assert value.endswith("+00:00"), value
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    return parsed


def _stack(session: Session, *, alias: str = "read-model", enabled: bool = True):
    machine = Machine(uid=f"rd-{alias}", name=f"rd-{alias}-machine", host="10.0.0.1")
    definition = ProviderDefinition(
        alias=alias,
        provider_type="mock",
        registration_token=f"tok-{alias}",
        backend_config={"delta_count": 1},
        enabled=enabled,
    )
    session.add_all([machine, definition])
    session.commit()
    session.refresh(machine)
    session.refresh(definition)
    return machine, definition


def _instance(
    session: Session,
    machine: Machine,
    definition: ProviderDefinition,
    *,
    connected: bool = False,
    backend_status: str = "stopped",
    created_at: datetime | None = None,
) -> ProviderInstance:
    # (machine_id, provider_definition_id) is UNIQUE: one backend per
    # instance per machine+definition. Stack a fresh machine per call so
    # tests can create several instances of one definition.
    tag = uuid.uuid4().hex[:6]
    extra = Machine(
        uid=f"{machine.uid}-x{tag}",
        name=f"{machine.name}-x{tag}",
        host=machine.host,
    )
    session.add(extra)
    session.commit()
    session.refresh(extra)
    machine = extra
    inst = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=definition.id,
        port=8081,
        version="dev",
        instance_status="running" if connected else "disconnected",
        backend_status=backend_status,
        websocket_connected=connected,
        epoch=1 if connected else 0,
        last_seen=BASE,
        last_request_at=BASE + timedelta(seconds=5),
        config_fingerprint="fp",
    )
    if created_at is not None:
        inst.created_at = created_at.replace(tzinfo=None)
    session.add(inst)
    session.commit()
    session.refresh(inst)
    return inst


# ---------------------------------------------------------------------------
# GET /admin/api/instances
# ---------------------------------------------------------------------------


def test_list_instances_empty(client: TestClient) -> None:
    resp = client.get("/admin/api/instances")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_instances_shape_and_ordering(
    client: TestClient, session: Session
) -> None:
    machine, definition = _stack(session)
    older = _instance(
        session, machine, definition, created_at=BASE - timedelta(hours=2)
    )
    newer = _instance(
        session,
        machine,
        definition,
        connected=True,
        backend_status="running",
        created_at=BASE - timedelta(hours=1),
    )

    resp = client.get("/admin/api/instances")
    assert resp.status_code == 200
    items = resp.json()
    assert [i["id"] for i in items] == [str(older.id), str(newer.id)]  # created asc

    item = items[1]
    assert item["machine_uid"].startswith(f"{machine.uid}-x")
    assert item["machine_name"].startswith(f"{machine.name}-x")
    assert item["alias"] == definition.alias
    assert item["provider_type"] == "mock"
    assert item["instance_status"] == "running"
    assert item["backend_status"] == "running"
    assert item["websocket_connected"] is True
    assert item["epoch"] == 1
    assert item["port"] == 8081
    assert item["config_fingerprint"] == "fp"
    _assert_utc_iso(item["created_at"])
    _assert_utc_iso(item["last_seen"])
    _assert_utc_iso(item["last_request_at"])


def test_get_instance_happy_and_404s(client: TestClient, session: Session) -> None:
    machine, definition = _stack(session)
    inst = _instance(session, machine, definition, connected=True)

    resp = client.get(f"/admin/api/instances/{inst.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(inst.id)
    assert body["websocket_connected"] is True
    _assert_utc_iso(body["created_at"])

    missing = client.get(f"/admin/api/instances/{uuid.uuid4()}")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "instance not found"

    # Malformed id must hit the ValueError path -> 404, never a 500.
    bad = client.get("/admin/api/instances/not-a-uuid")
    assert bad.status_code == 404
    assert bad.json()["detail"] == "instance not found"


# ---------------------------------------------------------------------------
# GET /admin/api/responses
# ---------------------------------------------------------------------------


def _record(
    session: Session,
    definition: ProviderDefinition,
    *,
    response_id: str,
    created_at: datetime,
    status: str = "completed",
    parameters: dict | None = None,
    completed: bool = True,
) -> ResponseRecord:
    rec = ResponseRecord(
        response_id=response_id,
        input_items=[{"role": "user", "content": "hi"}],
        output_items=[{"type": "message", "content": "hello"}],
        provider_definition_id=definition.id,
        parameters=parameters
        if parameters is not None
        else {"api_format": "responses"},
        status=status,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        created_at=created_at.replace(tzinfo=None),
    )
    if completed:
        rec.completed_at = (created_at + timedelta(seconds=1)).replace(tzinfo=None)
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def test_list_responses_empty(client: TestClient) -> None:
    resp = client.get("/admin/api/responses")
    assert resp.status_code == 200
    assert resp.json() == {"total": 0, "limit": 50, "offset": 0, "responses": []}


def test_list_responses_ordering_and_paging(
    client: TestClient, session: Session
) -> None:
    _, definition = _stack(session)
    for i in range(3):
        _record(
            session,
            definition,
            response_id=f"resp-{i}",
            created_at=BASE + timedelta(minutes=i),
        )

    page1 = client.get("/admin/api/responses?limit=2&offset=0").json()
    assert page1["total"] == 3
    assert [r["response_id"] for r in page1["responses"]] == [
        "resp-2",
        "resp-1",
    ]  # newest first
    page2 = client.get("/admin/api/responses?limit=2&offset=2").json()
    assert page2["total"] == 3
    assert [r["response_id"] for r in page2["responses"]] == ["resp-0"]


def test_list_response_fields(client: TestClient, session: Session) -> None:
    _, definition = _stack(session)
    _record(
        session,
        definition,
        response_id="resp-full",
        created_at=BASE,
        parameters={"api_format": "chat_completions"},
    )
    _record(
        session,
        definition,
        response_id="resp-inprog",
        created_at=BASE + timedelta(minutes=1),
        status="in_progress",
        completed=False,
    )
    _record(
        session,
        definition,
        response_id="resp-noformat",
        created_at=BASE + timedelta(minutes=2),
        parameters={},  # no api_format key -> defaults
    )

    resp = client.get("/admin/api/responses").json()
    by_id = {r["response_id"]: r for r in resp["responses"]}

    full = by_id["resp-full"]
    assert full["api_format"] == "chat_completions"
    assert full["model_alias"] == definition.alias
    assert full["provider_type"] == "mock"
    assert full["status"] == "completed"
    assert full["input_tokens"] == 10
    assert full["output_tokens"] == 5
    assert full["total_tokens"] == 15
    assert full["store"] is True
    _assert_utc_iso(full["created_at"])
    _assert_utc_iso(full["completed_at"])

    # In-progress rows have no completed_at.
    assert by_id["resp-inprog"]["completed_at"] is None

    # Missing api_format in parameters defaults to "responses".
    assert by_id["resp-noformat"]["api_format"] == "responses"

    # Heavy payload columns must not leak into the list view.
    for r in resp["responses"]:
        assert "input_items" not in r
        assert "output_items" not in r


@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=201", "offset=-1", "limit=1&offset=-1"],
)
def test_list_responses_validation(client: TestClient, query: str) -> None:
    resp = client.get(f"/admin/api/responses?{query}")
    assert resp.status_code == 422, query


# ---------------------------------------------------------------------------
# GET /admin/api/stats/usage
# ---------------------------------------------------------------------------


def test_usage_stats_empty(client: TestClient) -> None:
    resp = client.get("/admin/api/stats/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["totals"] == {
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
    }
    assert body["samples"] == []


def test_usage_stats_totals_and_limit(client: TestClient, session: Session) -> None:
    machine, definition = _stack(session)
    inst = _instance(session, machine, definition)
    for i in range(4):
        session.add(
            TokenUsageSample(
                provider_instance_id=inst.id,
                provider_definition_id=definition.id,
                prompt_tokens=100 + i,
                cached_tokens=10 * i,
                completion_tokens=50 + i,
                prompt_ms=1000.0,
                predicted_ms=500.0,
                prompt_per_second=100.0,
                predicted_per_second=200.0,
                created_at=(BASE + timedelta(minutes=i)).replace(tzinfo=None),
            )
        )
    session.commit()

    body = client.get("/admin/api/stats/usage?limit=100").json()
    # All 4 rows summed (totals are all-time, independent of limit).
    assert body["totals"] == {
        "prompt_tokens": 100 + 101 + 102 + 103,
        "cached_tokens": 0 + 10 + 20 + 30,
        "completion_tokens": 50 + 51 + 52 + 53,
    }
    assert len(body["samples"]) == 4
    # Newest first.
    assert body["samples"][0]["prompt_tokens"] == 103
    sample = body["samples"][0]
    assert sample["provider_instance_id"] == str(inst.id)
    assert sample["prompt_per_second"] == 100.0
    assert sample["predicted_per_second"] == 200.0
    _assert_utc_iso(sample["created_at"])

    limited = client.get("/admin/api/stats/usage?limit=2").json()
    assert len(limited["samples"]) == 2
    assert limited["totals"]["prompt_tokens"] == 406  # still all-time


# ---------------------------------------------------------------------------
# GET /admin/api/stats/overview
# ---------------------------------------------------------------------------


def test_overview_counts(client: TestClient, session: Session) -> None:
    machine, definition = _stack(session, alias="ov-enabled")
    _stack(session, alias="ov-disabled", enabled=False)
    _instance(session, machine, definition, backend_status="stopped")
    _instance(session, machine, definition, backend_status="running", connected=True)
    _instance(session, machine, definition, backend_status="in_use")
    _instance(session, machine, definition, backend_status="stopping")
    _instance(session, machine, definition, backend_status="error")

    body = client.get("/admin/api/stats/overview").json()
    # 2 stack machines + one fresh machine per instance (unique machine+def).
    assert body["machines"] == 7
    assert body["definitions"] == 2
    assert body["definitions_enabled"] == 1
    assert body["instances"] == 5
    assert body["instances_connected"] == 1
    # ONLY running + in_use count as running backends.
    assert body["backends_running"] == 2
    assert body["queued_requests"] == 0
    assert body["active_requests"] == 0
    assert body["input_tokens"] == 0


def test_overview_reads_scheduler_redis_mirror(
    client: TestClient, session: Session, clean_redis
) -> None:
    machine, definition = _stack(session, alias="ov-mirror")
    _instance(session, machine, definition)
    clean_redis.rpush(redis_keys.sched_queue_key("ov-mirror"), "r1", "r2", "r3")
    clean_redis.sadd(redis_keys.sched_active_key("ov-mirror"), "a1", "a2")
    # A disabled alias's mirror entries must NOT be counted.
    clean_redis.rpush(redis_keys.sched_queue_key("ov-disabled"), "x")

    body = client.get("/admin/api/stats/overview").json()
    assert body["queued_requests"] == 3
    assert body["active_requests"] == 2


def test_overview_redis_failure_degrades_to_zero(
    client: TestClient, session: Session
) -> None:
    from app.main import app

    machine, definition = _stack(session, alias="ov-broke")
    _instance(session, machine, definition)

    class BrokenRedis:
        async def llen(self, key):  # noqa: ANN001, ARG002
            raise ConnectionError("redis down")

        async def scard(self, key):  # noqa: ANN001, ARG002
            raise ConnectionError("redis down")

    app.dependency_overrides[get_redis] = lambda: BrokenRedis()
    try:
        resp = client.get("/admin/api/stats/overview")
        assert resp.status_code == 200
        body = resp.json()
        assert body["queued_requests"] == 0
        assert body["active_requests"] == 0
        # DB-derived counts are unaffected.
        assert body["instances"] == 1
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Regression: a live websocket is reflected in reads + counts
# ---------------------------------------------------------------------------


def test_live_websocket_reflected_in_reads(
    client: TestClient, session: Session
) -> None:
    machine, definition = _stack(session, alias="ov-live")
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": machine.uid,
            "registration_token": "tok-ov-live",
            "provider_type": "mock",
            "version": settings.VERSION,
            "port": 8081,
            "hardware": {"gpus": [], "total_vram_bytes": 1},
            "metrics_categories": [],
        },
    )
    assert resp.status_code == 200
    reg = resp.json()
    instance_id = reg["instance_id"]

    assert (
        client.get(f"/admin/api/instances/{instance_id}").json()["websocket_connected"]
        is False
    )

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers={"Authorization": f"Bearer {reg['instance_secret']}"},
    ) as ws:
        hello = Frame.model_validate_json(ws.receive_text())
        assert hello.type == "provider.hello"

        item = client.get(f"/admin/api/instances/{instance_id}").json()
        assert item["websocket_connected"] is True
        assert item["epoch"] == hello.epoch
        _assert_utc_iso(item["last_seen"])

        overview = client.get("/admin/api/stats/overview").json()
        assert overview["instances_connected"] == 1

    # After the socket closes the disconnect path flips the flag back
    # (async in the app's event loop — poll like the ws tests do).
    deadline = time.monotonic() + 3.0
    while True:
        after = client.get(f"/admin/api/instances/{instance_id}").json()
        if not after["websocket_connected"]:
            break
        assert time.monotonic() < deadline, "disconnect not reflected"
        time.sleep(0.02)
    assert client.get("/admin/api/stats/overview").json()["instances_connected"] == 0
