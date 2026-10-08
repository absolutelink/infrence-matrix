"""Phase 19 S2 — admin stats endpoints for the UI stats bar.

Covers the four new reads/clears plus the ``alias`` addition on
``stats/usage``:

- ``GET    /admin/api/stats/scheduler``            per-definition queue/active + totals
- ``DELETE /admin/api/stats/scheduler/queue/{alias}``  clear one alias's queue
- ``DELETE /admin/api/stats/scheduler/queue``          clear every queue
- ``GET    /admin/api/stats/metrics``              fleet VRAM/GPU rollup
- ``GET    /admin/api/stats/usage``                samples now carry ``alias``

Scheduler state is injected directly into the app-lifespan singleton's
in-process ``_states`` (the same authoritative structure ``queue_snapshot`` /
``clear_queue`` read); the QueueCleared *raise* itself is covered in
``test_scheduler.py`` (S1). Metrics are seeded as per-agent GPU partials in
Redis, exactly what ``read_machine_metrics`` merges.
"""

import json
import uuid
from collections import deque
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.redis import get_redis
from app.models import TokenUsageSample
from app.services import redis_keys
from app.services.scheduler import _AliasState, _Slot
from tests.helpers import get_or_create_machine, make_definition

BASE = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _inject_state(
    scheduler,
    alias: str,
    *,
    queued: int,
    active: int,  # noqa: ANN001
) -> _AliasState:
    """Populate the authoritative in-process queue for ``alias``."""
    state = _AliasState()
    state.waiters = deque(f"w{i}-{alias}" for i in range(queued))
    state.active = {
        f"a{i}-{alias}": _Slot(
            instance_id=str(uuid.uuid4()), machine_uid="m", vram_bytes=0
        )
        for i in range(active)
    }
    scheduler._states[alias] = state
    return state


# ---------------------------------------------------------------------------
# GET /admin/api/stats/scheduler
# ---------------------------------------------------------------------------


def test_stats_scheduler_lists_all_definitions_with_zero_defaults(
    client: TestClient, session: Session
) -> None:
    make_definition(session, alias="sched-enabled")
    make_definition(session, alias="sched-disabled", enabled=False)
    _inject_state(client.app.state.scheduler, "sched-enabled", queued=2, active=1)

    resp = client.get("/admin/api/stats/scheduler")
    assert resp.status_code == 200
    body = resp.json()
    by_alias = {d["alias"]: d for d in body["definitions"]}
    # Every definition is listed, enabled or not.
    assert set(by_alias) == {"sched-enabled", "sched-disabled"}
    assert by_alias["sched-enabled"] == {
        "alias": "sched-enabled",
        "queued": 2,
        "active": 1,
    }
    # A definition with no scheduler state defaults to zeros.
    assert by_alias["sched-disabled"] == {
        "alias": "sched-disabled",
        "queued": 0,
        "active": 0,
    }
    assert body["totals"] == {"queued": 2, "active": 1}


def test_stats_scheduler_totals_and_ignores_state_without_definition(
    client: TestClient, session: Session
) -> None:
    make_definition(session, alias="t1")
    make_definition(session, alias="t2")
    scheduler = client.app.state.scheduler
    _inject_state(scheduler, "t1", queued=1, active=2)
    _inject_state(scheduler, "t2", queued=3, active=0)
    # Scheduler state for an alias with no definition is not surfaced.
    _inject_state(scheduler, "ghost", queued=9, active=9)

    body = client.get("/admin/api/stats/scheduler").json()
    assert {d["alias"] for d in body["definitions"]} == {"t1", "t2"}
    assert body["totals"] == {"queued": 4, "active": 2}


def test_stats_scheduler_empty(client: TestClient) -> None:
    body = client.get("/admin/api/stats/scheduler").json()
    assert body == {"definitions": [], "totals": {"queued": 0, "active": 0}}


# ---------------------------------------------------------------------------
# DELETE /admin/api/stats/scheduler/queue/{alias}
# ---------------------------------------------------------------------------


def test_clear_queue_alias_drains_waiters_only(
    client: TestClient, session: Session
) -> None:
    make_definition(session, alias="clr-a")
    scheduler = client.app.state.scheduler
    state = _inject_state(scheduler, "clr-a", queued=2, active=1)

    resp = client.delete("/admin/api/stats/scheduler/queue/clr-a")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "alias": "clr-a", "cleared": 2}

    # Waiters drained and marked cleared (a real blocked acquire would now
    # raise QueueCleared); the admitted slot is untouched.
    assert list(state.waiters) == []
    assert len(state.cleared) == 2
    assert len(state.active) == 1


def test_clear_queue_alias_empty_is_zero(client: TestClient, session: Session) -> None:
    make_definition(session, alias="clr-empty")
    resp = client.delete("/admin/api/stats/scheduler/queue/clr-empty")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "alias": "clr-empty", "cleared": 0}


def test_clear_queue_unknown_alias_404(client: TestClient) -> None:
    resp = client.delete("/admin/api/stats/scheduler/queue/ghost")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "definition not found"


# ---------------------------------------------------------------------------
# DELETE /admin/api/stats/scheduler/queue
# ---------------------------------------------------------------------------


def test_clear_all_queues(client: TestClient, session: Session) -> None:
    make_definition(session, alias="all-1")
    make_definition(session, alias="all-2")
    scheduler = client.app.state.scheduler
    s1 = _inject_state(scheduler, "all-1", queued=1, active=0)
    s2 = _inject_state(scheduler, "all-2", queued=2, active=0)

    resp = client.delete("/admin/api/stats/scheduler/queue")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "cleared": 3}
    assert list(s1.waiters) == [] and list(s2.waiters) == []


def test_clear_all_queues_empty(client: TestClient) -> None:
    resp = client.delete("/admin/api/stats/scheduler/queue")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "cleared": 0}


# ---------------------------------------------------------------------------
# GET /admin/api/stats/metrics
# ---------------------------------------------------------------------------


def _seed_gpu_partial(
    redis_sync,
    machine_uid: str,
    agent_id: str,
    gpu_uuid: str,
    total: int,
    used: int,
    util: float,
) -> None:  # noqa: ANN001
    partial = {
        "vram": {
            "gpus": [
                {
                    "uuid": gpu_uuid,
                    "id": 0,
                    "name": "card",
                    "vendor": "nvidia",
                    "vram_total": total,
                    "vram_used": used,
                    "vram_free": total - used,
                    "utilization": util,
                }
            ]
        }
    }
    redis_sync.set(
        redis_keys.metrics_agent_partial_key(machine_uid, agent_id),
        json.dumps(partial),
        ex=60,
    )


def test_stats_metrics_rollup(
    client: TestClient, session: Session, clean_redis
) -> None:  # noqa: ANN001
    get_or_create_machine(session, uid="mt-1", total_vram=0)
    get_or_create_machine(session, uid="mt-2", total_vram=0)
    # No metrics -> VRAM total falls back to the DB column; used stays 0.
    get_or_create_machine(session, uid="mt-3", total_vram=5000)

    _seed_gpu_partial(clean_redis, "mt-1", "ag1", "g1", 1000, 400, 50.0)
    _seed_gpu_partial(clean_redis, "mt-2", "ag2", "g2", 2000, 1000, 100.0)

    resp = client.get("/admin/api/stats/metrics")
    assert resp.status_code == 200
    body = resp.json()

    assert body["vram"] == {"used_bytes": 1400, "total_bytes": 8000}
    # Mean across the two machines that report gpu_usage (mt-3 excluded).
    assert body["gpu_utilization_percent"] == 75.0

    per = {m["uid"]: m for m in body["machines"]}
    assert per["mt-1"] == {
        "uid": "mt-1",
        "name": "name-mt-1",
        "vram_used_bytes": 400,
        "vram_total_bytes": 1000,
        "gpu_utilization_percent": 50.0,
    }
    assert per["mt-3"] == {
        "uid": "mt-3",
        "name": "name-mt-3",
        "vram_used_bytes": 0,
        "vram_total_bytes": 5000,
        "gpu_utilization_percent": 0.0,
    }


def test_stats_metrics_no_machines(client: TestClient) -> None:
    body = client.get("/admin/api/stats/metrics").json()
    assert body == {
        "vram": {"used_bytes": 0, "total_bytes": 0},
        "gpu_utilization_percent": 0.0,
        "machines": [],
    }


def test_stats_metrics_degrades_on_redis_failure(
    client: TestClient, session: Session
) -> None:
    from app.main import app

    get_or_create_machine(session, uid="mt-broke", total_vram=7000)

    class BrokenRedis:
        def scan_iter(self, match=None):  # noqa: ANN001, ARG002
            async def _boom():
                raise ConnectionError("redis down")
                yield  # pragma: no cover

            return _boom()

        async def get(self, key):  # noqa: ANN001, ARG002
            raise ConnectionError("redis down")

    app.dependency_overrides[get_redis] = lambda: BrokenRedis()
    try:
        resp = client.get("/admin/api/stats/metrics")
        assert resp.status_code == 200
        body = resp.json()
        # Per-machine read failure degrades to zeros + DB total fallback.
        assert body["vram"] == {"used_bytes": 0, "total_bytes": 7000}
        assert body["gpu_utilization_percent"] == 0.0
        assert body["machines"][0]["vram_total_bytes"] == 7000
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /admin/api/stats/usage — samples carry alias
# ---------------------------------------------------------------------------


def test_usage_samples_carry_alias_and_null(
    client: TestClient, session: Session
) -> None:
    definition = make_definition(session, alias="ua-1")
    session.add_all(
        [
            TokenUsageSample(
                provider_definition_id=definition.id,
                prompt_tokens=5,
                created_at=BASE.replace(tzinfo=None),
            ),
            TokenUsageSample(
                provider_definition_id=None,
                prompt_tokens=7,
                created_at=(BASE + timedelta(minutes=1)).replace(tzinfo=None),
            ),
        ]
    )
    session.commit()

    body = client.get("/admin/api/stats/usage").json()
    by_tokens = {s["prompt_tokens"]: s["alias"] for s in body["samples"]}
    assert by_tokens[5] == "ua-1"
    assert by_tokens[7] is None


def test_usage_alias_null_when_definition_deleted(
    client: TestClient, session: Session
) -> None:
    definition = make_definition(session, alias="ua-del")
    session.add(
        TokenUsageSample(
            provider_definition_id=definition.id,
            prompt_tokens=9,
            created_at=BASE.replace(tzinfo=None),
        )
    )
    session.commit()
    session.delete(definition)
    session.commit()

    body = client.get("/admin/api/stats/usage").json()
    assert len(body["samples"]) == 1
    assert body["samples"][0]["alias"] is None
    assert body["samples"][0]["provider_definition_id"] is None
