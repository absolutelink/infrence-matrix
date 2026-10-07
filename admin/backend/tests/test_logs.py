"""Phase 13 admin log paths: WS ingest (connection_manager → Redis) and
the GET /admin/api/instances/{id}/logs read endpoint (cursor, kind=all
merge, cap, legacy payload tolerance, 404)."""

import json
import os
import uuid
from types import SimpleNamespace

import pytest
import redis.asyncio as aioredis
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Machine, ProviderDefinition
from app.services import redis_keys
from app.services.connection_manager import ConnectionState, manager
from app.services.wire import Frame, FrameKind


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


class FakeWebSocket:
    def __init__(self, app_stub) -> None:
        self.app = app_stub
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        pass


@pytest.fixture
async def aredis() -> aioredis.Redis:
    r = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    yield r
    await r.aclose()


@pytest.fixture
def app_stub(aredis: aioredis.Redis) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(redis=aredis))


async def _ingest(
    app_stub, instance_id: str, frame_type: str, payload: dict, epoch: int = 1
) -> None:
    state = ConnectionState(
        instance_id=instance_id,
        websocket=FakeWebSocket(app_stub),
        epoch=epoch,
        connection_token="tok",
    )
    frame = Frame(type=frame_type, id=str(uuid.uuid4()), epoch=epoch, payload=payload)
    await manager.handle_inbound(app_stub, state, frame)


@pytest.mark.asyncio
async def test_handle_inbound_backend_logs_populates_redis(
    app_stub, aredis: aioredis.Redis
) -> None:
    instance_id = str(uuid.uuid4())
    payload = {
        "lines": [
            {"ts": "2026-10-06T00:00:01+00:00", "stream": "stdout", "text": "a"},
            {"ts": "2026-10-06T00:00:02+00:00", "stream": "stderr", "text": "b"},
        ],
        "dropped": 7,
    }
    await _ingest(app_stub, instance_id, FrameKind.BACKEND_LOGS, payload)

    key = redis_keys.logs_backend_key(instance_id)
    raw = await aredis.lrange(key, 0, -1)
    assert len(raw) == 2
    entries = [json.loads(x) for x in raw]
    # LPUSH newest-left: "b" first.
    assert entries[0]["text"] == "b" and entries[0]["stream"] == "stderr"
    assert entries[1]["text"] == "a"
    assert sorted(e["seq"] for e in entries) == [1, 2]
    ttl = await aredis.ttl(key)
    assert 0 < ttl <= redis_keys.LOGS_TTL_SECONDS
    assert await aredis.get(redis_keys.logs_dropped_key(instance_id, "backend")) == "7"

    # provider.logs goes to the provider list.
    await _ingest(
        app_stub,
        instance_id,
        FrameKind.PROVIDER_LOGS,
        {"lines": [{"ts": "t", "stream": "stdout", "text": "p"}], "dropped": 0},
    )
    praw = await aredis.lrange(redis_keys.logs_provider_key(instance_id), 0, -1)
    assert len(praw) == 1
    assert json.loads(praw[0])["text"] == "p"


@pytest.mark.asyncio
async def test_ingest_caps_list(app_stub, aredis: aioredis.Redis) -> None:
    instance_id = str(uuid.uuid4())
    cap = redis_keys.LOGS_CAP
    # Two batches totaling cap + 100 lines.
    for batch in range(2):
        lines = [
            {"ts": "t", "stream": "stdout", "text": f"l{batch * (cap // 2 + 50) + i}"}
            for i in range(cap // 2 + 50)
        ]
        await _ingest(
            app_stub,
            instance_id,
            FrameKind.BACKEND_LOGS,
            {"lines": lines, "dropped": 0},
        )
    raw = await aredis.lrange(redis_keys.logs_backend_key(instance_id), 0, -1)
    assert len(raw) == cap
    # Newest kept: the very last line must be present at the head.
    assert json.loads(raw[0])["text"] == f"l{cap + 99}"


@pytest.mark.asyncio
async def test_ingest_tolerates_legacy_single_line(
    app_stub, aredis: aioredis.Redis
) -> None:
    instance_id = str(uuid.uuid4())
    await _ingest(
        app_stub,
        instance_id,
        FrameKind.BACKEND_LOGS,
        {"stream": "stderr", "line": "legacy per-line"},
    )
    raw = await aredis.lrange(redis_keys.logs_backend_key(instance_id), 0, -1)
    assert len(raw) == 1
    entry = json.loads(raw[0])
    assert entry["text"] == "legacy per-line"
    assert entry["stream"] == "stderr"


@pytest.mark.asyncio
async def test_ingest_malformed_never_raises(app_stub, aredis: aioredis.Redis) -> None:
    instance_id = str(uuid.uuid4())
    for bad in (
        {},
        {"lines": "not-a-list"},
        {"lines": [None, 42, "string"]},
        {"lines": [{"nope": 1}]},
        {"lines": [{"stream": "stdout", "text": ""}]},
        {"dropped": "nan"},
    ):
        await _ingest(app_stub, instance_id, FrameKind.BACKEND_LOGS, bad)
    assert await aredis.llen(redis_keys.logs_backend_key(instance_id)) == 0

    # Direct normalize coverage for non-dict payloads too.
    from app.services.log_store import normalize_log_batch

    assert normalize_log_batch(None) == ([], 0)
    assert normalize_log_batch(["x"]) == ([], 0)
    assert normalize_log_batch({"stream": "stderr", "line": "legacy"}) == (
        [{"ts": "", "stream": "stderr", "text": "legacy"}],
        0,
    )


@pytest.mark.asyncio
async def test_stale_epoch_log_frame_dropped(app_stub, aredis: aioredis.Redis) -> None:
    instance_id = str(uuid.uuid4())
    state = ConnectionState(
        instance_id=instance_id,
        websocket=FakeWebSocket(app_stub),
        epoch=5,
        connection_token="tok",
    )
    frame = Frame(
        type=FrameKind.BACKEND_LOGS,
        id="x",
        epoch=4,
        payload={"lines": [{"ts": "t", "stream": "stdout", "text": "stale"}]},
    )
    await manager.handle_inbound(app_stub, state, frame)
    assert await aredis.llen(redis_keys.logs_backend_key(instance_id)) == 0


# ---------------------------------------------------------------------------
# HTTP read endpoint
# ---------------------------------------------------------------------------


def _seed(client: TestClient, session: Session) -> str:
    machine = Machine(uid="log-http-mach", name="m", host="10.0.0.1")
    definition = ProviderDefinition(
        alias="log-http-model",
        provider_type="mock",
        registration_token="tok",
        backend_config={},
    )
    session.add(machine)
    session.add(definition)
    session.commit()
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": "log-http-mach",
            "registration_token": "tok",
            "provider_type": "mock",
            "schema": {"type": "object"},
            "version": settings.VERSION,
            "port": 8081,
            "hardware": {"gpus": [], "total_vram_bytes": 1},
            "metrics_categories": [],
        },
    )
    assert resp.status_code == 200
    return resp.json()["instance_id"]


def test_logs_endpoint_cursor_and_merge(client: TestClient, session: Session) -> None:
    import asyncio

    instance_id = _seed(client, session)

    async def seed_logs() -> None:
        r = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
        try:
            app = SimpleNamespace(state=SimpleNamespace(redis=r))
            await manager.handle_inbound(
                app,
                ConnectionState(
                    instance_id=instance_id,
                    websocket=FakeWebSocket(app),
                    epoch=1,
                    connection_token="t",
                ),
                Frame(
                    type=FrameKind.BACKEND_LOGS,
                    id="f1",
                    epoch=1,
                    payload={
                        "lines": [
                            {
                                "ts": "2026-01-01T00:00:01+00:00",
                                "stream": "stdout",
                                "text": "b1",
                            },
                            {
                                "ts": "2026-01-01T00:00:02+00:00",
                                "stream": "stderr",
                                "text": "b2",
                            },
                        ],
                        "dropped": 3,
                    },
                ),
            )
            await manager.handle_inbound(
                app,
                ConnectionState(
                    instance_id=instance_id,
                    websocket=FakeWebSocket(app),
                    epoch=1,
                    connection_token="t",
                ),
                Frame(
                    type=FrameKind.PROVIDER_LOGS,
                    id="f2",
                    epoch=1,
                    payload={
                        "lines": [
                            {
                                "ts": "2026-01-01T00:00:03+00:00",
                                "stream": "stdout",
                                "text": "p1",
                            },
                        ],
                        "dropped": 1,
                    },
                ),
            )
        finally:
            await r.aclose()

    asyncio.run(seed_logs())

    resp = client.get(f"/admin/api/instances/{instance_id}/logs?kind=backend")
    assert resp.status_code == 200
    body = resp.json()
    assert [e["text"] for e in body["entries"]] == ["b2", "b1"]  # newest first
    assert body["cursor"] == 2
    assert body["dropped"] == 3
    # M1: since=0, oldest retained seq=1 -> nothing lost, no gap.
    assert body["gap"] is False
    assert body["oldest_seq"] == 1

    # since cursor: only newer than 1 → just seq 2
    resp = client.get(f"/admin/api/instances/{instance_id}/logs?kind=backend&since=1")
    body = resp.json()
    assert [e["text"] for e in body["entries"]] == ["b2"]
    assert body["cursor"] == 2

    # kind=all merges backend + provider by shared ingest seq.
    resp = client.get(f"/admin/api/instances/{instance_id}/logs?kind=all")
    body = resp.json()
    assert [e["text"] for e in body["entries"]] == ["p1", "b2", "b1"]
    assert body["dropped"] == 4
    cursor = body["cursor"]
    assert cursor == 3
    # Tail from the cursor: nothing new.
    resp = client.get(
        f"/admin/api/instances/{instance_id}/logs?kind=all&since={cursor}"
    )
    assert resp.json()["entries"] == []
    assert resp.json()["cursor"] == cursor

    # limit caps the page.
    resp = client.get(f"/admin/api/instances/{instance_id}/logs?kind=all&limit=1")
    assert len(resp.json()["entries"]) == 1
    assert resp.json()["entries"][0]["text"] == "p1"


def test_logs_endpoint_unseen_total_pre_trim(
    client: TestClient, session: Session
) -> None:
    """H1: unseen_total is the pre-trim count of entries with
    seq > since, so the UI can warn when >limit new lines arrived."""
    import asyncio

    instance_id = _seed(client, session)

    async def seed() -> None:
        r = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
        try:
            key = redis_keys.logs_backend_key(instance_id)
            pipe = r.pipeline()
            # Newest left: seqs 900..1.
            for seq in range(900, 0, -1):
                pipe.lpush(
                    key,
                    json.dumps(
                        {"seq": seq, "ts": "t", "stream": "stdout", "text": f"l{seq}"}
                    ),
                )
            await pipe.execute()
        finally:
            await r.aclose()

    asyncio.run(seed())

    resp = client.get(
        f"/admin/api/instances/{instance_id}/logs?kind=backend&since=0&limit=500"
    )
    body = resp.json()
    assert len(body["entries"]) == 500
    assert body["unseen_total"] == 900
    assert body["cursor"] == 900
    assert body["entries"][0]["seq"] == 900

    # Caught-up reads: nothing unseen, unseen_total == 0.
    resp = client.get(f"/admin/api/instances/{instance_id}/logs?kind=backend&since=900")
    body = resp.json()
    assert body["entries"] == []
    assert body["unseen_total"] == 0

    # unseen_total equals the page size when everything fits.
    resp = client.get(
        f"/admin/api/instances/{instance_id}/logs?kind=backend&since=800&limit=500"
    )
    body = resp.json()
    assert body["unseen_total"] == 100
    assert len(body["entries"]) == 100


def test_logs_endpoint_empty_not_error(client: TestClient, session: Session) -> None:
    instance_id = _seed(client, session)
    resp = client.get(f"/admin/api/instances/{instance_id}/logs?kind=all")
    assert resp.status_code == 200
    assert resp.json() == {
        "entries": [],
        "cursor": 0,
        "dropped": 0,
        "gap": False,
        "oldest_seq": 0,
        "unseen_total": 0,
    }


def test_logs_endpoint_404_unknown_instance(client: TestClient) -> None:
    resp = client.get(f"/admin/api/instances/{uuid.uuid4()}/logs")
    assert resp.status_code == 404
    resp = client.get("/admin/api/instances/not-a-uuid/logs")
    assert resp.status_code == 404


def test_logs_endpoint_bad_params(client: TestClient, session: Session) -> None:
    instance_id = _seed(client, session)
    assert (
        client.get(f"/admin/api/instances/{instance_id}/logs?kind=nope").status_code
        == 422
    )
    assert (
        client.get(f"/admin/api/instances/{instance_id}/logs?limit=99999").status_code
        == 422
    )
    assert (
        client.get(f"/admin/api/instances/{instance_id}/logs?since=-1").status_code
        == 422
    )


def test_logs_endpoint_reports_eviction_gap(
    client: TestClient, session: Session
) -> None:
    """M1: when `since` points before the oldest retained entry (rows
    were LTRIM'd away), the response flags gap=True with oldest_seq."""
    import asyncio

    instance_id = _seed(client, session)

    async def seed() -> None:
        r = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
        try:
            key = redis_keys.logs_backend_key(instance_id)
            # Simulate a list whose first 9 entries were trimmed away:
            # retained seqs 10..12 (newest left).
            for seq in (10, 11, 12):
                await r.lpush(
                    key,
                    json.dumps(
                        {"seq": seq, "ts": "t", "stream": "stdout", "text": f"l{seq}"}
                    ),
                )
        finally:
            await r.aclose()

    asyncio.run(seed())

    # since=0 but oldest retained is 10 -> gap (entries 1..9 lost).
    resp = client.get(f"/admin/api/instances/{instance_id}/logs?since=0&limit=1")
    body = resp.json()
    assert body["gap"] is True
    assert body["oldest_seq"] == 10

    # since=9: next expected is 10 == oldest retained -> no gap.
    resp = client.get(f"/admin/api/instances/{instance_id}/logs?since=9")
    assert resp.json()["gap"] is False

    # since=12: caught up, no gap.
    resp = client.get(f"/admin/api/instances/{instance_id}/logs?since=12")
    assert resp.json()["gap"] is False
    assert resp.json()["entries"] == []


def test_read_logs_rejects_unknown_kind() -> None:
    import asyncio

    from app.services.log_store import read_logs

    with pytest.raises(ValueError):
        asyncio.run(read_logs(None, "x", kind="bogus"))  # type: ignore[arg-type]
