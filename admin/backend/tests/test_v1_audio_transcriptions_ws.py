"""Phase 24 S4: live-ASR WebSocket relay (``WS /v1/audio/transcriptions/stream``).

Drives the endpoint through the starlette ``TestClient`` websocket with a real
scheduler-admitted ``asr`` instance (seeded like the HTTP transcription tests)
and a stubbed upstream agent WS (``websockets.connect`` monkeypatched to a
``FakeUpstream``). Asserts the relay contract (ARCHITECTURE.md §7 Audio
specifics + the module docstring close-code table):

- bidirectional text + binary passthrough (no parsing);
- the scheduler slot is held for the connection lifetime and released exactly
  once on every teardown path (happy end, client disconnect, upstream close);
- pre-flight rejections (missing/unknown/disabled/non-asr alias) close 1008 and
  never acquire or connect;
- scheduler failures (NoProviderAvailable / QueueTimeout) close 1013 and never
  connect upstream;
- an upstream busy close (agent 1013) is relayed to the client + released;
- an oversized inbound frame closes 1009;
- the max-duration and idle timers close the socket + release the slot;
- the slot release is SHIELDED: cancelling the endpoint task mid-relay still
  runs ``scheduler.release`` exactly once and lets it complete;
- an upstream→client BINARY frame is forwarded intact (locks ``send_bytes``).
"""

import asyncio
import contextlib
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from starlette.websockets import WebSocketDisconnect

from app.api.v1 import audio_transcriptions_ws as route_mod
from app.core.config import settings
from app.services.scheduler import NoProviderAvailable, QueueTimeout
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

PATH = "/v1/audio/transcriptions/stream"


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def seed_asr(
    session: Session,
    *,
    alias: str,
    machine_uid: str,
    modality: str = "asr",
    enabled: bool = True,
    connected: bool = True,
) -> None:
    machine = get_or_create_machine(session, uid=machine_uid, host="127.0.0.1")
    agent = make_agent(session, machine, connected=connected)
    definition = make_definition(
        session,
        alias=alias,
        provider_type="mock",
        modality=modality,
        enabled=enabled,
        backend_config={"model": {"file": "m.gguf"}},
    )
    make_instance(session, agent, definition, backend_status="running")


_END = object()


class FakeUpstream:
    """Stand-in for a ``websockets`` client connection.

    ``frames`` are emitted upstream→client (text or bytes). ``echo`` reflects
    every client→upstream frame back so a test can deterministically observe that
    the relay forwarded it. ``close=(code, reason)`` makes the upstream close
    with that code after draining ``frames`` (the agent's 1013/1008/1011). With
    neither, the connection is held open until the relay tears it down.
    """

    def __init__(
        self,
        frames: list = (),
        *,
        echo: bool = False,
        close: tuple[int, str] | None = None,
    ) -> None:
        self._q: asyncio.Queue = asyncio.Queue()
        for frame in frames:
            self._q.put_nowait(frame)
        if close is not None:
            self._q.put_nowait((_END, close[0], close[1]))
        self._echo = echo
        self.sent: list = []
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None

    async def send(self, data) -> None:
        self.sent.append(data)
        if self._echo:
            self._q.put_nowait(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        while True:
            item = await self._q.get()
            if isinstance(item, tuple) and len(item) == 3 and item[0] is _END:
                self.close_code, self.close_reason = item[1], item[2]
                return
            yield item


def install_upstream(monkeypatch, upstream: FakeUpstream) -> list[str]:
    """Patch ``websockets.connect`` to hand back ``upstream``; return the list of
    URLs it was asked to dial (empty == no connect attempted)."""
    urls: list[str] = []

    async def fake_connect(url, *args, **kwargs):  # noqa: ARG001
        urls.append(url)
        return upstream

    monkeypatch.setattr(route_mod.websockets, "connect", fake_connect)
    return urls


def spy_scheduler(scheduler, monkeypatch):
    """Wrap acquire/release to record (alias, request_id) calls."""
    acquires: list[tuple[str, str]] = []
    releases: list[tuple[str, str]] = []
    real_acquire, real_release = scheduler.acquire, scheduler.release

    async def spy_acquire(alias: str, request_id: str):
        acquires.append((alias, request_id))
        return await real_acquire(alias, request_id)

    async def spy_release(alias: str, request_id: str):
        releases.append((alias, request_id))
        return await real_release(alias, request_id)

    monkeypatch.setattr(scheduler, "acquire", spy_acquire)
    monkeypatch.setattr(scheduler, "release", spy_release)
    return acquires, releases


# ---------------------------------------------------------------------------
# (1) happy path: text upstream→client, binary client→upstream (echo), release
# ---------------------------------------------------------------------------
def test_ws_happy_relay(client: TestClient, session: Session, monkeypatch) -> None:
    seed_asr(session, alias="asrws-a", machine_uid="asrws-a-m")
    upstream = FakeUpstream(frames=["transcript-1"], echo=True)
    urls = install_upstream(monkeypatch, upstream)
    scheduler = client.app.state.scheduler
    acquires, releases = spy_scheduler(scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-a") as ws:
        # upstream→client canned text frame flows through untouched.
        assert ws.receive_text() == "transcript-1"
        # client→upstream binary frame is forwarded (echoed back as proof).
        ws.send_bytes(b"\x00\x01\x02audio")
        assert ws.receive_bytes() == b"\x00\x01\x02audio"

    assert urls == ["ws://127.0.0.1:8081/v1/audio/transcriptions/stream?model=asrws-a"]
    assert upstream.sent == [b"\x00\x01\x02audio"]
    assert [a for a, _ in acquires] == ["asrws-a"]
    assert [a for a, _ in releases] == ["asrws-a"]
    assert scheduler.active_count("asrws-a") == 0


# ---------------------------------------------------------------------------
# (2) text client→upstream passthrough (control frame)
# ---------------------------------------------------------------------------
def test_ws_text_client_to_upstream(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-tx", machine_uid="asrws-tx-m")
    upstream = FakeUpstream(echo=True)
    install_upstream(monkeypatch, upstream)
    _, releases = spy_scheduler(client.app.state.scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-tx") as ws:
        ws.send_text('{"type":"start","model":"asrws-tx"}')
        assert ws.receive_text() == '{"type":"start","model":"asrws-tx"}'

    assert upstream.sent == ['{"type":"start","model":"asrws-tx"}']
    assert [a for a, _ in releases] == ["asrws-tx"]


# ---------------------------------------------------------------------------
# (3) client disconnect mid-stream: upstream closed + slot released
# ---------------------------------------------------------------------------
def test_ws_client_disconnect_releases(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-disc", machine_uid="asrws-disc-m")
    upstream = FakeUpstream(frames=["partial"], echo=True)
    install_upstream(monkeypatch, upstream)
    scheduler = client.app.state.scheduler
    _, releases = spy_scheduler(scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-disc") as ws:
        assert ws.receive_text() == "partial"
        # Exiting the context sends a websocket.disconnect mid-stream.

    assert upstream.closed is True
    assert [a for a, _ in releases] == ["asrws-disc"]
    assert scheduler.active_count("asrws-disc") == 0


# ---------------------------------------------------------------------------
# (4) upstream busy (agent closes 1013) -> relayed to client + released
# ---------------------------------------------------------------------------
def test_ws_upstream_busy_relayed(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-busy", machine_uid="asrws-busy-m")
    upstream = FakeUpstream(close=(1013, "busy"))
    install_upstream(monkeypatch, upstream)
    scheduler = client.app.state.scheduler
    _, releases = spy_scheduler(scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-busy") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1013
    assert exc.value.reason == "busy"
    assert upstream.closed is True
    assert [a for a, _ in releases] == ["asrws-busy"]
    assert scheduler.active_count("asrws-busy") == 0


# ---------------------------------------------------------------------------
# (5) upstream dies mid-stream (after one frame) -> client closed + released
# ---------------------------------------------------------------------------
def test_ws_upstream_dies_mid_stream(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-die", machine_uid="asrws-die-m")
    upstream = FakeUpstream(frames=["partial"], close=(1011, "boom"))
    install_upstream(monkeypatch, upstream)
    scheduler = client.app.state.scheduler
    _, releases = spy_scheduler(scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-die") as ws:
        assert ws.receive_text() == "partial"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1011
    assert [a for a, _ in releases] == ["asrws-die"]
    assert scheduler.active_count("asrws-die") == 0


# ---------------------------------------------------------------------------
# (6) oversized inbound frame -> 1009 + released
# ---------------------------------------------------------------------------
def test_ws_oversized_frame_closes_1009(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-big", machine_uid="asrws-big-m")
    monkeypatch.setattr(settings, "AUDIO_WS_MAX_FRAME_BYTES", 4)
    upstream = FakeUpstream()  # held open; never emits
    install_upstream(monkeypatch, upstream)
    scheduler = client.app.state.scheduler
    _, releases = spy_scheduler(scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-big") as ws:
        ws.send_bytes(b"x" * 10)
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1009
    assert [a for a, _ in releases] == ["asrws-big"]
    assert scheduler.active_count("asrws-big") == 0


# ---------------------------------------------------------------------------
# (7) max connection duration -> closed + released
# ---------------------------------------------------------------------------
def test_ws_max_duration_closes(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-dur", machine_uid="asrws-dur-m")
    monkeypatch.setattr(settings, "AUDIO_WS_MAX_DURATION_SECONDS", 0.1)
    monkeypatch.setattr(settings, "AUDIO_WS_IDLE_TIMEOUT_SECONDS", 30.0)
    upstream = FakeUpstream()
    install_upstream(monkeypatch, upstream)
    scheduler = client.app.state.scheduler
    _, releases = spy_scheduler(scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-dur") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1000
    assert "limit" in (exc.value.reason or "")
    assert [a for a, _ in releases] == ["asrws-dur"]
    assert scheduler.active_count("asrws-dur") == 0


# ---------------------------------------------------------------------------
# (8) idle timeout -> closed + released
# ---------------------------------------------------------------------------
def test_ws_idle_timeout_closes(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-idle", machine_uid="asrws-idle-m")
    monkeypatch.setattr(settings, "AUDIO_WS_MAX_DURATION_SECONDS", 3600.0)
    monkeypatch.setattr(settings, "AUDIO_WS_IDLE_TIMEOUT_SECONDS", 0.1)
    upstream = FakeUpstream()
    install_upstream(monkeypatch, upstream)
    scheduler = client.app.state.scheduler
    _, releases = spy_scheduler(scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-idle") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1000
    assert "idle" in (exc.value.reason or "")
    assert [a for a, _ in releases] == ["asrws-idle"]
    assert scheduler.active_count("asrws-idle") == 0


# ---------------------------------------------------------------------------
# (9) pre-flight rejections: 1008, never acquire, never connect
# ---------------------------------------------------------------------------
def _assert_preflight_reject(client: TestClient, monkeypatch, url: str) -> None:
    scheduler = client.app.state.scheduler
    acquires, releases = spy_scheduler(scheduler, monkeypatch)
    urls = install_upstream(monkeypatch, FakeUpstream())
    with client.websocket_connect(url) as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1008
    assert acquires == []
    assert releases == []
    assert urls == []


def test_ws_missing_model_rejected(client: TestClient, monkeypatch) -> None:
    _assert_preflight_reject(client, monkeypatch, PATH)


def test_ws_unknown_alias_rejected(client: TestClient, monkeypatch) -> None:
    _assert_preflight_reject(client, monkeypatch, f"{PATH}?model=nope")


def test_ws_disabled_alias_rejected(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-off", machine_uid="asrws-off-m", enabled=False)
    _assert_preflight_reject(client, monkeypatch, f"{PATH}?model=asrws-off")


def test_ws_llm_alias_rejected(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-llm", machine_uid="asrws-llm-m", modality="llm")
    _assert_preflight_reject(client, monkeypatch, f"{PATH}?model=asrws-llm")


def test_ws_tts_alias_rejected(
    client: TestClient, session: Session, monkeypatch
) -> None:
    # asr-only route: a tts alias must be refused (locked Phase 24 gate).
    seed_asr(session, alias="asrws-tts", machine_uid="asrws-tts-m", modality="tts")
    _assert_preflight_reject(client, monkeypatch, f"{PATH}?model=asrws-tts")


def test_ws_embedding_alias_rejected(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(
        session, alias="asrws-emb", machine_uid="asrws-emb-m", modality="embedding"
    )
    _assert_preflight_reject(client, monkeypatch, f"{PATH}?model=asrws-emb")


# ---------------------------------------------------------------------------
# (10) scheduler failures: 1013, never connect
# ---------------------------------------------------------------------------
def test_ws_no_provider_closes_1013(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-np", machine_uid="asrws-np-m")
    scheduler = client.app.state.scheduler

    async def raising_acquire(alias, request_id):  # noqa: ARG001
        raise NoProviderAvailable(f"no connected provider instance for alias '{alias}'")

    monkeypatch.setattr(scheduler, "acquire", raising_acquire)
    urls = install_upstream(monkeypatch, FakeUpstream())

    with client.websocket_connect(f"{PATH}?model=asrws-np") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1013
    assert urls == []
    assert scheduler.active_count("asrws-np") == 0


def test_ws_queue_timeout_closes_1013(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-qt", machine_uid="asrws-qt-m")
    scheduler = client.app.state.scheduler

    async def timeout_acquire(alias, request_id):  # noqa: ARG001
        raise QueueTimeout("queue timeout")

    monkeypatch.setattr(scheduler, "acquire", timeout_acquire)
    urls = install_upstream(monkeypatch, FakeUpstream())

    with client.websocket_connect(f"{PATH}?model=asrws-qt") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1013
    assert urls == []


# ---------------------------------------------------------------------------
# (11) upstream connect failure -> 1011 + released
# ---------------------------------------------------------------------------
def test_ws_upstream_connect_failure_closes_1011(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-conn", machine_uid="asrws-conn-m")
    scheduler = client.app.state.scheduler
    _, releases = spy_scheduler(scheduler, monkeypatch)

    async def failing_connect(url, *args, **kwargs):  # noqa: ARG001
        raise OSError("connection refused")

    monkeypatch.setattr(route_mod.websockets, "connect", failing_connect)

    with client.websocket_connect(f"{PATH}?model=asrws-conn") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1011
    assert [a for a, _ in releases] == ["asrws-conn"]
    assert scheduler.active_count("asrws-conn") == 0


# ---------------------------------------------------------------------------
# (12) upstream→client BINARY frame forwarded intact (locks send_bytes branch)
# ---------------------------------------------------------------------------
def test_ws_upstream_binary_frame_to_client(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-bin", machine_uid="asrws-bin-m")
    payload = b"\x89PNG\r\n\x1a\n\x00\x01\x02\xff\xfe"
    upstream = FakeUpstream(frames=[payload])  # held open after the frame
    install_upstream(monkeypatch, upstream)
    scheduler = client.app.state.scheduler
    _, releases = spy_scheduler(scheduler, monkeypatch)

    with client.websocket_connect(f"{PATH}?model=asrws-bin") as ws:
        assert ws.receive_bytes() == payload

    assert [a for a, _ in releases] == ["asrws-bin"]
    assert scheduler.active_count("asrws-bin") == 0


# ---------------------------------------------------------------------------
# (13) SHIELDED release: cancelling the endpoint task mid-relay still runs
# scheduler.release exactly once AND lets it complete (the shield defends the
# in-flight release from a second cancellation delivered during cleanup).
# ---------------------------------------------------------------------------
def test_ws_release_is_shielded_on_task_cancellation(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr(session, alias="asrws-cancel", machine_uid="asrws-cancel-m")
    scheduler = client.app.state.scheduler
    install_upstream(monkeypatch, FakeUpstream())  # held open

    releases_called: list[str] = []
    releases_done: list[str] = []
    real_release = scheduler.release

    async def spy_release(alias: str, request_id: str):
        releases_called.append(alias)
        await asyncio.sleep(0.1)  # slow: the second cancel lands mid-release
        await real_release(alias, request_id)
        releases_done.append(alias)

    monkeypatch.setattr(scheduler, "release", spy_release)

    # Replace the relay with a coroutine that parks, then cancels the handler
    # task twice: once to unwind into the shielded finally, once while the
    # release is in flight. Without the shield the second cancel would abort the
    # release before releases_done is appended.
    async def cancelling_relay(websocket, upstream, **kwargs):  # noqa: ARG001
        task = asyncio.current_task()

        async def canceller():
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.sleep(0.02)
            task.cancel()

        asyncio.create_task(canceller())
        await asyncio.sleep(1000)

    monkeypatch.setattr(route_mod, "_relay", cancelling_relay)

    with (
        contextlib.suppress(Exception),
        client.websocket_connect(f"{PATH}?model=asrws-cancel"),
    ):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not releases_done:
            time.sleep(0.02)

    assert releases_called == ["asrws-cancel"]
    assert releases_done == ["asrws-cancel"]  # completed despite the cancels
    assert scheduler.active_count("asrws-cancel") == 0
