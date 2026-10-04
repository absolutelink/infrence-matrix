"""Phase 4 mock provider e2e: admin-driven boot + /v1 streaming.

Drives the mock against the same fake-admin harness style as
``test_mock_e2e.py``: register over HTTP, WS for commands/events, then
send ``backend.start`` / ``backend.stop`` from the fake admin and assert
the provider emits backend.status transitions. Finally hit the shared
lifecycle's /v1 surface locally (ASGITransport) and assert streaming +
slot accounting, including the release-on-upstream-close invariant over
a real HTTP response.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import pytest
import websockets
from provider_lib.admin_client import AdminClient
from provider_lib.backend import BackendNotReady
from provider_lib.config import ProviderSettings
from provider_lib.wire import BackendStatusValue, Frame

from provider_mock.main import (
    DEFAULT_CAPACITY,
    apply_registration,
    build_app,
    make_lifecycle,
    register_and_connect,
)

INSTANCE_ID = "11111111-2222-3333-4444-555555555555"
SECRET = "fake-instance-secret"
CAPACITY = 2

WS_PORT_HOLDER: dict[str, int] = {}
RECEIVED: list[Frame] = []
ADMIN_WS_URL_HOLDER: dict[str, str] = {}


class _FakeAdminHTTP(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802  (http.server API)
        assert self.path == "/admin/api/providers/register"
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        assert body["provider_type"] == "mock"
        payload = json.dumps(
            {
                "instance_id": INSTANCE_ID,
                "instance_secret": SECRET,
                "machine": {"uid": body["machine_uid"]},
                "provider_definition": {
                    "alias": "mock-alias-model",
                    "provider_type": "mock",
                    "config_fingerprint": "deadbeef",
                    "capacity": CAPACITY,
                    "admin_ws_url": ADMIN_WS_URL_HOLDER["url"],
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # noqa: ARG002
        pass


async def _fake_ws_admin(websocket):
    auth = websocket.request.headers.get("Authorization", "")
    if auth != f"Bearer {SECRET}":
        await websocket.close(code=4401)
        return
    await websocket.send(
        Frame(type="provider.hello", epoch=7, payload={"epoch": 7}).to_json()
    )
    async for raw in websocket:
        RECEIVED.append(Frame.model_validate_json(raw))


@pytest.fixture
async def fake_admin():
    RECEIVED.clear()

    http = HTTPServer(("127.0.0.1", 0), _FakeAdminHTTP)
    http_thread = threading.Thread(target=http.serve_forever, daemon=True)
    http_thread.start()

    stop = asyncio.Event()

    async def serve_ws():
        async with websockets.serve(
            _fake_ws_admin, "127.0.0.1", 0, ping_interval=None
        ) as srv:
            port = srv.sockets[0].getsockname()[1]
            ADMIN_WS_URL_HOLDER["url"] = f"ws://127.0.0.1:{port}/provider/ws"
            await stop.wait()

    ws_task = asyncio.get_running_loop().create_task(serve_ws())
    while "url" not in ADMIN_WS_URL_HOLDER:
        await asyncio.sleep(0.01)

    yield f"http://127.0.0.1:{http.server_address[1]}"

    stop.set()
    ws_task.cancel()
    http.shutdown()
    http.server_close()


def _settings(base_url: str) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="mock-e2e",
        PROVIDER_REGISTRATION_TOKEN="tok",
        ADMIN_BASE_URL=base_url,
        PROVIDER_PORT=8081,
        CACHE_DIR="/tmp/mock_provider_test_cache",
        MODELS_DIR="/tmp/mock_provider_test_models",
    )


async def _await_frames(pred, count: int = 1, timeout: float = 5.0) -> list[Frame]:
    async def poll() -> None:
        while True:
            hits = [f for f in RECEIVED if pred(f)]
            if len(hits) >= count:
                return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)
    return [f for f in RECEIVED if pred(f)]


async def test_admin_driven_boot_and_v1_streaming(fake_admin) -> None:
    client = AdminClient(_settings(fake_admin))
    lifecycle = make_lifecycle(client)
    try:
        result = await register_and_connect(client, lifecycle)
        assert result.instance_id == INSTANCE_ID

        # Registration config was adopted: capacity + model alias.
        assert lifecycle.capacity == CAPACITY
        assert lifecycle.driver.model == "mock-alias-model"

        # Backend is NOT auto-started; admin drives it.
        assert lifecycle.backend_status == BackendStatusValue.STOPPED
        with pytest.raises(BackendNotReady):
            await lifecycle.acquire_slot()

        # Fake admin sends backend.start; provider handler awaits it.
        start_cmd = Frame(type="backend.start", id="cmd-start-1", epoch=7, payload={})
        # The provider's client is connected; emulate the admin pushing
        # the command by feeding it into the client's dispatch directly
        # (the fake WS server would need bidirectional plumbing; the
        # dispatch path is what the recv loop calls).
        await client._dispatch(start_cmd)

        acks = await _await_frames(
            lambda f: f.type == "ack" and f.reply_to == "cmd-start-1"
        )
        assert acks[0].payload["ok"] is True
        assert acks[0].payload["detail"]["capacity"] == CAPACITY

        # backend.status RUNNING reached the admin.
        statuses = await _await_frames(
            lambda f: (
                f.type == "backend.status"
                and f.payload.get("backend_status") == BackendStatusValue.RUNNING
            )
        )
        assert statuses
        assert lifecycle.backend_status == BackendStatusValue.RUNNING

        # /v1 through the SAME lifecycle instance.
        app = build_app(lifecycle)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://provider"
        ) as v1:
            models = await v1.get("/v1/models")
            assert models.status_code == 200
            assert models.json()["data"][0]["id"] == "mock-alias-model"

            resp = await v1.post("/v1/responses", json={"input": "go"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            types = [
                json.loads(line[len("data: ") :])["type"]
                for line in resp.text.splitlines()
                if line.startswith("data: ")
            ]
            assert types[0] == "response.created"
            assert types[-1] == "response.completed"
            assert "response.output_text.delta" in types
            # Slot released after the stream completed.
            assert lifecycle.in_flight == 0
            assert lifecycle.backend_status == BackendStatusValue.RUNNING

        # backend.stop from the admin drives the lifecycle down.
        stop_cmd = Frame(type="backend.stop", id="cmd-stop-1", epoch=7, payload={})
        await client._dispatch(stop_cmd)
        await _await_frames(lambda f: f.type == "ack" and f.reply_to == "cmd-stop-1")
        assert lifecycle.backend_status == BackendStatusValue.STOPPED
    finally:
        await client.disconnect()


async def test_v1_429_when_at_capacity(fake_admin) -> None:
    client = AdminClient(_settings(fake_admin))
    lifecycle = make_lifecycle(client)
    try:
        await register_and_connect(client, lifecycle)
        await lifecycle.start()
        app = build_app(lifecycle)
        # Occupy one slot of two so a second concurrent request fits;
        # then fill both and expect 429.
        await lifecycle.acquire_slot()
        await lifecycle.acquire_slot()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://provider"
        ) as v1:
            resp = await v1.post("/v1/responses", json={})
            assert resp.status_code == 429
            assert lifecycle.in_flight == 2
        await lifecycle.release_slot()
        await lifecycle.release_slot()
    finally:
        await client.disconnect()


def test_apply_registration_defaults() -> None:
    """Missing capacity falls back to the library default."""
    client = AdminClient(_settings("http://127.0.0.1:1"))
    lifecycle = make_lifecycle(client)
    assert lifecycle.capacity == DEFAULT_CAPACITY

    class _R:
        provider_definition: dict[str, Any] = {"alias": "zzz", "capacity": 3}

    apply_registration(lifecycle, _R())  # type: ignore[arg-type]
    assert lifecycle.capacity == 3
    assert lifecycle.driver.model == "zzz"


async def test_v1_real_socket_streaming_and_disconnect(fake_admin) -> None:
    """Real uvicorn socket: stream /v1/responses, disconnect mid-stream,
    and the slot must free when the upstream generator closes."""
    import uvicorn

    client = AdminClient(_settings(fake_admin))
    # Long, slow stream so the client can bail before exhaustion.
    lifecycle = make_lifecycle(client, delta_count=100, delta_delay=0.02)
    try:
        await register_and_connect(client, lifecycle)
        await lifecycle.start()
        config = uvicorn.Config(
            build_app(lifecycle), host="127.0.0.1", port=0, log_level="warning"
        )
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
                health = await http.get("/health")
                assert health.json()["backend_status"] == "running"
                got = 0
                async with http.stream(
                    "POST", "/v1/responses", json={"input": "go"}
                ) as resp:
                    assert resp.status_code == 200
                    assert resp.headers["content-type"].startswith("text/event-stream")
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            got += 1
                        if got >= 2:
                            break
                # Client closed mid-stream; the upstream generator's close
                # (via uvicorn disconnect propagation) frees the slot.
                for _ in range(200):
                    if lifecycle.in_flight == 0:
                        break
                    await asyncio.sleep(0.02)
                assert lifecycle.in_flight == 0
                assert lifecycle.backend_status == BackendStatusValue.RUNNING
                assert lifecycle.driver.stream_close_count == 1
        finally:
            server.should_exit = True
            await serve_task
    finally:
        await client.disconnect()
