"""End-to-end mock provider <-> fake admin over real sockets.

Drives provider_mock.register_and_connect() against a minimal fake admin:
an in-process aiohttp-free HTTP handler plus a websockets server. The
registration response points admin_ws_url at the local fake WS so no real
admin is required.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import websockets
from provider_lib.admin_client import AdminClient
from provider_lib.config import ProviderSettings
from provider_lib.wire import Frame

INSTANCE_ID = "11111111-2222-3333-4444-555555555555"
SECRET = "fake-instance-secret"

WS_PORT_HOLDER: dict[str, int] = {}
RECEIVED: list[Frame] = []
ADMIN_WS_URL_HOLDER: dict[str, str] = {}


class _FakeAdminHTTP(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802  (http.server API)
        assert self.path == "/admin/api/providers/register"
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        assert body["provider_type"] == "mock"
        assert body["machine_uid"] == "mock-e2e"
        payload = json.dumps(
            {
                "instance_id": INSTANCE_ID,
                "instance_secret": SECRET,
                "machine": {"uid": body["machine_uid"]},
                "provider_definition": {
                    "alias": "mock-model",
                    "provider_type": "mock",
                    "config_fingerprint": "deadbeef",
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


async def test_mock_register_and_connect(fake_admin) -> None:
    from provider_mock.main import register_and_connect

    client = AdminClient(_settings(fake_admin))
    try:
        result = await register_and_connect(client)
        assert result.instance_id == INSTANCE_ID
        assert client.epoch == 7

        # The initial provider.status event must reach the fake admin with
        # the hello epoch stamped on it.
        for _ in range(300):
            if RECEIVED:
                break
            await asyncio.sleep(0.01)
        assert RECEIVED, "no events reached the fake admin"
        status = RECEIVED[0]
        assert status.type == "provider.status"
        assert status.epoch == 7
        assert status.payload["instance_status"] == "running"
    finally:
        await client.disconnect()
