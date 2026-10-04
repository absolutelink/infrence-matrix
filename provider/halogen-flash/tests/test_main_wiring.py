"""Provider main wiring test: admin-driven boot + usage-normalized stream."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
import websockets
from conftest import make_settings
from provider_lib.admin_client import AdminClient
from provider_lib.backend import BackendNotReady
from provider_lib.wire import BackendStatusValue, Frame

from provider_halogen_flash.main import (
    PROVIDER_TYPE,
    VERSION,
    apply_registration,
    build_app,
    make_driver,
    make_lifecycle,
    register_and_connect,
)

INSTANCE_ID = "cccccccc-2222-3333-4444-555555555555"
SECRET = "fake-flash-secret"
CAPACITY = 2

RECEIVED: list[Frame] = []
ADMIN_WS_URL_HOLDER: dict[str, str] = {}
BACKEND_CONFIG: dict = {}


class _FakeAdminHTTP(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        assert self.path == "/admin/api/providers/register"
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        assert body["provider_type"] == PROVIDER_TYPE
        assert body["version"] == VERSION
        payload = json.dumps(
            {
                "instance_id": INSTANCE_ID,
                "instance_secret": SECRET,
                "machine": {"uid": body["machine_uid"]},
                "provider_definition": {
                    "alias": "flash-alias",
                    "provider_type": PROVIDER_TYPE,
                    "config_fingerprint": "cafe",
                    "capacity": CAPACITY,
                    "backend_config": BACKEND_CONFIG,
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


async def _await_frames(pred, timeout: float = 5.0) -> list[Frame]:
    async def poll() -> None:
        while True:
            hits = [f for f in RECEIVED if pred(f)]
            if hits:
                return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)
    return [f for f in RECEIVED if pred(f)]


async def test_admin_driven_boot_and_normalized_stream(
    fake_admin, tmp_path, fake_flash_binary, local_artifacts
) -> None:
    BACKEND_CONFIG.clear()
    BACKEND_CONFIG.update(
        {
            "model": {"path": local_artifacts["checkpoint"]},
            "tokenizer": {"path": local_artifacts["tokenizer"]},
            "options": {"kv_slots": 2},
        }
    )
    settings = make_settings(tmp_path, fake_flash_binary, ADMIN_BASE_URL=fake_admin)
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client, BACKEND_CONFIG)
    try:
        result, _lc, _em = await register_and_connect(client, lifecycle)
        assert result.instance_id == INSTANCE_ID
        assert lifecycle.capacity == CAPACITY
        assert lifecycle.backend_status == BackendStatusValue.STOPPED
        with pytest.raises(BackendNotReady):
            await lifecycle.acquire_slot()

        start_cmd = Frame(type="backend.start", id="cmd-start-1", epoch=7, payload={})
        await client._dispatch(start_cmd)
        acks = await _await_frames(
            lambda f: f.type == "ack" and f.reply_to == "cmd-start-1"
        )
        assert acks[0].payload["ok"] is True
        detail = acks[0].payload["detail"]
        assert detail["backend"] == PROVIDER_TYPE
        assert detail["effective_capacity"] == 2
        # Static fingerprint-stable ports (not PROVIDER_PORT-derived).
        assert detail["api_port"] >= 8200
        await _await_frames(
            lambda f: (
                f.type == "backend.status"
                and f.payload.get("backend_status") == BackendStatusValue.RUNNING
            )
        )

        app = build_app(lifecycle)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://provider"
        ) as v1:
            resp = await v1.post("/v1/responses", json={"input": "go", "stream": True})
            assert resp.status_code == 200
            events = [
                json.loads(line[len("data: ") :])
                for line in resp.text.splitlines()
                if line.startswith("data: ")
            ]
            assert events[-1]["type"] == "response.completed"
            usage = events[-1]["response"]["usage"]
            # The override normalized the fake's chat-style blob.
            assert usage["input_tokens"] == 100
            assert usage["output_tokens"] == 8
            assert "prompt_tokens" not in usage
            assert lifecycle.in_flight == 0

        stop_cmd = Frame(type="backend.stop", id="cmd-stop-1", epoch=7, payload={})
        await client._dispatch(stop_cmd)
        await _await_frames(lambda f: f.type == "ack" and f.reply_to == "cmd-stop-1")
        assert lifecycle.backend_status == BackendStatusValue.STOPPED
    finally:
        await client.disconnect()
        await lifecycle.driver.aclose()


def test_make_driver_wires_callbacks(tmp_path, fake_flash_binary) -> None:
    settings = make_settings(tmp_path, fake_flash_binary)
    driver = make_driver(settings, {})
    assert driver._binary == settings.HALOGEN_FLASH_SERVER_PATH


def test_apply_registration_adopts_capacity(tmp_path, fake_flash_binary) -> None:
    settings = make_settings(tmp_path, fake_flash_binary)
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client, {})

    class _R:
        provider_definition = {
            "capacity": 7,
            "backend_config": {"options": {"kv_slots": 7}},
        }

    apply_registration(lifecycle, _R())  # type: ignore[arg-type]
    assert lifecycle.capacity == 7
    assert lifecycle.driver.effective_capacity == 7


def test_build_app_reports_provider_type() -> None:
    app = build_app()
    routes = {r.path for r in app.routes}
    assert "/v1/responses" in routes
    assert "/health" in routes
