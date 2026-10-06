"""Phase 9 wiring test: gufo installs the shared config.update /
cache.clear / prune_unused handlers and applies a config change against
the FAKE gufo server."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
import websockets
from conftest import make_settings
from provider_lib.admin_client import AdminClient
from provider_lib.config_update import prompt_cache_dir
from provider_lib.wire import BackendStatusValue, Frame

from provider_gufo.main import (
    PROVIDER_TYPE,
    install_command_handlers,
    make_lifecycle,
    register_and_connect,
)

INSTANCE_ID = "eeeeeeee-2222-3333-4444-555555555555"
SECRET = "fake-gufo-secret"
RECEIVED: list[Frame] = []
ADMIN_WS_URL_HOLDER: dict[str, str] = {}
BACKEND_CONFIG: dict = {}


class _FakeAdminHTTP(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        payload = json.dumps(
            {
                "instance_id": INSTANCE_ID,
                "instance_secret": SECRET,
                "machine": {"uid": body["machine_uid"]},
                "provider_definition": {
                    "alias": "llama-alias",
                    "provider_type": PROVIDER_TYPE,
                    "config_fingerprint": "cafe",
                    "capacity": 1,
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


async def _await_ack(client: AdminClient, reply_to: str) -> dict[str, Any]:  # noqa: ARG001
    async def poll() -> Frame:
        while True:
            hits = [f for f in RECEIVED if f.type == "ack" and f.reply_to == reply_to]
            if hits:
                return hits[0]
            await asyncio.sleep(0.02)

    frame = await asyncio.wait_for(poll(), timeout=15)
    return frame.payload


class _NoopEmitter:
    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def test_handlers_installed(tmp_path, fake_gufo_binary) -> None:
    settings = make_settings(tmp_path, fake_gufo_binary)
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client, {})
    install_command_handlers(client, lifecycle, _NoopEmitter())  # type: ignore[arg-type]
    for name in ("provider.config.update", "cache.clear", "storage.prune_unused"):
        assert name in client._command_handlers


async def test_config_update_applies_new_model_and_clears_cache(
    fake_admin, tmp_path, fake_gufo_binary, local_model_file
) -> None:
    BACKEND_CONFIG.clear()
    BACKEND_CONFIG.update({"artifacts": {"model": {"path": local_model_file}}})
    settings = make_settings(tmp_path, fake_gufo_binary, ADMIN_BASE_URL=fake_admin)
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client, BACKEND_CONFIG)
    try:
        await register_and_connect(client, lifecycle)
        # Boot once so the driver has a resolved artifact set.
        await client._dispatch(
            Frame(type="backend.start", id="c-start", epoch=7, payload={})
        )
        ack = await _await_ack(client, "c-start")
        assert ack["ok"] is True

        # Seed the old fingerprint's prompt-cache dir.
        old = prompt_cache_dir(settings, "cafe")
        old.mkdir(parents=True, exist_ok=True)
        (old / "slot.bin").write_bytes(b"x" * 40)

        new_cfg = {
            "artifacts": {"model": {"path": local_model_file}},
            "server": {"sessions": 2},
        }
        await client._dispatch(
            Frame(
                type="provider.config.update",
                id="c-cfg",
                epoch=7,
                payload={
                    "backend_config": new_cfg,
                    "config_fingerprint": "beef",
                    "capacity": 1,
                },
            )
        )
        ack = await _await_ack(client, "c-cfg")
        assert ack["ok"] is True, ack
        assert ack["detail"]["config_fingerprint"] == "beef"
        assert ack["detail"]["model_metadata"] == [
            {"id": "served-alpha", "object": "model", "owned_by": "gufo"},
            {"id": "served-beta", "object": "model", "owned_by": "gufo"},
        ]
        assert not old.exists()  # prompt cache cleared
        assert lifecycle.backend_status == BackendStatusValue.RUNNING
        assert lifecycle.driver._config == new_cfg
        assert lifecycle.driver.resolved_artifacts == [local_model_file]
    finally:
        await client.disconnect()
        await lifecycle.driver.aclose()
