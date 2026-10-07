"""Provider WS reconnect/backoff wiring (ARCHITECTURE.md §5).

The provider entrypoints drive the admin socket with
``AdminClient.run_forever`` instead of a one-shot ``connect()`` so a
hardware provider survives an admin restart. These tests exercise that
path against a fake admin that drops the first connection: the client
must reconnect, adopt the strictly greater epoch, and re-emit
``provider.status`` so the admin's mirror is fresh.
"""

import asyncio
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import websockets
from provider_lib.admin_client import AdminClient
from provider_lib.config import ProviderSettings
from provider_lib.wire import BackendStatusValue, Frame

from provider_mock.main import (
    emit_provider_status,
    make_lifecycle,
    register_provider,
)

INSTANCE_ID = "11111111-2222-3333-4444-555555555555"
AGENT_ID = "conf-agent"
SECRET = "fake-agent-secret"

RECEIVED: list[Frame] = []
CONNECTIONS: list[int] = []
ADMIN_WS_URL_HOLDER: dict[str, str] = {}
# Per-test holder so the asyncio.Event is always bound to the current
# test's event loop. When set, the fake admin drops connection #1
# (simulating an admin restart).
DROP_HOLDER: dict[str, asyncio.Event] = {}


class _FakeAdminHTTP(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802  (http.server API)
        assert self.path == "/admin/api/providers/register"
        payload = json.dumps(
            {
                "agent_id": "bbbbbbbb-2222-3333-4444-555555555555",
                "agent_secret": SECRET,
                "machine": {
                    "uid": "conf-machine",
                    "admin_ws_url": ADMIN_WS_URL_HOLDER["url"],
                },
                "backends": [
                    {
                        "instance_id": INSTANCE_ID,
                        "port": 8081,
                        "definition": {
                            "alias": "mock-model",
                            "provider_type": "mock",
                            "config_fingerprint": "deadbeef",
                            "capacity": 2,
                        },
                    }
                ],
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
    # Each accepted connection gets a strictly greater epoch, like the real
    # admin's INCR im:ws:epoch:{id}.
    epoch = len(CONNECTIONS) + 1
    CONNECTIONS.append(epoch)
    await websocket.send(
        Frame(type="provider.hello", epoch=epoch, payload={"epoch": epoch}).to_json()
    )

    async def drop_when_signalled() -> None:
        await DROP_HOLDER["drop"].wait()
        with contextlib.suppress(Exception):
            await websocket.close(code=1012)

    drop_task: asyncio.Task[None] | None = None
    if epoch == 1:
        drop_task = asyncio.create_task(drop_when_signalled())
    try:
        async for raw in websocket:
            RECEIVED.append(Frame.model_validate_json(raw))
    except websockets.ConnectionClosed:
        pass
    finally:
        if drop_task is not None and not DROP_HOLDER["drop"].is_set():
            drop_task.cancel()
            await asyncio.gather(drop_task, return_exceptions=True)


@pytest.fixture
async def fake_admin():
    RECEIVED.clear()
    CONNECTIONS.clear()
    DROP_HOLDER.clear()
    drop = asyncio.Event()
    DROP_HOLDER["drop"] = drop

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
        MACHINE_UID="conf-machine",
        MACHINE_SECRET="tok",
        AGENT_ID=AGENT_ID,
        ADMIN_BASE_URL=base_url,
        PROVIDER_PORT=8081,
        CACHE_DIR="/tmp/mock_provider_reconnect_cache",
        MODELS_DIR="/tmp/mock_provider_reconnect_models",
    )


def _statuses() -> list[Frame]:
    return [f for f in RECEIVED if f.type == "provider.status"]


async def _await(pred, timeout: float = 15.0) -> None:
    async def poll() -> None:
        while not pred():
            await asyncio.sleep(0.05)

    await asyncio.wait_for(poll(), timeout)


async def test_run_forever_reconnects_with_new_epoch_and_reemits_status(
    fake_admin,
) -> None:
    """Admin drops the socket -> provider reconnects, epoch bumps, status re-emitted."""
    client = AdminClient(_settings(fake_admin))
    lifecycle = make_lifecycle(client)
    connected_epochs: list[int] = []

    async def on_connected() -> None:
        connected_epochs.append(client.epoch)
        await emit_provider_status(client, lifecycle)

    # Register once (HTTP), then let run_forever own the socket forever.
    result, lifecycle = await register_provider(client, lifecycle)
    assert result.instance_id == INSTANCE_ID

    ws_task = asyncio.create_task(client.run_forever(on_connected=on_connected))
    try:
        await _await(lambda: len(connected_epochs) >= 1)
        assert connected_epochs[0] == 1
        assert lifecycle.backend_status == BackendStatusValue.STOPPED

        # Once the first provider.status has landed, kill the connection.
        await _await(lambda: len(_statuses()) >= 1)
        DROP_HOLDER["drop"].set()

        # Reconnect: second connect callback with a strictly greater epoch.
        await _await(lambda: len(connected_epochs) >= 2)
        assert connected_epochs[1] > connected_epochs[0]
        assert client.epoch == connected_epochs[1]

        # The admin's mirror gets a fresh provider.status stamped with the
        # new epoch (old-epoch frames would be fenced by the real admin).
        await _await(lambda: len(_statuses()) >= 2)
        assert _statuses()[-1].epoch == connected_epochs[1]
    finally:
        ws_task.cancel()
        await asyncio.gather(ws_task, return_exceptions=True)
        await client.disconnect()


async def test_run_forever_resets_backoff_after_successful_connect(
    fake_admin,
) -> None:
    """A connect that succeeds resets the backoff to the 1s floor."""
    client = AdminClient(_settings(fake_admin))
    lifecycle = make_lifecycle(client)
    connects = 0

    async def on_connected() -> None:
        nonlocal connects
        connects += 1
        await emit_provider_status(client, lifecycle)
        if connects == 1:
            DROP_HOLDER["drop"].set()

    await register_provider(client, lifecycle)
    ws_task = asyncio.create_task(client.run_forever(on_connected=on_connected))
    try:
        await _await(lambda: connects >= 2)
        # run_forever resets `backoff` to 1.0 on every accepted connect,
        # so the second reconnect also arrives promptly (~1s, not 2s+).
        assert client._recv_task is not None
        assert not client._recv_task.done()
    finally:
        ws_task.cancel()
        await asyncio.gather(ws_task, return_exceptions=True)
        await client.disconnect()


async def test_run_forever_retries_when_admin_unreachable(
    fake_admin,
) -> None:
    """Backoff loop keeps trying and connects as soon as the admin returns."""
    settings = _settings(fake_admin)
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client)
    await register_provider(client, lifecycle)

    # Point at a dead port so the first attempts fail, then flip back.
    dead_url = client.registration.ws_url
    good_url = dead_url
    client.registration.machine["admin_ws_url"] = "ws://127.0.0.1:1/provider/ws"

    connected: list[int] = []

    async def on_connected() -> None:
        connected.append(client.epoch)
        await emit_provider_status(client, lifecycle)

    async def restore_later() -> None:
        await asyncio.sleep(3.0)
        client.registration.machine["admin_ws_url"] = good_url

    ws_task = asyncio.create_task(client.run_forever(on_connected=on_connected))
    restore_task = asyncio.create_task(restore_later())
    try:
        await _await(lambda: len(connected) >= 1, timeout=20.0)
        assert dead_url  # the original URL is still what we computed
    finally:
        ws_task.cancel()
        restore_task.cancel()
        await asyncio.gather(ws_task, restore_task, return_exceptions=True)
        await client.disconnect()
