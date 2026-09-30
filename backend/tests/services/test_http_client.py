"""Tests for the process-wide shared httpx client."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import app.services.http_client as http_client_module
from app.services.http_client import aclose_http_client, get_http_client


@pytest.fixture(autouse=True)
async def _reset_shared_client() -> None:
    """Keep the process-wide singleton from leaking between tests."""
    await aclose_http_client()
    yield
    await aclose_http_client()


class _KeepAliveHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    connections = 0
    requests = 0

    def setup(self) -> None:
        type(self).connections += 1
        super().setup()

    def do_GET(self) -> None:
        type(self).requests += 1
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def _start_server() -> tuple[ThreadingHTTPServer, int]:
    handler = type("Handler", (_KeepAliveHandler,), {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


def test_get_http_client_is_shared() -> None:
    client = get_http_client()

    assert client is get_http_client()
    assert isinstance(client, httpx.AsyncClient)
    assert client.is_closed is False


@pytest.mark.asyncio
async def test_shared_client_reuses_connection() -> None:
    server, port = _start_server()
    handler = server.RequestHandlerClass
    assert handler is not None
    try:
        client = get_http_client()
        first = await client.get(f"http://127.0.0.1:{port}/one")
        second = await client.get(f"http://127.0.0.1:{port}/two")

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.text == "ok"
        # Two requests rode a single TCP connection: no second handshake.
        assert handler.requests == 2
        assert handler.connections == 1
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_aclose_resets_singleton() -> None:
    client = get_http_client()
    await aclose_http_client()

    assert http_client_module._shared_client is None

    replacement = get_http_client()
    assert replacement is not client
    assert replacement.is_closed is False
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_aclose_is_idempotent() -> None:
    await aclose_http_client()
    await aclose_http_client()
    assert http_client_module._shared_client is None


def test_shared_client_pool_limits() -> None:
    pool = get_http_client()._transport._pool

    assert pool._max_connections == 500
    assert pool._max_keepalive_connections == 100
    assert pool._keepalive_expiry == 30.0
    assert get_http_client().timeout.read == 300.0


@pytest.mark.asyncio
async def test_lifespan_closes_shared_client() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = get_http_client()
    assert client.is_closed is False

    with TestClient(app):
        pass

    assert client.is_closed is True
    assert http_client_module._shared_client is None
