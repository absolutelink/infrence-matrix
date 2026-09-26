"""Tests for transient upstream connection handling."""

import os
from unittest.mock import AsyncMock, Mock, patch

os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"
os.environ["MODELS_PATH"] = "/tmp/models"

import httpx
import pytest

from app.services.proxy import ServerProxy


@pytest.mark.asyncio
async def test_proxy_request_retries_connection_failures():
    manager = Mock(configs={"server-1": Mock(port=8091)})
    proxy = ServerProxy(manager)
    response = Mock()
    proxy.client.request = AsyncMock(
        side_effect=[httpx.ConnectError("not ready"), response]
    )

    with patch("app.services.proxy.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await proxy.proxy_request("server-1", "GET", "/health", {})

    assert result is response
    assert proxy.client.request.await_count == 2
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_metrics_are_returned_as_prometheus_text(monkeypatch):
    from app.api.routes import proxy as proxy_route

    server_id = "server-1"
    proxy_route.server_manager.servers[server_id] = Mock()
    response = Mock(is_error=False, text="requests_processing 2\n")
    proxy = Mock()
    proxy.proxy_request = AsyncMock(return_value=response)
    monkeypatch.setattr(proxy_route, "ServerProxy", Mock(return_value=proxy))

    from httpx import ASGITransport, AsyncClient
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        result = await client.get(f"/proxy/{server_id}/metrics")

    assert result.status_code == 200
    assert result.json() == {"metrics": "requests_processing 2\n"}
