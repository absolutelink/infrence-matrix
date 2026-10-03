"""Tests for agent registration payload shape."""

import os
from unittest.mock import AsyncMock

os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"
os.environ["MODELS_PATH"] = "/tmp/models"

import pytest

from app.core.config import settings
from app.services.frontend_client import FrontendClient


class FakeResponse:
    def raise_for_status(self) -> None:
        pass


class FakeAsyncClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def post(self, _url: str, *, json: dict) -> FakeResponse:
        self.payloads.append(json)
        return FakeResponse()


@pytest.mark.asyncio
async def test_registration_payload_shape_without_slot_protocol(monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr(
        "app.services.frontend_client.httpx.AsyncClient",
        lambda: FakeAsyncClient(payloads),
    )
    client = FrontendClient()
    monkeypatch.setattr(client, "_get_gpu_info", AsyncMock(return_value={}))

    assert await client.register()
    payload = payloads[0]
    assert payload["agent_id"] == settings.AGENT_ID
    assert payload["name"] == settings.AGENT_NAME
    assert payload["platform"] == settings.AGENT_PLATFORM
    assert payload["type"] == settings.AGENT_TYPE
    assert payload["port"] == settings.AGENT_PORT
    assert "server_statuses" in payload
    # The inference slot protocol concept has been removed entirely.
    assert "inference_slot_protocol" not in payload
