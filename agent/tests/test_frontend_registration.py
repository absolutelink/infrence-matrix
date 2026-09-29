"""Tests for agent registration capability negotiation."""

import os
from unittest.mock import AsyncMock

os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"
os.environ["MODELS_PATH"] = "/tmp/models"

import pytest

from app.services.frontend_client import FrontendClient
from app.services.inference_operations import INFERENCE_SLOT_PROTOCOL_VERSION


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
async def test_registration_advertises_agent_slot_protocol(monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr(
        "app.services.frontend_client.httpx.AsyncClient",
        lambda: FakeAsyncClient(payloads),
    )
    client = FrontendClient()
    monkeypatch.setattr(client, "_get_gpu_info", AsyncMock(return_value={}))

    assert await client.register()
    assert payloads[0]["inference_slot_protocol"] == INFERENCE_SLOT_PROTOCOL_VERSION
