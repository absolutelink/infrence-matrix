from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.agent_manager import agent_manager
from app.services.inference_scheduler import (
    get_slot_telemetry,
    normalize_slot_telemetry,
)


def test_normalize_wrapped_slots():
    telemetry = normalize_slot_telemetry(
        {"slots": [{"id": 0, "state": 0}, {"id": 1, "state": 1}]}
    )

    assert telemetry.capacity == 4
    assert telemetry.active == 1
    assert telemetry.available == 3
    assert telemetry.known


def test_normalize_unavailable_payload_is_safe():
    telemetry = normalize_slot_telemetry({"unexpected": True})

    assert (telemetry.capacity, telemetry.active, telemetry.available) == (4, 0, 4)
    assert not telemetry.known


def test_normalize_uses_configured_parallel_capacity():
    telemetry = normalize_slot_telemetry(
        {"slots": [{"state": 1}, {"state": 0}]}, capacity=8
    )

    assert telemetry.capacity == 8
    assert telemetry.active == 1
    assert telemetry.available == 7


@pytest.mark.asyncio
async def test_flash_uses_kv_slots_for_capacity(monkeypatch):
    server = SimpleNamespace(
        id=uuid4(),
        agent_id=uuid4(),
        engine="halogen-flash",
        engine_options={"kv_slots": 2},
        server_options={},
    )
    monkeypatch.setattr(agent_manager, "get_agent", AsyncMock(return_value=None))

    telemetry = await get_slot_telemetry(server)

    assert (telemetry.capacity, telemetry.available) == (2, 2)
