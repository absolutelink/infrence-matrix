from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.agent_manager import agent_manager
from app.services.inference_scheduler import (
    SlotTelemetry,
    _vram_requirements_fit,
    admitted_active_count,
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


@pytest.mark.asyncio
async def test_metrics_use_requests_processing(monkeypatch):
    server = SimpleNamespace(
        id=uuid4(),
        agent_id=uuid4(),
        engine="llamacpp",
        engine_options={},
        server_options={"parallel": 8},
    )
    monkeypatch.setattr(
        agent_manager,
        "get_agent",
        AsyncMock(return_value=SimpleNamespace(status="online")),
    )
    send_to_agent = AsyncMock(
        return_value={
            "metrics": """
# HELP llamacpp:requests_processing Active requests.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 3
"""
        }
    )
    monkeypatch.setattr(agent_manager, "send_to_agent", send_to_agent)

    telemetry = await get_slot_telemetry(server)

    assert (telemetry.capacity, telemetry.active, telemetry.available) == (8, 3, 5)
    send_to_agent.assert_awaited_once()
    assert send_to_agent.await_args.args[2] == f"/proxy/{server.id}/metrics"


def test_vram_admission_uses_configured_requirements():
    assert not _vram_requirements_fit(113, 30, 110)
    assert _vram_requirements_fit(113, 30, 0)


def test_persisted_leases_are_lower_bound_after_slot_reset():
    telemetry = SlotTelemetry(capacity=1, active=0, available=1)

    assert admitted_active_count(1, telemetry) == 1
    assert admitted_active_count(0, telemetry) == 0


def test_unknown_telemetry_does_not_claim_capacity():
    telemetry = SlotTelemetry(capacity=1, active=0, available=1, known=False)

    assert admitted_active_count(1, telemetry) == 1
