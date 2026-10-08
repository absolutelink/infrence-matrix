"""Phase 17 slice 5: mock synthetic machine-metrics emitter.

Mirrors how ``provider/lib/tests/test_metrics.py`` exercises the real
:class:`MachineMetricsEmitter` (fake client capturing ``send_event``), but
against the mock's synthetic values derived from its two fake GPUs. Asserts
the Phase 17 category split: GPU categories emit regardless of ownership
(filtered to ``ASSIGNED_GPU_UUIDS``), machine-wide categories only while owned,
and skip-empty behavior.
"""

import asyncio
from pathlib import Path
from typing import Any

from provider_lib.config import ProviderSettings

from provider_mock.metrics import MockMachineMetricsEmitter

GIB = 1024**3


class FakeClient:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def send_event(self, type_: str, payload: dict) -> None:
        self.events.append((type_, payload))


class FakeLifecycle:
    backend_status = "running"
    in_flight = 0
    capacity = 1


def _settings(tmp_path: Path, categories: str, **kw: Any) -> ProviderSettings:
    base: dict[str, Any] = {
        "MACHINE_UID": "mock-machine-1",
        "MACHINE_SECRET": "tok",
        "AGENT_ID": "mock-agent-1",
        "ADMIN_BASE_URL": "http://localhost:9999",
        "CACHE_DIR": tmp_path / "cache",
        "MODELS_DIR": tmp_path / "models",
        "METRICS_CATEGORIES": categories,
        "MACHINE_METRICS_INTERVAL": 0.02,
    }
    base.update(kw)
    return ProviderSettings(**base)  # type: ignore[arg-type]


async def test_gpu_categories_emit_without_ownership(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "vram gpu_usage")
    client = FakeClient()
    emitter = MockMachineMetricsEmitter(FakeLifecycle(), client, settings)
    assert emitter.owned is False
    payload = await emitter.emit_once()
    assert payload is not None
    assert set(payload) >= {"vram", "gpu_usage"}
    # Both fake GPUs by default (no assignment).
    assert payload["vram"]["gpu_count"] == 2
    assert payload["assigned_gpus"] == ["mock-gpu-1", "mock-gpu-2"]
    # GPU categories are NOT owner-gated: machine-wide sections absent.
    assert "cpu" not in payload and "os_ram" not in payload and "storage" not in payload
    assert client.events and client.events[0][0] == "metrics.machine"


async def test_vram_gpu_entries_carry_required_fields(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "vram")
    client = FakeClient()
    emitter = MockMachineMetricsEmitter(FakeLifecycle(), client, settings)
    payload = await emitter.emit_once()
    assert payload is not None
    for gpu in payload["vram"]["gpus"]:
        assert {"uuid", "vram_total", "vram_used", "vram_free", "utilization"} <= set(
            gpu
        )
        assert gpu["vram_free"] == gpu["vram_total"] - gpu["vram_used"]
        assert gpu["vram_used"] > 0
    # Aggregates match the union the admin recomputes on read.
    assert payload["vram"]["total_bytes"] == 48 * GIB
    assert payload["vram"]["used_bytes"] == sum(
        g["vram_used"] for g in payload["vram"]["gpus"]
    )


async def test_machine_wide_gated_by_ownership(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "cpu os_ram storage")
    client = FakeClient()
    emitter = MockMachineMetricsEmitter(FakeLifecycle(), client, settings)
    # Not owned + only machine-wide cats declared -> nothing to report -> skip.
    assert await emitter.emit_once() is None
    assert client.events == []
    emitter.set_owned(True)
    payload = await emitter.emit_once()
    assert payload is not None
    assert set(payload) >= {"cpu", "os_ram", "storage"}
    assert payload["assigned_gpus"] == []
    assert client.events and client.events[0][0] == "metrics.machine"


async def test_assigned_gpu_uuids_filters_emitted_gpus(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "vram gpu_usage", ASSIGNED_GPU_UUIDS="mock-gpu-2")
    client = FakeClient()
    emitter = MockMachineMetricsEmitter(FakeLifecycle(), client, settings)
    payload = await emitter.emit_once()
    assert payload is not None
    assert payload["vram"]["gpu_count"] == 1
    assert payload["assigned_gpus"] == ["mock-gpu-2"]
    assert payload["vram"]["gpus"][0]["uuid"] == "mock-gpu-2"
    assert payload["vram"]["total_bytes"] == 24 * GIB
    # Fidelity: a filtered report keeps the GPU's stable inventory id (1), not
    # its position in the filtered list (0) — matching a real isolated agent.
    assert payload["vram"]["gpus"][0]["id"] == 1


async def test_skip_empty_no_send(tmp_path: Path) -> None:
    # Declares only machine-wide categories but is not owned -> empty snapshot
    # -> no frame sent.
    settings = _settings(tmp_path, "cpu")
    client = FakeClient()
    emitter = MockMachineMetricsEmitter(FakeLifecycle(), client, settings)
    assert await emitter.emit_once() is None
    assert client.events == []


async def test_payload_includes_lifecycle_and_machine_uid(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "vram")
    client = FakeClient()
    emitter = MockMachineMetricsEmitter(FakeLifecycle(), client, settings)
    payload = await emitter.emit_once()
    assert payload is not None
    assert payload["backend_status"] == "running"
    assert payload["in_flight"] == 0
    assert payload["capacity"] == 1
    assert payload["machine_uid"] == "mock-machine-1"


async def test_emitter_runs_only_while_started(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "vram gpu_usage")
    client = FakeClient()
    emitter = MockMachineMetricsEmitter(FakeLifecycle(), client, settings)
    assert not emitter.running
    emitter.start()
    assert emitter.running
    for _ in range(200):
        if len(client.events) >= 2:
            break
        await asyncio.sleep(0.01)
    assert len(client.events) >= 2
    await emitter.stop()
    assert not emitter.running
    count = len(client.events)
    await asyncio.sleep(0.1)
    assert len(client.events) == count  # no events after stop
    # start/stop idempotent
    emitter.start()
    emitter.start()
    await emitter.stop()
    await emitter.stop()
