"""Metrics collectors + emitter tests (fake nvidia-smi, category gating)."""

import asyncio
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from provider_lib import metrics
from provider_lib.config import ProviderSettings

FAKE_NVIDIA_CSV = (
    "0, GPU-abc, NVIDIA GeForce RTX 4090, 24576, 1234, 57, 61\n"
    "1, GPU-def, NVIDIA GeForce RTX 3090, 24576, 512, 12, 45\n"
)

FAKE_NVIDIA_SCRIPT = f"""#!/bin/sh
cat <<'EOF'
{FAKE_NVIDIA_CSV}EOF
"""


@pytest.fixture
def fake_nvidia_smi(tmp_path: Path):
    """Put a fake nvidia-smi emitting CSV on PATH for this process."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text(FAKE_NVIDIA_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
    yield bin_dir
    os.environ["PATH"] = old_path


async def test_collect_vram_parses_nvidia_smi(fake_nvidia_smi) -> None:  # noqa: ARG001
    vram = await metrics.collect_vram()
    assert vram["gpu_count"] == 2
    assert vram["total_bytes"] == (24576 + 24576) * 1024 * 1024
    assert vram["used_bytes"] == (1234 + 512) * 1024 * 1024
    assert vram["free_bytes"] == vram["total_bytes"] - vram["used_bytes"]
    assert vram["gpus"][0]["uuid"] == "GPU-abc"
    assert vram["gpus"][0]["name"] == "NVIDIA GeForce RTX 4090"
    assert vram["gpus"][1]["vram_used"] == 512 * 1024 * 1024


async def test_collect_gpu_usage_from_same_sampler(fake_nvidia_smi) -> None:  # noqa: ARG001
    usage = await metrics.collect_gpu_usage()
    assert usage["utilization_percent"] == round((57 + 12) / 2, 2)


async def test_category_gating_only_requested_keys(
    fake_nvidia_smi,  # noqa: ARG001
    tmp_path,
) -> None:
    snap = await metrics.collect_machine_snapshot(
        {"cpu"}, models_dir=tmp_path, cache_dir=tmp_path
    )
    assert set(snap) == {"cpu"}
    snap = await metrics.collect_machine_snapshot(
        {"vram", "os_ram"}, models_dir=tmp_path, cache_dir=tmp_path
    )
    assert set(snap) == {"vram", "os_ram"}
    snap = await metrics.collect_machine_snapshot(
        set(), models_dir=tmp_path, cache_dir=tmp_path
    )
    assert snap == {}


async def test_collector_failure_omits_key_no_raise(
    fake_nvidia_smi,  # noqa: ARG001
    tmp_path,
    monkeypatch,
) -> None:
    # No nvidia-smi on PATH (and no AMD sysfs in most CI envs) -> vram
    # omitted. Even if the host HAS a GPU, force the sampler to fail.
    async def failing_sample():
        raise RuntimeError("no gpu here")

    monkeypatch.setattr(metrics, "collect_gpu_sample", failing_sample)
    snap = await metrics.collect_machine_snapshot(
        {"vram", "gpu_usage", "cpu", "os_ram", "storage", "bogus"},
        models_dir=tmp_path,
        cache_dir=tmp_path,
    )
    assert "vram" not in snap
    assert "gpu_usage" not in snap
    assert "bogus" not in snap
    # These work on any Linux-ish host; if one is unavailable it's simply
    # omitted, but the call itself must not raise.
    assert isinstance(snap, dict)


async def test_os_ram_cpu_storage_work_in_this_env(tmp_path) -> None:
    ram = await metrics.collect_os_ram()
    assert ram["total_bytes"] > 0
    assert ram["available_bytes"] > 0
    cpu = await metrics.collect_cpu()
    assert cpu["cores"] >= 1
    assert cpu["load_1m"] >= 0
    storage = await metrics.collect_storage(tmp_path, tmp_path)
    assert storage["models"]["total_bytes"] > 0
    assert storage["cache"]["free_bytes"] >= 0


async def test_no_gpu_source_raises_lookup(
    fake_nvidia_smi,  # noqa: ARG001
    monkeypatch,
) -> None:
    monkeypatch.setattr(metrics, "_sample_gpus_blocking", lambda: [])
    with pytest.raises(LookupError):
        await metrics.collect_vram()  # direct collector raises; snapshot gates
    snap = await metrics.collect_machine_snapshot({"vram"})
    assert snap == {}


async def test_collect_vram_parses_via_thread(fake_nvidia_smi) -> None:  # noqa: ARG001
    # Sanity: the subprocess call itself works under asyncio.to_thread.
    out = await asyncio.to_thread(metrics._query_nvidia_smi, metrics._NVIDIA_QUERY)
    assert out and "RTX 4090" in out


class FakeClient:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def send_event(self, type_: str, payload: dict) -> None:
        self.events.append((type_, payload))


class FakeLifecycle:
    backend_status = "running"
    in_flight = 0
    capacity = 1


async def test_emitter_runs_only_while_started(tmp_path) -> None:
    settings = ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="m1",
        MACHINE_SECRET="t",
        AGENT_ID="m1-agent",
        ADMIN_BASE_URL="http://x",
        CACHE_DIR=tmp_path,
        MODELS_DIR=tmp_path,
        METRICS_CATEGORIES="cpu os_ram",
        MACHINE_METRICS_INTERVAL=0.02,
    )
    client = FakeClient()
    emitter = metrics.MachineMetricsEmitter(FakeLifecycle(), client, settings)
    # Machine-wide categories are owner-gated (Phase 17): own the machine so
    # cpu/os_ram are actually included in each frame.
    emitter.set_owned(True)
    assert not emitter.running
    emitter.start()
    assert emitter.running
    # Wait for a few emissions.
    for _ in range(200):
        if len(client.events) >= 2:
            break
        await asyncio.sleep(0.01)
    assert len(client.events) >= 2
    type_, payload = client.events[0]
    assert type_ == "metrics.machine"
    assert payload["backend_status"] == "running"
    assert payload["in_flight"] == 0
    assert payload["machine_uid"] == "m1"
    assert set(payload) >= {"cpu", "os_ram"}
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


# ----------------------------------------------------------------------
# Phase 17: GPU scoping + split emitter
# ----------------------------------------------------------------------


def test_parse_gpu_assignment_normalizes() -> None:
    assert metrics.parse_gpu_assignment([]) == set()
    assert metrics.parse_gpu_assignment(["", "  "]) == set()
    assert metrics.parse_gpu_assignment(["GPU-ABC", "1"]) == {"gpu-abc", "1"}


def test_filter_gpus_implicit_passthrough() -> None:
    sample = [{"id": 0, "uuid": "GPU-abc"}, {"id": 1, "uuid": "GPU-def"}]
    # Empty assignment = report every GPU unchanged.
    assert metrics.filter_gpus(sample, set()) == sample


def test_filter_gpus_uuid_match_case_insensitive() -> None:
    sample = [{"id": 0, "uuid": "GPU-abc"}, {"id": 1, "uuid": "GPU-def"}]
    kept = metrics.filter_gpus(sample, metrics.parse_gpu_assignment(["gpu-ABC"]))
    assert [g["uuid"] for g in kept] == ["GPU-abc"]


def test_filter_gpus_index_match() -> None:
    sample = [{"id": 0, "uuid": "GPU-abc"}, {"id": 1, "uuid": "GPU-def"}]
    kept = metrics.filter_gpus(sample, metrics.parse_gpu_assignment(["1"]))
    assert [g["uuid"] for g in kept] == ["GPU-def"]


def test_filter_gpus_nonmatch_excluded() -> None:
    sample = [{"id": 0, "uuid": "GPU-abc"}, {"id": 1, "uuid": "GPU-def"}]
    assert metrics.filter_gpus(sample, metrics.parse_gpu_assignment(["9"])) == []


def test_gpu_uuid_helper() -> None:
    # Real uuid wins; otherwise the synthesized gpu-<id> fallback (the same
    # string providers report and operators pass back as a token).
    assert metrics.gpu_uuid({"uuid": "GPU-abc", "id": 0}) == "GPU-abc"
    assert metrics.gpu_uuid({"id": 3}) == "gpu-3"
    assert metrics.gpu_uuid({"uuid": "", "id": 1}) == "gpu-1"


def test_filter_gpus_accepts_synthesized_uuid_token() -> None:
    # A GPU with no real uuid reports the synthesized gpu-<id> form; that token
    # must match it (reported uuid and accepted token share one space).
    sample = [{"id": 0, "name": "amd"}, {"id": 1, "name": "amd"}]
    kept = metrics.filter_gpus(sample, metrics.parse_gpu_assignment(["gpu-1"]))
    assert [g["id"] for g in kept] == [1]


def test_filter_gpus_accepts_real_uuid_and_decimal_index() -> None:
    sample = [{"id": 0, "uuid": "GPU-abc"}, {"id": 1, "uuid": "GPU-def"}]
    assert [g["uuid"] for g in metrics.filter_gpus(sample, {"gpu-abc"})] == ["GPU-abc"]
    assert [g["uuid"] for g in metrics.filter_gpus(sample, {"1"})] == ["GPU-def"]


def test_amd_uuid_prefers_pci_slot(tmp_path) -> None:
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "uevent").write_text(
        "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0\nMODALIAS=pci:v00001002\n"
    )
    assert metrics._amd_uuid(device) == "amd-0000:03:00.0"


def test_amd_uuid_falls_back_to_card_dir_name(tmp_path) -> None:
    a = tmp_path / "card1" / "device"
    b = tmp_path / "card2" / "device"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    # No uevent: distinct cardN names → distinct uuids, never the
    # container-local index (which is 0 in every device-isolated container).
    assert metrics._amd_uuid(a) == "amd-card1"
    assert metrics._amd_uuid(b) == "amd-card2"


def test_amd_uuid_distinct_across_isolated_containers(tmp_path) -> None:
    # Two device-isolated containers each enumerate their single card as
    # card0 (id=0) but on distinct PCI slots → distinct uuids, so the
    # union-by-uuid merge keeps them separate (the collision this fixes).
    a = tmp_path / "boxA" / "card0" / "device"
    b = tmp_path / "boxB" / "card0" / "device"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "uevent").write_text("PCI_SLOT_NAME=0000:03:00.0\n")
    (b / "uevent").write_text("PCI_SLOT_NAME=0000:04:00.0\n")
    assert metrics._amd_uuid(a) == "amd-0000:03:00.0"
    assert metrics._amd_uuid(b) == "amd-0000:04:00.0"
    assert metrics._amd_uuid(a) != metrics._amd_uuid(b)


async def test_collect_machine_snapshot_filters_gpus(
    fake_nvidia_smi,  # noqa: ARG001
) -> None:
    assignment = metrics.parse_gpu_assignment(["GPU-abc"])
    snap = await metrics.collect_machine_snapshot(
        {"vram", "gpu_usage"}, gpu_assignment=assignment
    )
    assert snap["vram"]["gpu_count"] == 1
    assert snap["vram"]["gpus"][0]["uuid"] == "GPU-abc"
    assert [g["id"] for g in snap["gpu_usage"]["gpus"]] == [0]


async def test_collect_vram_respects_assignment(
    fake_nvidia_smi,  # noqa: ARG001
) -> None:
    assignment = metrics.parse_gpu_assignment(["1"])
    vram = await metrics.collect_vram(assignment)
    assert vram["gpu_count"] == 1
    assert vram["gpus"][0]["uuid"] == "GPU-def"


def _emitter_settings(tmp_path, categories: str, **kw) -> ProviderSettings:
    base: dict[str, Any] = {
        "MACHINE_UID": "m1",
        "MACHINE_SECRET": "t",
        "AGENT_ID": "m1-agent",
        "ADMIN_BASE_URL": "http://x",
        "CACHE_DIR": tmp_path,
        "MODELS_DIR": tmp_path,
        "METRICS_CATEGORIES": categories,
        "MACHINE_METRICS_INTERVAL": 0.02,
    }
    base.update(kw)
    return ProviderSettings(**base)  # type: ignore[arg-type]


async def test_emitter_gpu_categories_emit_without_ownership(
    tmp_path,
    fake_nvidia_smi,  # noqa: ARG001
) -> None:
    settings = _emitter_settings(tmp_path, "vram gpu_usage")
    client = FakeClient()
    emitter = metrics.MachineMetricsEmitter(FakeLifecycle(), client, settings)
    assert emitter.owned is False
    payload = await emitter.emit_once()
    assert payload is not None
    assert set(payload) >= {"vram", "gpu_usage"}
    assert payload["vram"]["gpu_count"] == 2
    # GPU categories are NOT owner-gated: emitted even though not owned.
    assert "cpu" not in payload and "os_ram" not in payload
    assert payload["assigned_gpus"] == ["GPU-abc", "GPU-def"]
    assert client.events and client.events[0][0] == "metrics.machine"


async def test_emitter_machine_wide_gated_by_ownership(tmp_path) -> None:
    settings = _emitter_settings(tmp_path, "cpu os_ram")
    client = FakeClient()
    emitter = metrics.MachineMetricsEmitter(FakeLifecycle(), client, settings)
    # Not owned + only machine-wide cats declared → nothing to report → skip.
    assert await emitter.emit_once() is None
    assert client.events == []
    # Once owned, the machine-wide categories flow.
    emitter.set_owned(True)
    payload = await emitter.emit_once()
    assert payload is not None
    assert set(payload) >= {"cpu", "os_ram"}
    assert payload["assigned_gpus"] == []
    assert client.events and client.events[0][0] == "metrics.machine"


async def test_emitter_skip_empty_no_send(tmp_path, monkeypatch) -> None:
    # Declares only GPU categories but no GPU source is available → empty
    # snapshot → no frame sent.
    async def no_gpus():
        return []

    monkeypatch.setattr(metrics, "collect_gpu_sample", no_gpus)
    settings = _emitter_settings(tmp_path, "vram gpu_usage")
    client = FakeClient()
    emitter = metrics.MachineMetricsEmitter(FakeLifecycle(), client, settings)
    assert await emitter.emit_once() is None
    assert client.events == []


async def test_emitter_assigned_gpu_uuids_filters(
    tmp_path,
    fake_nvidia_smi,  # noqa: ARG001
) -> None:
    settings = _emitter_settings(
        tmp_path, "vram gpu_usage", ASSIGNED_GPU_UUIDS="GPU-def"
    )
    client = FakeClient()
    emitter = metrics.MachineMetricsEmitter(FakeLifecycle(), client, settings)
    payload = await emitter.emit_once()
    assert payload is not None
    assert payload["vram"]["gpu_count"] == 1
    assert payload["assigned_gpus"] == ["GPU-def"]
