"""Metrics collectors + emitter tests (fake nvidia-smi, category gating)."""

import asyncio
import os
import stat
from pathlib import Path

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
        PROVIDER_REGISTRATION_TOKEN="t",
        ADMIN_BASE_URL="http://x",
        CACHE_DIR=tmp_path,
        MODELS_DIR=tmp_path,
        METRICS_CATEGORIES="cpu os_ram",
        MACHINE_METRICS_INTERVAL=0.02,
    )
    client = FakeClient()
    emitter = metrics.MachineMetricsEmitter(FakeLifecycle(), client, settings)
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
