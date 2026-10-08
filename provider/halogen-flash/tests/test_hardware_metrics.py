"""Phase 17 slice 2: assignment-aware hardware report + split emitter wiring."""

import os
import stat
from pathlib import Path
from typing import Any

import pytest
from conftest import make_settings
from provider_lib.admin_client import AdminClient
from provider_lib.metrics import MachineMetricsEmitter
from provider_lib.wire import Frame

from provider_halogen_flash.main import (
    build_hardware_report,
    install_command_handlers,
    make_lifecycle,
)

FAKE_NVIDIA_CSV = (
    "0, GPU-abc, NVIDIA GeForce RTX 4090, 24576, 1234, 57, 61\n"
    "1, GPU-def, NVIDIA GeForce RTX 3090, 12288, 512, 12, 45\n"
)
MIB = 1024 * 1024


@pytest.fixture
def fake_nvidia_smi(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{FAKE_NVIDIA_CSV}EOF\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old}"
    yield bin_dir
    os.environ["PATH"] = old


async def test_hardware_report_default_reports_all_gpus(
    tmp_path,
    fake_nvidia_smi,  # noqa: ARG001
) -> None:
    settings = make_settings(tmp_path, "dummy-binary")
    report = await build_hardware_report(settings)
    assert [g["uuid"] for g in report["gpus"]] == ["GPU-abc", "GPU-def"]
    assert report["total_vram_bytes"] == (24576 + 12288) * MIB


async def test_hardware_report_filters_by_assigned_uuid(
    tmp_path,
    fake_nvidia_smi,  # noqa: ARG001
) -> None:
    settings = make_settings(tmp_path, "dummy-binary", ASSIGNED_GPU_UUIDS="GPU-abc")
    report = await build_hardware_report(settings)
    assert [g["uuid"] for g in report["gpus"]] == ["GPU-abc"]
    assert report["total_vram_bytes"] == 24576 * MIB


async def test_hardware_report_filters_by_index(
    tmp_path,
    fake_nvidia_smi,  # noqa: ARG001
) -> None:
    settings = make_settings(tmp_path, "dummy-binary", ASSIGNED_GPU_UUIDS="1")
    report = await build_hardware_report(settings)
    assert [g["uuid"] for g in report["gpus"]] == ["GPU-def"]
    assert report["total_vram_bytes"] == 12288 * MIB


async def test_metrics_assign_toggles_ownership_without_stopping(
    tmp_path,
    fake_nvidia_smi,  # noqa: ARG001
) -> None:
    settings = make_settings(tmp_path, "dummy-binary")
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client, {})
    emitter = MachineMetricsEmitter(lifecycle, client, settings)
    install_command_handlers(client, lifecycle, emitter)

    calls: dict[str, list[int]] = {"start": [], "stop": []}
    emitter.start = lambda: calls["start"].append(1)  # type: ignore[method-assign]

    async def fake_stop() -> None:
        calls["stop"].append(1)

    emitter.stop = fake_stop  # type: ignore[method-assign]

    assign = client._command_handlers["metrics.assign"]
    unassign = client._command_handlers["metrics.unassign"]

    ack: dict[str, Any] = await assign(
        Frame(type="metrics.assign", id="a1", epoch=1, payload={})
    )
    assert ack["ok"] is True and ack["detail"]["emitting"] is True
    assert emitter.owned is True
    assert calls["start"]

    ack = await unassign(Frame(type="metrics.unassign", id="a2", epoch=1, payload={}))
    assert ack["ok"] is True and ack["detail"]["emitting"] is False
    # GPU categories keep flowing: unassign drops ownership but never stops.
    assert emitter.owned is False
    assert not calls["stop"]
