"""Phase 17 slice 2: mock hardware report is assignment-aware.

The mock keeps a single fake GPU (the two-GPU split is a later slice); these
tests assert the report is unchanged by default and filters to the assigned
subset when ``ASSIGNED_GPU_UUIDS`` is set.
"""

from pathlib import Path
from typing import Any

from provider_lib.config import ProviderSettings

from provider_mock.main import FAKE_HARDWARE, build_hardware_report

GIB = 1024**3


def _settings(tmp_path: Path, **kw: Any) -> ProviderSettings:
    base: dict[str, Any] = {
        "MACHINE_UID": "mock-test",
        "MACHINE_SECRET": "tok",
        "AGENT_ID": "mock-agent",
        "ADMIN_BASE_URL": "http://localhost:9999",
        "CACHE_DIR": tmp_path / "cache",
        "MODELS_DIR": tmp_path / "models",
    }
    base.update(kw)
    return ProviderSettings(**base)  # type: ignore[arg-type]


def test_default_report_unchanged(tmp_path: Path) -> None:
    report = build_hardware_report(_settings(tmp_path))
    assert report["gpus"] == FAKE_HARDWARE["gpus"]
    assert report["total_vram_bytes"] == 24 * GIB


def test_report_keeps_assigned_gpu(tmp_path: Path) -> None:
    report = build_hardware_report(_settings(tmp_path, ASSIGNED_GPU_UUIDS="mock-gpu-1"))
    assert [g["uuid"] for g in report["gpus"]] == ["mock-gpu-1"]
    assert report["total_vram_bytes"] == 24 * GIB


def test_report_drops_unassigned_gpu(tmp_path: Path) -> None:
    report = build_hardware_report(_settings(tmp_path, ASSIGNED_GPU_UUIDS="gpu-9"))
    assert report["gpus"] == []
    assert report["total_vram_bytes"] == 0
