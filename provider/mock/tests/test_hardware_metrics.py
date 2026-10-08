"""Phase 17: mock hardware report is assignment-aware (two fake GPUs).

The mock advertises two 24 GiB fake GPUs (48 GiB total) so a device-isolated
two-agent split can be demonstrated with no real hardware. These tests assert
the report shows both GPUs by default and filters to the assigned subset when
``ASSIGNED_GPU_UUIDS`` is set.
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


def test_fake_inventory_has_two_gpus() -> None:
    assert [g["uuid"] for g in FAKE_HARDWARE["gpus"]] == ["mock-gpu-1", "mock-gpu-2"]
    assert FAKE_HARDWARE["total_vram_bytes"] == 48 * GIB


def test_default_report_both_gpus(tmp_path: Path) -> None:
    report = build_hardware_report(_settings(tmp_path))
    assert report["gpus"] == FAKE_HARDWARE["gpus"]
    assert len(report["gpus"]) == 2
    assert report["total_vram_bytes"] == 48 * GIB


def test_report_filters_to_assigned_gpu(tmp_path: Path) -> None:
    report = build_hardware_report(_settings(tmp_path, ASSIGNED_GPU_UUIDS="mock-gpu-1"))
    assert [g["uuid"] for g in report["gpus"]] == ["mock-gpu-1"]
    assert report["total_vram_bytes"] == 24 * GIB


def test_report_filters_to_second_gpu(tmp_path: Path) -> None:
    report = build_hardware_report(_settings(tmp_path, ASSIGNED_GPU_UUIDS="mock-gpu-2"))
    assert [g["uuid"] for g in report["gpus"]] == ["mock-gpu-2"]
    assert report["total_vram_bytes"] == 24 * GIB


def test_report_drops_unassigned_gpu(tmp_path: Path) -> None:
    report = build_hardware_report(_settings(tmp_path, ASSIGNED_GPU_UUIDS="gpu-9"))
    assert report["gpus"] == []
    assert report["total_vram_bytes"] == 0
