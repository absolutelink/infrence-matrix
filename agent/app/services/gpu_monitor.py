"""GPU monitoring service."""

import asyncio
import subprocess
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event


def _query_nvidia_smi(query: str) -> str | None:
    """Run an nvidia-smi query, returning stdout or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (OSError, subprocess.SubprocessError):
        return None


def _sample_gpu() -> dict | None:
    """Sample GPU metrics from NVIDIA or AMD device telemetry."""
    output = _query_nvidia_smi(
        "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu"
    )
    if output:
        return _parse_nvidia_output(output)
    return _sample_amd_sysfs()


def _parse_nvidia_output(output: str) -> dict | None:
    """Parse nvidia-smi CSV output."""
    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                {
                    "id": int(parts[0]),
                    "name": parts[1],
                    "vram_total": int(float(parts[2])) * 1024 * 1024,
                    "vram_used": int(float(parts[3])) * 1024 * 1024,
                    "vram_free": int(float(parts[2])) * 1024 * 1024
                    - int(float(parts[3])) * 1024 * 1024,
                    "utilization": float(parts[4]),
                    "temperature": float(parts[5]),
                    "backend": "cuda",
                }
            )
        except (ValueError, IndexError):
            continue

    return {"gpus": gpus} if gpus else None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _sample_amd_sysfs() -> dict | None:
    """Read AMDGPU shared-memory metrics exposed by the kernel driver."""
    for device in sorted(Path("/sys/class/drm").glob("card*/device")):
        try:
            if device.joinpath("vendor").read_text().strip().lower() != "0x1002":
                continue
        except OSError:
            continue

        vram_total = _read_int(device / "mem_info_vram_total") or 0
        vram_used = _read_int(device / "mem_info_vram_used") or 0
        gtt_total = _read_int(device / "mem_info_gtt_total") or 0
        gtt_used = _read_int(device / "mem_info_gtt_used") or 0
        shared_memory = gtt_total > vram_total * 2
        total = gtt_total if shared_memory else vram_total
        used = gtt_used if shared_memory else vram_used
        if total <= 0:
            continue

        busy = _read_int(device / "gpu_busy_percent") or 0
        return {
            "gpus": [
                {
                    "id": 0,
                    "name": "AMD GPU",
                    "vram_total": total,
                    "vram_used": used,
                    "vram_free": max(total - used, 0),
                    "utilization": float(busy),
                    "temperature": None,
                    "backend": settings.GPU_BACKEND,
                    "memory_type": "shared" if shared_memory else "dedicated",
                }
            ]
        }
    return None


class GPUMonitor:
    """Monitors GPU usage and health."""

    def __init__(self) -> None:
        self.backend = settings.GPU_BACKEND

    async def get_gpu_info(self) -> dict:
        """Get GPU information."""
        info = _sample_gpu()
        if info:
            return info

        logger.debug("No supported GPU telemetry source detected")
        return {"gpus": []}

    async def get_vram_usage(self) -> int:
        """Get current VRAM usage in bytes."""
        info = _sample_gpu()
        if info and info["gpus"]:
            return sum(g["vram_used"] for g in info["gpus"])
        return 0

    async def get_utilization(self) -> float:
        """Get GPU utilization percentage."""
        info = _sample_gpu()
        if info and info["gpus"]:
            return sum(g["utilization"] for g in info["gpus"]) / len(info["gpus"])
        return 0.0


async def _gpu_usage_loop() -> None:
    """Periodically emit gpu.usage events."""
    while True:
        try:
            sample = _sample_gpu()
            if sample:
                total = sum(g["vram_total"] for g in sample["gpus"])
                used = sum(g["vram_used"] for g in sample["gpus"])
                utilization = sum(g["utilization"] for g in sample["gpus"]) / len(
                    sample["gpus"]
                )
                publish_event(
                    "gpu.usage",
                    {
                        "vram_used": used,
                        "vram_free": total - used,
                        "vram_total": total,
                        "utilization": utilization,
                        "gpus": sample["gpus"],
                    },
                )
        except (OSError, ValueError, KeyError, IndexError) as e:
            logger.debug(f"GPU usage sampling failed: {e}")
        await asyncio.sleep(settings.GPU_USAGE_INTERVAL)


def start_gpu_monitoring() -> None:
    """Start the background GPU usage emission task."""
    asyncio.create_task(_gpu_usage_loop())


gpu_monitor = GPUMonitor()
