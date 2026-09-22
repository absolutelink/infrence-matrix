"""GPU monitoring service."""

import asyncio
import subprocess

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
    """Sample GPU metrics via nvidia-smi. Returns None if unavailable."""
    output = _query_nvidia_smi(
        "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu"
    )
    if not output:
        return None

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


class GPUMonitor:
    """Monitors GPU usage and health."""

    def __init__(self) -> None:
        self.backend = settings.GPU_BACKEND

    async def get_gpu_info(self) -> dict:
        """Get GPU information."""
        info = _sample_gpu()
        if info:
            return info

        logger.debug("No NVIDIA GPU detected via nvidia-smi")
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
