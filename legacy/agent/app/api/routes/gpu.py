"""GPU info endpoints."""

import subprocess

from fastapi import APIRouter

from app.core.config import settings
from app.services.gpu_monitor import _sample_gpu, aggregate_gpu_info

router = APIRouter(prefix="/gpu", tags=["gpu"])


@router.get("")
async def get_gpu_info() -> dict:
    """Get GPU information."""
    gpu_info = aggregate_gpu_info(_sample_gpu(), settings.GPU_BACKEND)

    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and "Metal" in result.stdout:
            gpu_info["backend"] = "metal"
    except Exception:
        pass

    return gpu_info


@router.get("/usage")
async def get_gpu_usage() -> dict:
    """Get GPU usage statistics."""
    usage = {
        "gpu_percent": 0.0,
        "memory_percent": 0.0,
        "temperature": None,
    }

    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            usage["gpu_percent"] = float(parts[0])
            usage["memory_percent"] = float(parts[1])
            usage["temperature"] = float(parts[2])
    except Exception:
        pass

    sample = _sample_gpu()
    if sample and sample["gpus"]:
        gpus = sample["gpus"]
        total = sum(gpu["vram_total"] for gpu in gpus)
        used = sum(gpu["vram_used"] for gpu in gpus)
        usage["gpu_percent"] = sum(gpu["utilization"] for gpu in gpus) / len(gpus)
        usage["memory_percent"] = used / total * 100 if total else 0.0
        temperatures = [gpu["temperature"] for gpu in gpus if gpu["temperature"]]
        usage["temperature"] = max(temperatures, default=None)

    return usage
