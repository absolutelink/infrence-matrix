"""GPU info endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/gpu", tags=["gpu"])


@router.get("")
async def get_gpu_info() -> dict:
    """Get GPU information."""
    gpu_info = {
        "name": "Unknown",
        "vram_total": 0,
        "vram_used": 0,
        "backend": "auto",
    }

    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(", ")
                gpu_info["name"] = parts[0]
                gpu_info["vram_total"] = int(parts[1]) * 1024 * 1024
                gpu_info["vram_used"] = int(parts[2]) * 1024 * 1024
                gpu_info["backend"] = "cuda"
    except Exception:
        pass

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

    return usage
