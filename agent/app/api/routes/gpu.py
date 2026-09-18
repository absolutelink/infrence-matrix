from fastapi import APIRouter
import psutil

router = APIRouter(prefix="/gpu", tags=["gpu"])


@router.get("/info")
async def get_gpu_info() -> dict:
    """Get GPU information."""
    # TODO: Implement actual GPU detection
    # For now, return mock data
    return {
        "gpus": [{
            "id": 0,
            "name": "NVIDIA GPU",
            "vram_total": 24576000000,
            "vram_used": 8589934592,
            "vram_free": 15986065408,
            "utilization": 45,
            "backend": "cuda",
        }]
    }
