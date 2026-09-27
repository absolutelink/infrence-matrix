"""Tests for aggregate GPU reporting."""

import os
from unittest.mock import patch

import pytest

os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"
os.environ["MODELS_PATH"] = "/tmp/models"

from app.api.routes.gpu import get_gpu_info
from app.services.frontend_client import FrontendClient
from app.services.gpu_monitor import _parse_nvidia_output, aggregate_gpu_info

SAMPLE = {
    "gpus": [
        {
            "id": 0,
            "name": "GPU 0",
            "vram_total": 10,
            "vram_used": 4,
            "vram_free": 6,
            "utilization": 20.0,
            "temperature": 50.0,
            "backend": "cuda",
        },
        {
            "id": 1,
            "name": "GPU 1",
            "vram_total": 20,
            "vram_used": 5,
            "vram_free": 15,
            "utilization": 40.0,
            "temperature": 60.0,
            "backend": "cuda",
        },
    ]
}


def test_aggregate_gpu_info_keeps_legacy_fields_and_totals_all_devices():
    info = aggregate_gpu_info(SAMPLE, "auto")

    assert info["name"] == "GPU 0"
    assert info["vram_total"] == 30
    assert info["vram_used"] == 9
    assert info["vram_free"] == 21
    assert info["gpu_count"] == 2
    assert info["gpus"] == SAMPLE["gpus"]


def test_nvidia_parser_preserves_hardware_uuid():
    info = _parse_nvidia_output(
        "0, GPU-abc, NVIDIA Test, 1024, 128, 25, 45"
    )

    assert info is not None
    assert info["gpus"][0]["uuid"] == "GPU-abc"
    assert info["gpus"][0]["vram_total"] == 1024 * 1024 * 1024


@pytest.mark.asyncio
async def test_gpu_endpoint_and_registration_share_aggregate_shape():
    with (
        patch("app.api.routes.gpu._sample_gpu", return_value=SAMPLE),
        patch("app.api.routes.gpu.subprocess.run", side_effect=OSError),
        patch("app.services.frontend_client._sample_gpu", return_value=SAMPLE),
    ):
        endpoint_info = await get_gpu_info()
        registration_info = await FrontendClient()._get_gpu_info()

    assert endpoint_info["vram_total"] == 30
    assert registration_info == endpoint_info
