from types import SimpleNamespace

from app.services.inference_scheduler import (
    _vram_requirements_fit,
    server_capacity,
)


def test_server_capacity_uses_configured_parallel():
    server = SimpleNamespace(
        engine="llamacpp", engine_options={}, server_options={"parallel": 8}
    )

    assert server_capacity(server) == 8


def test_flash_uses_kv_slots_for_capacity():
    server = SimpleNamespace(
        engine="halogen-flash",
        engine_options={"kv_slots": 2},
        server_options={},
    )
    assert server_capacity(server) == 2


def test_vram_admission_uses_configured_requirements():
    assert not _vram_requirements_fit(113, 30, 110)
    assert _vram_requirements_fit(113, 30, 0)


def test_default_server_capacity_is_four():
    server = SimpleNamespace(engine="llamacpp", engine_options={}, server_options={})

    assert server_capacity(server) == 4
