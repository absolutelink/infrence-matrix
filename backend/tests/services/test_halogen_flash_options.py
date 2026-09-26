"""Halogen Flash settings validation tests."""

import pytest
from pydantic import ValidationError

from app.services.server_options import validate_halogen_flash_options


def test_flash_options_use_flash_names_and_omit_none() -> None:
    assert validate_halogen_flash_options(
        {
            "kv_pool_positions": 524288,
            "prompt_cache": 2,
            "cache_dir_enabled": True,
            "cache_disk_gib": 64,
            "vision_tower": 1,
            "temperature": None,
        }
    ) == {
        "kv_pool_positions": 524288,
        "prompt_cache": 2,
        "cache_dir_enabled": True,
        "cache_disk_gib": 64,
        "vision_tower": 1,
    }


def test_flash_options_reject_ordinary_halogen_settings() -> None:
    with pytest.raises(ValidationError):
        validate_halogen_flash_options({"cache_mb": 1024})


def test_flash_options_reject_invalid_vision_tower_value() -> None:
    with pytest.raises(ValidationError):
        validate_halogen_flash_options({"vision_tower": 2})
