"""Validation tests for the server settings exposed by the UI."""

import pytest
from pydantic import ValidationError

from app.services.server_options import ServerOptions


def test_valid_core_options() -> None:
    options = ServerOptions(
        threads=8,
        batch_size=1024,
        cache_prompt=True,
        cache_reuse=128,
        top_p=0.9,
    )

    assert options.batch_size == 1024


@pytest.mark.parametrize(
    "values",
    [
        {"cache_prompt": False, "cache_reuse": 1},
        {"fit_target": "1024", "fit": "off"},
        {"tensor_split": "3,1", "split_mode": "layer"},
    ],
)
def test_dependencies_are_rejected(values: dict) -> None:
    with pytest.raises(ValidationError):
        ServerOptions.model_validate(values)
