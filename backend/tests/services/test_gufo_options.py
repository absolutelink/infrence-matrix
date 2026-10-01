"""Gufo settings validation tests."""

import pytest
from pydantic import ValidationError

from app.api.routes.agents import AgentRegisterRequest
from app.api.routes.server_instances import StartServerRequest
from app.services.server_options import validate_gufo_options


def test_gufo_platform_is_registered() -> None:
    request = AgentRegisterRequest(
        agent_id="gufo-agent",
        name="gufo-agent",
        platform="gufo",
        type="gufo",
        host="127.0.0.1",
        port=8080,
    )
    assert request.platform == "gufo"


def test_gufo_engine_is_accepted() -> None:
    request = StartServerRequest(alias="gufo", engine="gufo")
    assert request.engine == "gufo"


def test_gufo_options_omit_none_and_keep_set_values() -> None:
    assert validate_gufo_options(
        {
            "context": 262144,
            "temperature": 0.7,
            "think": "on",
            "reasoning_effort": "high",
            "preserve_thinking": "auto",
            "speculative": "dflash2",
            "dflash_model": "/models/draft.gguf",
            "draft_policy": "adaptive",
            "draft_tokens": 7,
            "sessions": 4,
            "cache_disk": True,
            "api_key": None,
        }
    ) == {
        "context": 262144,
        "temperature": 0.7,
        "think": "on",
        "reasoning_effort": "high",
        "preserve_thinking": "auto",
        "speculative": "dflash2",
        "dflash_model": "/models/draft.gguf",
        "draft_policy": "adaptive",
        "draft_tokens": 7,
        "sessions": 4,
        "cache_disk": True,
    }


def test_gufo_cache_disk_rejects_directory_string() -> None:
    # cache_disk is now a boolean enable flag; the agent derives the
    # directory from its own CACHE_PATH, so paths must not be accepted.
    with pytest.raises(ValidationError):
        validate_gufo_options({"cache_disk": "/cache/srv-1"})


def test_gufo_options_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        validate_gufo_options({"parallel": 4})
    with pytest.raises(ValidationError):
        validate_gufo_options({"served_model_name": "x"})


def test_gufo_options_reject_invalid_enum_values() -> None:
    with pytest.raises(ValidationError):
        validate_gufo_options({"think": "sometimes"})
    with pytest.raises(ValidationError):
        validate_gufo_options({"speculative": "magic"})
    with pytest.raises(ValidationError):
        validate_gufo_options({"reasoning_effort": "ultra"})


def test_gufo_options_reject_effort_with_think_off() -> None:
    with pytest.raises(ValidationError):
        validate_gufo_options({"think": "off", "reasoning_effort": "high"})


def test_gufo_options_allow_effort_with_think_auto_or_on() -> None:
    assert validate_gufo_options({"think": "auto", "reasoning_effort": "max"}) == {
        "think": "auto",
        "reasoning_effort": "max",
    }
    assert validate_gufo_options({"think": "on", "reasoning_effort": "low"}) == {
        "think": "on",
        "reasoning_effort": "low",
    }


def test_gufo_options_enforce_numeric_bounds() -> None:
    with pytest.raises(ValidationError):
        validate_gufo_options({"top_p": 1.5})
    with pytest.raises(ValidationError):
        validate_gufo_options({"sessions": 0})
    with pytest.raises(ValidationError):
        validate_gufo_options({"draft_tokens": 0})
    with pytest.raises(ValidationError):
        validate_gufo_options({"context": -1})
