"""Platform registration and compatibility schema tests."""

import pytest
from pydantic import ValidationError

from app.api.routes.agents import AgentRegisterRequest
from app.api.routes.server_instances import StartServerRequest


def test_flash_platform_is_registered() -> None:
    request = AgentRegisterRequest(
        agent_id="flash-agent",
        name="flash-agent",
        platform="halogen-flash",
        type="rocm",
        host="127.0.0.1",
        port=8080,
    )
    assert request.platform == "halogen-flash"


def test_unknown_platform_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentRegisterRequest(
            agent_id="bad-agent",
            name="bad-agent",
            platform="halogen-server",
            host="127.0.0.1",
            port=8080,
        )


def test_flash_engine_is_accepted() -> None:
    request = StartServerRequest(alias="flash", engine="halogen-flash")
    assert request.engine == "halogen-flash"
