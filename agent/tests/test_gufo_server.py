"""Tests for the Gufo process manager and platform gating."""

import os
from unittest.mock import Mock

import pytest

# Settings are instantiated at import time.
os.environ.setdefault("AGENT_ID", "test-agent")
os.environ.setdefault("FRONTEND_URL", "http://test:8000")
os.environ.setdefault("MODELS_PATH", "/tmp/models")

from app.api.routes.proxy import _is_supported_gufo_path
from app.api.routes.servers import _validate_engine_platform
from app.services.gufo_server import (
    GufoServerConfig,
    GufoServerManager,
    build_command,
)


def test_gufo_proxy_allows_documented_routes_only() -> None:
    assert _is_supported_gufo_path("v1/chat/completions")
    assert _is_supported_gufo_path("v1/completions")
    assert _is_supported_gufo_path("v1/responses")
    assert _is_supported_gufo_path("health")
    assert _is_supported_gufo_path("ready")
    assert not _is_supported_gufo_path("v1/embeddings")
    assert not _is_supported_gufo_path("slots")


def test_gufo_agent_rejects_other_engines(monkeypatch) -> None:
    monkeypatch.setattr("app.api.routes.servers.settings.AGENT_PLATFORM", "gufo")
    with pytest.raises(Exception, match="only supports engine=gufo"):
        _validate_engine_platform("llamacpp")
    with pytest.raises(Exception, match="only supports engine=gufo"):
        _validate_engine_platform("halogen-flash")


def test_gufo_agent_accepts_gufo_engine(monkeypatch) -> None:
    monkeypatch.setattr("app.api.routes.servers.settings.AGENT_PLATFORM", "gufo")
    _validate_engine_platform("gufo")


def test_build_command_always_includes_serve_llm_host_port_model() -> None:
    config = GufoServerConfig(model_path="/models/m.gguf", port=8123)
    cmd = build_command(config)
    assert cmd[:8] == [
        "gufo",
        "serve",
        "llm",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--model",
    ]
    # No optional flags when options are empty.
    assert cmd[8:] == ["/models/m.gguf"]


def test_build_command_maps_value_enum_and_bool_flags() -> None:
    config = GufoServerConfig(
        model_path="/models/m.gguf",
        port=8123,
        options={
            "context": 131072,
            "think": "on",
            "reasoning_effort": "high",
            "speculative": "dflash2",
            "dflash_model": "/models/draft.gguf",
            "draft_policy": "adaptive",
            "sessions": 4,
            "api_key": "secret",
            "log_progress": True,
            "verbose": False,
        },
    )
    cmd = build_command(config)
    joined = " ".join(cmd)
    assert "--context 131072" in joined
    assert "--think on" in joined
    assert "--reasoning-effort high" in joined
    assert "--speculative dflash2" in joined
    assert "--dflash-model /models/draft.gguf" in joined
    assert "--draft-policy adaptive" in joined
    assert "--sessions 4" in joined
    assert "--api-key secret" in joined
    assert "--log-progress" in cmd
    # False booleans are omitted so gufo keeps its own default.
    assert "--verbose" not in cmd


def test_build_command_maps_forced_served_model_name() -> None:
    # The backend always injects served_model_name = instance.alias for gufo;
    # the agent must surface it as --served-model-name so gufo advertises the
    # alias instead of the GGUF filename.
    config = GufoServerConfig(
        model_path="/models/Big-Model-Q8.gguf",
        port=8123,
        options={"served_model_name": "my-alias"},
    )
    cmd = build_command(config)
    joined = " ".join(cmd)
    assert "--served-model-name my-alias" in joined


def test_effective_capacity_uses_sessions() -> None:
    manager = GufoServerManager()
    manager.configs["s1"] = GufoServerConfig(
        model_path="/m", port=1, options={"sessions": 3}
    )
    assert manager.get_effective_capacity("s1") == 3
    manager.configs["s2"] = GufoServerConfig(model_path="/m", port=1, options={})
    assert manager.get_effective_capacity("s2") == 1


@pytest.mark.asyncio
async def test_instance_limit_rejects_extra_servers(monkeypatch) -> None:
    monkeypatch.setattr("app.services.gufo_server.settings.GUFO_MAX_INSTANCES", 1)
    manager = GufoServerManager()
    manager.servers["existing"] = Mock()
    with pytest.raises(RuntimeError, match="Gufo instance limit reached"):
        await manager._start_server_locked(
            "another", GufoServerConfig(model_path="/m", port=9)
        )
