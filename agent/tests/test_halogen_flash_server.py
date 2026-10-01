"""Tests for the dedicated Halogen Flash process manager."""

import os
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

# Settings are instantiated at import time.
os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"
os.environ["MODELS_PATH"] = "/tmp/models"

from app.api.routes.proxy import _is_supported_halogen_flash_path
from app.api.routes.servers import _validate_engine_platform
from app.services.halogen_flash_server import (
    HALOGEN_FLASH_CHECKPOINT,
    HALOGEN_FLASH_REPO_ID,
    HALOGEN_FLASH_TOKENIZER,
    HalogenFlashServerConfig,
    HalogenFlashServerManager,
)
from app.services.halogen_server import HalogenServerConfig, HalogenServerManager


def test_flash_proxy_allows_documented_routes_only() -> None:
    assert _is_supported_halogen_flash_path("v1/chat/completions")
    assert _is_supported_halogen_flash_path("v1/completions")
    assert not _is_supported_halogen_flash_path("v1/responses")
    assert not _is_supported_halogen_flash_path("v1/embeddings")


def test_flash_agent_rejects_other_engines(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.servers.settings.AGENT_PLATFORM", "halogen-flash"
    )
    with pytest.raises(Exception, match="only supports engine=halogen-flash"):
        _validate_engine_platform("halogen")


@pytest.mark.asyncio
async def test_halogen_health_requires_consecutive_failures():
    manager = HalogenServerManager()
    manager.servers["server-1"] = Mock(poll=Mock(return_value=None))
    manager.configs["server-1"] = HalogenServerConfig(
        model_path="/models/test",
        port=8091,
        api_port=8091,
        engine_port=8092,
        options={},
    )
    manager.server_health["server-1"] = "healthy"
    manager.healthy_servers.add("server-1")

    with patch.object(
        manager, "_check_health", new=AsyncMock(return_value=(False, "not ready"))
    ):
        with patch("app.services.halogen_server.publish_event") as publish:
            await manager._monitor("server-1")
            await manager._monitor("server-1")
            assert manager.server_health["server-1"] == "healthy"
            publish.assert_not_called()

            await manager._monitor("server-1")

    assert manager.server_health["server-1"] == "unhealthy"
    publish.assert_called_once_with(
        "server.health",
        {
            "server_id": "server-1",
            "status": "unhealthy",
            "error": "not ready",
            "slot_generation": 0,
        },
    )


@pytest.mark.asyncio
@patch("app.services.halogen_flash_server.subprocess.Popen")
async def test_flash_start_uses_flash_environment_and_two_ports(mock_popen):
    manager = HalogenFlashServerManager()
    mock_popen.return_value = Mock()
    config = HalogenFlashServerConfig(
        model_path=HALOGEN_FLASH_CHECKPOINT,
        port=8091,
        api_port=8091,
        engine_port=8092,
        options={
            "kv_pool_positions": 524288,
            "max_tok": 16384,
            "max_tokens_default": 8192,
            "host_reserve_gib": 20,
            "rope_yarn": 4,
            "prompt_cache": 2,
            "temperature": None,
        },
    )

    with (
        patch(
            "app.services.halogen_flash_server.model_manager.download_repository",
            new=AsyncMock(),
        ) as download,
        patch.object(manager, "_wait_for_health", new=AsyncMock()),
    ):
        assert await manager.start_server("flash-1", config) is True

    download.assert_awaited_once_with(
        HALOGEN_FLASH_REPO_ID, job_id="halogen-flash-flash-1"
    )
    command = mock_popen.call_args.args[0]
    environment = mock_popen.call_args.kwargs["env"]
    assert command[-2:] == ["/usr/local/bin/entrypoint.sh", "all"]
    assert environment["HALOGEN_API_PORT"] == "8091"
    assert environment["HALOGEN_PORT"] == "8092"
    assert environment["HALOGEN_ENGINE"] == "127.0.0.1:8092"
    assert environment["HALOGEN_CHECKPOINT"] == HALOGEN_FLASH_CHECKPOINT
    assert environment["HALOGEN_TOKENIZER"] == HALOGEN_FLASH_TOKENIZER
    assert environment["HALOGEN_KV_POOL_POSITIONS"] == "524288"
    assert environment["HALOGEN_MAX_TOK"] == "16384"
    assert environment["HALOGEN_MAX_TOKENS_DEFAULT"] == "8192"
    assert environment["HALOGEN_HOST_RESERVE_GIB"] == "20"
    assert environment["HALOGEN_ROPE_YARN"] == "4"
    assert environment["HALOGEN_PROMPT_CACHE"] == "2"
    assert "HALOGEN_TEMPERATURE" not in environment
    assert mock_popen.call_args.kwargs["start_new_session"] is True
    assert mock_popen.call_args.kwargs["bufsize"] == 1


@pytest.mark.asyncio
@patch("app.services.halogen_flash_server.subprocess.Popen")
async def test_flash_cache_directory_is_created_before_start(mock_popen, tmp_path):
    manager = HalogenFlashServerManager()
    mock_popen.return_value = Mock()
    config = HalogenFlashServerConfig(
        model_path=HALOGEN_FLASH_CHECKPOINT,
        port=8091,
        api_port=8091,
        engine_port=8092,
        options={"cache_dir_enabled": True},
    )

    with (
        patch("app.services.halogen_flash_server.settings.CACHE_PATH", str(tmp_path)),
        patch(
            "app.services.halogen_flash_server.model_manager.download_repository",
            new=AsyncMock(),
        ),
        patch.object(manager, "_wait_for_health", new=AsyncMock()),
    ):
        assert await manager.start_server("flash-cache-uuid", config) is True

    cache_dir = tmp_path / "flash-cache-uuid"
    assert cache_dir.is_dir()
    assert mock_popen.call_args.kwargs["env"]["HALOGEN_CACHE_DIR"] == str(cache_dir)


@pytest.mark.asyncio
async def test_flash_stop_kills_the_whole_process_group():
    manager = HalogenFlashServerManager()
    process = Mock(pid=1234)
    process.wait = Mock()
    manager.servers["flash-1"] = process
    manager.configs["flash-1"] = HalogenFlashServerConfig(
        model_path=HALOGEN_FLASH_CHECKPOINT,
        port=8091,
        api_port=8091,
        engine_port=8092,
        options={},
    )

    with patch("app.services.halogen_server.os.killpg") as killpg:
        assert await manager.stop_server("flash-1", force=True) is True

    killpg.assert_called_once_with(1234, 9)
    assert "flash-1" not in manager.servers


def test_halogen_capacity_uses_actual_kv_slots():
    manager = HalogenFlashServerManager()
    manager.configs["default"] = HalogenFlashServerConfig(
        model_path=HALOGEN_FLASH_CHECKPOINT,
        port=8091,
        api_port=8091,
        engine_port=8092,
        options={},
    )
    manager.configs["configured"] = HalogenFlashServerConfig(
        model_path=HALOGEN_FLASH_CHECKPOINT,
        port=8093,
        api_port=8093,
        engine_port=8094,
        options={"kv_slots": 6},
    )

    assert manager.get_effective_capacity("default") == 1
    assert manager.get_effective_capacity("configured") == 6


def _flash_start_request(**overrides):
    from app.api.routes.servers import ServerSpec, ServerStartRequest

    base = {
        "id": f"flash-{uuid4().hex[:8]}",
        "model_path": HALOGEN_FLASH_CHECKPOINT,
        "engine": "halogen-flash",
    }
    base.update(overrides)
    return ServerStartRequest(config=ServerSpec(**base))


@pytest.fixture(autouse=True)
def _clear_server_manager_state():
    from app.services.server_manager import server_manager

    yield
    server_manager.servers.clear()
    server_manager.configs.clear()
    server_manager.slot_generations.clear()


def _capture_start(captured: dict):
    """Record the started config and register it so capacity lookups succeed."""

    async def _start(sid, cfg):
        from app.services.server_manager import server_manager

        captured["config"] = cfg
        server_manager.configs[sid] = cfg
        server_manager.servers[sid] = Mock()
        return True

    return _start


@pytest.mark.asyncio
async def test_flash_first_start_allocates_static_range_ports(monkeypatch):
    from app.api.routes import servers as servers_route

    monkeypatch.setattr(
        "app.api.routes.servers.settings.AGENT_PLATFORM", "halogen-flash"
    )
    captured: dict = {}

    with (
        patch(
            "app.api.routes.servers.model_manager.download_repository",
            new=AsyncMock(),
        ),
        patch.object(
            servers_route.server_manager,
            "start_server",
            new=AsyncMock(side_effect=_capture_start(captured)),
        ),
    ):
        response = await servers_route.start_server(_flash_start_request())

    config = captured["config"]
    assert (
        servers_route.FLASH_PORT_START <= config.api_port < servers_route.FLASH_PORT_END
    )
    assert (
        servers_route.FLASH_PORT_START
        <= config.engine_port
        < servers_route.FLASH_PORT_END
    )
    assert config.api_port != config.engine_port
    assert response["port"] == config.api_port
    assert response["engine_port"] == config.engine_port


@pytest.mark.asyncio
async def test_flash_restart_reuses_pinned_ports_from_backend(monkeypatch):
    from app.api.routes import servers as servers_route

    monkeypatch.setattr(
        "app.api.routes.servers.settings.AGENT_PLATFORM", "halogen-flash"
    )
    captured: dict = {}

    with (
        patch(
            "app.api.routes.servers.model_manager.download_repository",
            new=AsyncMock(),
        ),
        patch.object(
            servers_route.server_manager,
            "start_server",
            new=AsyncMock(side_effect=_capture_start(captured)),
        ),
        patch.object(
            servers_route,
            "_allocate_port",
            side_effect=AssertionError("must not allocate when ports are pinned"),
        ),
    ):
        response = await servers_route.start_server(
            _flash_start_request(port=8210, engine_port=8211)
        )

    config = captured["config"]
    assert config.api_port == 8210
    assert config.engine_port == 8211
    assert response["port"] == 8210
    assert response["engine_port"] == 8211


@pytest.mark.asyncio
async def test_llamacpp_ports_stay_in_ephemeral_range():
    from app.api.routes import servers as servers_route

    captured: dict = {}

    with (
        patch(
            "app.api.routes.servers._ensure_model",
            new=AsyncMock(return_value="/models/x.gguf"),
        ),
        patch.object(
            servers_route.server_manager,
            "start_server",
            new=AsyncMock(side_effect=_capture_start(captured)),
        ),
    ):
        await servers_route.start_server(
            _flash_start_request(engine="llamacpp", model_path="/models/x.gguf")
        )

    config = captured["config"]
    assert (
        servers_route.EPHEMERAL_PORT_START
        <= config.port
        < servers_route.EPHEMERAL_PORT_END
    )
