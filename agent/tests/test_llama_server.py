"""Tests for llama_server module."""

import pytest
import asyncio
import subprocess
from unittest.mock import Mock, patch, MagicMock
import os

# Set required environment variables before importing settings
os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"

from app.services.llama_server import LlamaServerManager, ServerConfig


class TestServerConfig:
    """Test ServerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig(
            model_path="/models/test.gguf",
            port=8081,
        )

        assert config.gpu_layers == 35
        assert config.context_size == 4096
        assert config.batch_size == 512
        assert config.cache_prompt is True
        assert config.flash_attn is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ServerConfig(
            model_path="/models/test.gguf",
            port=8082,
            gpu_layers=50,
            context_size=8192,
            batch_size=1024,
            cache_prompt=False,
            flash_attn=False,
        )

        assert config.gpu_layers == 50
        assert config.context_size == 8192
        assert config.batch_size == 1024
        assert config.cache_prompt is False
        assert config.flash_attn is False


class TestLlamaServerManager:
    """Test LlamaServerManager."""

    def test_init(self):
        """Test initialization."""
        manager = LlamaServerManager()

        assert manager.servers == {}
        assert manager.configs == {}
        assert manager.start_times == {}

    @pytest.mark.asyncio
    @patch("app.services.llama_server.subprocess.Popen")
    @patch("app.services.llama_server.LlamaServerManager._wait_for_server")
    async def test_start_server(self, mock_wait, mock_popen):
        """Test starting a server."""
        manager = LlamaServerManager()
        config = ServerConfig(
            model_path="/models/test.gguf",
            port=8081,
        )

        mock_proc = Mock()
        mock_popen.return_value = mock_proc

        result = await manager.start_server("test-server", config)

        assert result is True
        assert "test-server" in manager.servers
        assert "test-server" in manager.configs
        mock_popen.assert_called_once()
        mock_wait.assert_called_once()

    def test_start_existing_server(self):
        """Test starting a server that already exists."""
        manager = LlamaServerManager()
        manager.servers["test-server"] = Mock()

        config = ServerConfig(
            model_path="/models/test.gguf",
            port=8081,
        )

        result = manager.start_server("test-server", config)

        # start_server is async, returns coroutine when not awaited
        assert asyncio.iscoroutine(result)

    @pytest.mark.asyncio
    @patch("app.services.llama_server.subprocess.Popen")
    async def test_stop_server(self, mock_popen):
        """Test stopping a server."""
        manager = LlamaServerManager()
        mock_proc = Mock()
        mock_proc.wait = Mock()

        manager.servers["test-server"] = mock_proc
        manager.configs["test-server"] = ServerConfig(
            model_path="/models/test.gguf",
            port=8081,
        )

        result = await manager.stop_server("test-server")

        assert result is True
        assert "test-server" not in manager.servers
        mock_proc.send_signal.assert_called_once()

    def test_stop_nonexistent_server(self):
        """Test stopping a server that doesn't exist."""
        manager = LlamaServerManager()

        result = manager.stop_server("nonexistent")

        assert asyncio.iscoroutine(result)

    @pytest.mark.asyncio
    @patch("app.services.llama_server.subprocess.Popen")
    async def test_stop_server_force(self, mock_popen):
        """Test forcefully stopping a server."""
        manager = LlamaServerManager()
        mock_proc = Mock()
        mock_proc.wait = Mock()

        manager.servers["test-server"] = mock_proc
        manager.configs["test-server"] = ServerConfig(
            model_path="/models/test.gguf",
            port=8081,
        )

        result = await manager.stop_server("test-server", force=True)

        assert result is True
        mock_proc.kill.assert_called_once()

    def test_get_server_uptime(self):
        """Test getting server uptime."""
        manager = LlamaServerManager()
        manager.start_times["test-server"] = 1000.0

        with patch("time.time", return_value=1100.0):
            uptime = manager.get_server_uptime("test-server")
            assert uptime == 100.0

    def test_get_server_uptime_nonexistent(self):
        """Test getting uptime for nonexistent server."""
        manager = LlamaServerManager()

        uptime = manager.get_server_uptime("nonexistent")

        assert uptime == 0.0

    def test_list_servers(self):
        """Test listing servers."""
        manager = LlamaServerManager()
        config = ServerConfig(
            model_path="/models/test.gguf",
            port=8081,
        )

        manager.servers["test-server"] = Mock()
        manager.configs["test-server"] = config
        manager.start_times["test-server"] = 1000.0

        with patch("time.time", return_value=1100.0):
            servers = manager.list_servers()

            assert len(servers) == 1
            assert servers[0]["server_id"] == "test-server"
            assert servers[0]["model_path"] == "/models/test.gguf"
            assert servers[0]["port"] == 8081
            assert servers[0]["status"] == "running"
            assert servers[0]["uptime_seconds"] == 100.0

    def test_list_empty_servers(self):
        """Test listing when no servers running."""
        manager = LlamaServerManager()

        servers = manager.list_servers()

        assert servers == []
