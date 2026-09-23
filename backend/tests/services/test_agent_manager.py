"""Tests for Agent Manager service."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import Agent
from app.services.agent_manager import AgentManager


class TestAgentManagerRegistration:
    """Test agent registration functionality."""

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_register_new_agent(self, mock_session_maker):
        """Test registering a new agent."""
        manager = AgentManager()

        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        mock_session_maker.return_value = mock_session

        agent_data = {
            "agent_id": "test-agent-id",
            "name": "Test Agent",
            "host": "localhost",
            "port": 8080,
            "gpu_info": {"name": "RTX 4090", "vram_total": 24576000000},
        }

        agent = await manager.register_agent(agent_data)

        assert agent is not None
        assert str(agent.id) in manager.agents
        assert agent.status == "online"
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_register_existing_agent(self, mock_session_maker):
        """Test updating an existing agent."""
        manager = AgentManager()

        existing_agent = Agent(
            name="Test Agent",
            host="old-host",
            port=8080,
            status="offline",
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=existing_agent)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        mock_session_maker.return_value = mock_session

        agent_data = {
            "agent_id": "test-agent-id",
            "name": "Test Agent",
            "host": "new-host",
            "port": 8080,
            "gpu_info": {},
        }

        agent = await manager.register_agent(agent_data)

        assert agent.host == "new-host"
        assert agent.status == "online"


class TestAgentManagerList:
    """Test agent listing functionality."""

    @pytest.mark.asyncio
    async def test_list_agents_empty(self):
        """Test listing agents when none registered."""
        manager = AgentManager()

        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all = Mock(return_value=[])
        mock_result.scalars = Mock(return_value=mock_scalars)
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.agent_manager.AsyncSessionMaker", return_value=mock_session
        ):
            agents = await manager.list_agents()

        assert agents == []

    @pytest.mark.asyncio
    async def test_list_agents_with_data(self):
        """Test listing agents with registered agents."""
        manager = AgentManager()

        agent = Agent(
            name="Test Agent",
            host="localhost",
            port=8080,
            status="online",
            gpu_info={"name": "RTX 4090"},
            websocket_connected=True,
            last_seen=datetime.utcnow(),
        )

        mock_scalars = Mock()
        mock_scalars.all = Mock(return_value=[agent])
        mock_result = Mock()
        mock_result.scalars = Mock(return_value=mock_scalars)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.agent_manager.AsyncSessionMaker", return_value=mock_session
        ):
            agents = await manager.list_agents()

        assert len(agents) == 1
        assert agents[0]["name"] == "Test Agent"
        assert agents[0]["status"] == "online"


class TestAgentManagerSend:
    """Test sending requests to agents."""

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.httpx.AsyncClient")
    async def test_send_to_agent_success(self, mock_client):
        """Test successful request to agent."""
        manager = AgentManager()

        agent = Agent(
            id="test-agent-id",
            name="Test Agent",
            host="localhost",
            port=8080,
            status="online",
        )

        manager.agents["test-agent-id"] = agent

        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = Mock()

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.request = AsyncMock(return_value=mock_response)

        mock_client.return_value = mock_client_instance

        result = await manager.send_to_agent(
            "test-agent-id",
            "POST",
            "/servers/start",
            {"model_path": "/models/test.gguf"},
        )

        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_send_to_agent_not_found(self):
        """Test request to non-existent agent."""
        manager = AgentManager()

        with pytest.raises(ValueError, match="Agent nonexistent not found"):
            await manager.send_to_agent("nonexistent", "GET", "/health")

    @pytest.mark.asyncio
    async def test_send_to_agent_offline(self):
        """Test request to offline agent."""
        manager = AgentManager()

        agent = Agent(
            id="test-agent-id",
            name="Test Agent",
            host="localhost",
            port=8080,
            status="offline",
        )

        manager.agents["test-agent-id"] = agent

        with pytest.raises(RuntimeError, match="Agent test-agent-id is offline"):
            await manager.send_to_agent("test-agent-id", "GET", "/health")


class TestAgentManagerGet:
    """Test getting individual agents."""

    @pytest.mark.asyncio
    async def test_get_agent_exists(self):
        """Test getting an existing agent."""
        manager = AgentManager()

        agent = Agent(
            name="Test Agent",
            host="localhost",
            port=8080,
            status="online",
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(return_value=agent)

        with patch(
            "app.services.agent_manager.AsyncSessionMaker", return_value=mock_session
        ):
            result = await manager.get_agent(str(agent.id))

        assert result == agent

    @pytest.mark.asyncio
    async def test_get_agent_not_exists(self):
        """Test getting a non-existent agent."""
        manager = AgentManager()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(return_value=None)

        with patch(
            "app.services.agent_manager.AsyncSessionMaker", return_value=mock_session
        ):
            result = await manager.get_agent("b8d06e5c-1c4d-47d5-ba47-c8a55fc67592")

        assert result is None


class TestHandleDownloadProgressSyntheticJobId:
    """Regression: auto-start downloads emit download.progress with a synthetic
    job_id like 'server-<uuid>' (not a DownloadJob row). The handler used to
    run a UUID cast against it, raising psycopg InvalidTextRepresentation and
    killing the whole agent WebSocket on every progress tick."""

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_synthetic_job_id_is_skipped(self, mock_session_maker):
        from unittest.mock import AsyncMock, MagicMock

        from app.services.agent_manager import agent_manager

        session = MagicMock()
        session.execute = AsyncMock(return_value=MagicMock())
        mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_session_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        await agent_manager._handle_download_progress(
            "agent-1",
            {
                "job_id": "server-eb1f85c7-4ca2-41e6-bdf9-ce0da7451aa4",
                "progress_percent": 42.0,
                "bytes_downloaded": 1000,
                "speed_mbps": 5.0,
            },
        )

        # No query against download_jobs at all
        session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_valid_uuid_job_id_still_updates(self, mock_session_maker):
        from unittest.mock import AsyncMock, MagicMock

        from app.services.agent_manager import agent_manager

        job = MagicMock()
        session = MagicMock()
        session.commit = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = job
        session.execute = AsyncMock(return_value=result)
        mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_session_maker.return_value.__aexit__ = AsyncMock(return_value=None)

        await agent_manager._handle_download_progress(
            "agent-1",
            {
                "job_id": "12345678-1234-5678-1234-567812345678",
                "progress_percent": 50.0,
                "bytes_downloaded": 2000,
                "speed_mbps": 10.0,
            },
        )

        assert job.progress_percent == 50.0
        assert job.current_speed == 10_000_000
        session.add.assert_called_once_with(job)
        assert session.commit.await_count == 1
