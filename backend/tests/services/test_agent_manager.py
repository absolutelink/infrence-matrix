"""Tests for Agent Manager service."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import Agent, ServerInstance
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

        mock_scalars = Mock()
        mock_scalars.all = Mock(return_value=[])
        mock_restore_result = Mock()
        mock_restore_result.scalars = Mock(return_value=mock_scalars)
        mock_stale_result = Mock()
        mock_stale_result.rowcount = 0

        # First execute() = agent lookup, second = stale update,
        # third = restore query
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(
            side_effect=[mock_result, mock_stale_result, mock_restore_result]
        )
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

        mock_scalars = Mock()
        mock_scalars.all = Mock(return_value=[])
        mock_restore_result = Mock()
        mock_restore_result.scalars = Mock(return_value=mock_scalars)
        mock_stale_result = Mock()
        mock_stale_result.rowcount = 0

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(
            side_effect=[mock_result, mock_stale_result, mock_restore_result]
        )
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


class TestAgentManagerRestoreOnRegister:
    """Agent registration must not start stopped server instances."""

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_stopped_instances_are_not_redispatched(self, mock_session_maker):
        from unittest.mock import MagicMock

        from app.models import Model
        from app.services.agent_manager import agent_manager

        agent = Agent(
            name="restart-agent",
            host="localhost",
            port=8080,
            status="online",
            last_seen=datetime.now(),
        )
        model = Model(
            name="restore-model.gguf",
            path="/models/restore-model.gguf",
            size_bytes=123,
            architecture="llama",
            quantization="Q4_K_M",
            source="huggingface",
        )
        stopped_instance = MagicMock()
        stopped_instance.id = "6f1d2c3a-0000-0000-0000-000000000001"
        stopped_instance.model_id = model.id
        stopped_instance.status = "stopped"

        lookup_result = MagicMock()
        lookup_result.scalar_one_or_none.return_value = agent
        restore_result = MagicMock()
        restore_result.scalars.return_value.all.return_value = [stopped_instance]
        stale_result = MagicMock()
        stale_result.rowcount = 0

        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(
            side_effect=[lookup_result, stale_result, restore_result]
        )
        session.get = AsyncMock(return_value=model)
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        mock_session_maker.return_value = session

        await agent_manager.register_agent(
            {
                "name": "restart-agent",
                "host": "localhost",
                "port": 8080,
                "running_server_ids": [],
            }
        )

        assert stopped_instance.status == "stopped"

    @pytest.mark.asyncio
    @patch("app.services.server_startup.dispatch_start", new_callable=AsyncMock)
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_no_instances_no_dispatch(self, mock_session_maker, mock_dispatch):
        from unittest.mock import MagicMock

        from app.services.agent_manager import agent_manager

        agent = Agent(
            name="idle-agent",
            host="localhost",
            port=8080,
            status="online",
            last_seen=datetime.now(),
        )

        lookup_result = MagicMock()
        lookup_result.scalar_one_or_none.return_value = agent
        stale_result = MagicMock()
        stale_result.rowcount = 0

        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(side_effect=[lookup_result, stale_result])
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        mock_session_maker.return_value = session

        await agent_manager.register_agent(
            {
                "name": "idle-agent",
                "host": "localhost",
                "port": 8080,
                "running_server_ids": [],
            }
        )

        mock_dispatch.assert_not_called()


class TestAgentManagerStartingPromotion:
    """Registration treats the agent's running_server_ids report as ground
    truth: a stale "starting" row the agent reports as live must be promoted
    to running. Regression: a backend restart loses dispatch_start's ack, so
    the row stayed "starting" forever and every request waited in
    wait_until_ready (900s) — a full deploy deadlock."""

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_starting_row_promoted_when_agent_reports_running(
        self, mock_session_maker
    ):
        from unittest.mock import MagicMock

        from app.services.agent_manager import agent_manager

        agent = Agent(
            name="lfai-node-01",
            host="localhost",
            port=8080,
            status="online",
            last_seen=datetime.now(),
        )
        running_id = "5eb19624-0000-0000-0000-000000000001"

        lookup_result = MagicMock()
        lookup_result.scalar_one_or_none.return_value = agent
        restore_result = MagicMock()
        restore_result.scalars.return_value.all.return_value = []
        confirmed_result = MagicMock()
        confirmed_result.rowcount = 1
        stale_result = MagicMock()
        stale_result.rowcount = 0

        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(
            side_effect=[lookup_result, confirmed_result, stale_result, restore_result]
        )
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        mock_session_maker.return_value = session

        await agent_manager.register_agent(
            {
                "name": "lfai-node-01",
                "host": "localhost",
                "port": 8080,
                "running_server_ids": [running_id],
            }
        )

        # Second execute() call must be the promotion update
        promote_call = session.execute.call_args_list[1]
        assert promote_call is not None

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_no_promotion_without_running_ids(self, mock_session_maker):
        """Empty running_server_ids (agent down) must not promote anything."""
        from unittest.mock import MagicMock

        from app.services.agent_manager import agent_manager

        agent = Agent(
            name="empty-report-agent",
            host="localhost",
            port=8080,
            status="online",
            last_seen=datetime.now(),
        )

        lookup_result = MagicMock()
        lookup_result.scalar_one_or_none.return_value = agent
        restore_result = MagicMock()
        restore_result.scalars.return_value.all.return_value = []
        stale_result = MagicMock()
        stale_result.rowcount = 0

        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(
            side_effect=[lookup_result, stale_result, restore_result]
        )
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        mock_session_maker.return_value = session

        await agent_manager.register_agent(
            {
                "name": "empty-report-agent",
                "host": "localhost",
                "port": 8080,
                "running_server_ids": [],
            }
        )

        # Only lookup + stale queries ran — no promotion or start update.
        assert session.execute.await_count == 2

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_spawned_but_unhealthy_server_is_not_promoted(
        self, mock_session_maker
    ):
        """A spawned llama process is not healthy until its /health check passes."""
        from unittest.mock import MagicMock

        from app.services.agent_manager import agent_manager

        agent = Agent(
            name="starting-agent",
            host="localhost",
            port=8080,
            status="online",
            last_seen=datetime.now(),
        )
        running_id = "5eb19624-0000-0000-0000-000000000002"

        lookup_result = MagicMock()
        lookup_result.scalar_one_or_none.return_value = agent
        restore_result = MagicMock()
        restore_result.scalars.return_value.all.return_value = []
        stale_result = MagicMock()
        stale_result.rowcount = 0
        confirmed_result = MagicMock()
        confirmed_result.rowcount = 0
        reset_result = MagicMock()
        reset_result.rowcount = 0

        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(
            side_effect=[
                lookup_result,
                confirmed_result,
                reset_result,
                stale_result,
                restore_result,
            ]
        )
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        mock_session_maker.return_value = session

        await agent_manager.register_agent(
            {
                "name": "starting-agent",
                "host": "localhost",
                "port": 8080,
                "running_server_ids": [running_id],
                "healthy_server_ids": [],
            }
        )

        promote_call = session.execute.call_args_list[1]
        assert running_id not in str(promote_call)


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


class TestAgentManagerServerHealth:
    """Test persistence of agent-reported server health transitions."""

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_server_health_event_updates_instance(self, mock_session_maker):
        server = ServerInstance(
            model_id="11111111-1111-1111-1111-111111111111",
            agent_id="22222222-2222-2222-2222-222222222222",
            alias="health-test",
            process_command="llama-server",
            status="running",
            health_status="healthy",
        )
        result = Mock()
        result.scalar_one_or_none.return_value = server
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(return_value=result)
        session.add = Mock()
        session.commit = AsyncMock()
        mock_session_maker.return_value = session

        manager = AgentManager()
        await manager._handle_server_health(
            "agent-1",
            {
                "server_id": str(server.id),
                "status": "unhealthy",
                "error": "health endpoint returned HTTP 503",
            },
        )

        assert server.health_status == "unhealthy"
        assert server.error_message == "health endpoint returned HTTP 503"
        assert server.last_health_check is not None
        session.add.assert_called_once_with(server)
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("app.services.agent_manager.AsyncSessionMaker")
    async def test_healthy_event_clears_previous_error(self, mock_session_maker):
        server = ServerInstance(
            model_id="11111111-1111-1111-1111-111111111111",
            agent_id="22222222-2222-2222-2222-222222222222",
            alias="health-recovery",
            process_command="llama-server",
            status="running",
            health_status="unhealthy",
            error_message="connection refused",
        )
        result = Mock()
        result.scalar_one_or_none.return_value = server
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(return_value=result)
        session.add = Mock()
        session.commit = AsyncMock()
        mock_session_maker.return_value = session

        manager = AgentManager()
        await manager._handle_server_health(
            "agent-1", {"server_id": str(server.id), "status": "healthy"}
        )

        assert server.health_status == "healthy"
        assert server.error_message is None


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
