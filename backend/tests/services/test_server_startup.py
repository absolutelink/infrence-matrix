"""Tests for the shared server startup service (invisible cold starts)."""

import asyncio
import uuid as uuid_module
from unittest.mock import patch

import pytest

from app.db.session import AsyncSessionMaker
from app.models import Agent, Model, ServerInstance
from app.services.server_startup import (
    ServerStartupError,
    build_start_payload,
    ensure_server_ready_by_id,
)


def _make_model(db, name: str = "startup-model.Q4_K_M.gguf") -> Model:
    model = Model(
        name=name,
        path=f"/models/{name}",
        size_bytes=1234567890,
        architecture="llama",
        parameter_count=3000000000,
        quantization="Q4_K_M",
        supports_embeddings=False,
        supports_vision=False,
        source="huggingface",
    )
    db.add(model)
    db.commit()
    return model


class TestBuildStartPayload:
    def _agent(self, db) -> Agent:
        agent = Agent(
            name=f"payload-test-agent-{uuid_module.uuid4().hex[:8]}",
            host="127.0.0.1",
            port=8080,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return agent

    def test_payload_shape(self, db) -> None:
        model = _make_model(db)
        model.source_repo_id = "test-org/test-repo"
        model.source_file = "startup-model.Q4_K_M.gguf"
        db.add(model)
        db.commit()
        instance = ServerInstance(
            model_id=model.id,
            agent_id=self._agent(db).id,
            alias="test-alias",
            process_command="llama-server",
            gpu_layers=40,
            context_size=8192,
            flash_attn=True,
            status="stopped",
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)

        payload = build_start_payload(instance, model)
        assert payload["config"]["id"] == str(instance.id)
        assert payload["config"]["model_path"] == model.path
        assert payload["config"]["context_size"] == 8192
        assert payload["config"]["options"] == {}
        assert payload["source"]["repo_id"] == "test-org/test-repo"
        assert payload["source"]["filename"] == "startup-model.Q4_K_M.gguf"

    def test_no_source_when_no_repo(self, db) -> None:
        model = _make_model(db, "nosource.Q4_K_M.gguf")
        model.source_repo_id = None
        db.add(model)
        db.commit()
        instance = ServerInstance(
            model_id=model.id,
            agent_id=self._agent(db).id,
            alias="nosource-alias",
            process_command="llama-server",
            status="stopped",
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)

        payload = build_start_payload(instance, model)
        assert payload["source"] is None

    def test_selected_mmproj_included_in_payload(self, db) -> None:
        model = _make_model(db, "vision-model.Q4_K_M.gguf")
        model.source_repo_id = "test-org/vision-repo"
        mmproj = Model(
            name="vision-mmproj.gguf",
            path="/models/vision-mmproj.gguf",
            size_bytes=123456789,
            architecture="clip",
            model_type="mmproj",
            quantization="F16",
            source="huggingface",
            source_repo_id="test-org/vision-repo",
        )
        db.add(mmproj)
        db.commit()
        db.refresh(mmproj)
        instance = ServerInstance(
            model_id=model.id,
            agent_id=self._agent(db).id,
            alias="mmproj-alias",
            process_command="llama-server",
            mmproj_model_id=mmproj.id,
            status="stopped",
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)

        payload = build_start_payload(instance, model)
        assert payload["config"]["mmproj_path"] == mmproj.path

    def test_no_mmproj_selected_omits_flag(self, db) -> None:
        model = _make_model(db, "plain-model.Q4_K_M.gguf")
        model.source_repo_id = None
        db.add(model)
        db.commit()
        instance = ServerInstance(
            model_id=model.id,
            agent_id=self._agent(db).id,
            alias="no-mmproj-alias",
            process_command="llama-server",
            status="stopped",
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)

        payload = build_start_payload(instance, model)
        assert "mmproj_path" not in payload["config"]
        assert "mmproj_source" not in payload


class TestEnsureServerReady:
    def _instance(self, db, model: Model, status: str) -> ServerInstance:
        agent = Agent(
            name=f"startup-test-agent-{uuid_module.uuid4().hex[:8]}",
            host="127.0.0.1",
            port=8080,
            status="online",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        instance = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            alias=f"alias-{status}-{model.id.hex[:8]}",
            process_command="llama-server",
            status=status,
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)
        return instance

    async def test_running_returns_immediately(self, db) -> None:
        model = _make_model(db, "running-model.gguf")
        instance = self._instance(db, model, "running")

        result = await ensure_server_ready_by_id(str(instance.id))
        assert result.status == "running"

    async def test_starting_waits_until_running(self, db) -> None:
        """A starting instance is waited on; when the row flips to running
        the waiter returns it."""
        model = _make_model(db, "starting-model.gguf")
        instance = self._instance(db, model, "starting")

        async def flip_to_running():
            instance.status = "running"
            db.add(instance)
            db.commit()

        task = asyncio.create_task(flip_to_running())
        result = await ensure_server_ready_by_id(
            str(instance.id), start_timeout=5, poll_interval=0.05
        )
        await task
        assert result.status == "running"

    async def test_error_status_raises(self, db) -> None:
        model = _make_model(db, "error-model.gguf")
        instance = self._instance(db, model, "error")
        instance.error_message = "boom"
        db.add(instance)
        db.commit()

        from app.services import server_startup as ss

        async def failing_dispatch(_agent_id, server_id, _payload):
            async with AsyncSessionMaker() as session:
                row = await session.get(ServerInstance, uuid_module.UUID(server_id))
                row.status = "error"
                row.error_message = "boom"
                session.add(row)
                await session.commit()

        with patch.object(ss, "dispatch_start", new=failing_dispatch):
            with pytest.raises(ServerStartupError, match="boom"):
                await ensure_server_ready_by_id(
                    str(instance.id), start_timeout=2, poll_interval=0.1
                )

    async def test_stopped_dispatches_start_and_waits(self, db) -> None:
        model = _make_model(db, "stopped-model.gguf")
        instance = self._instance(db, model, "stopped")

        dispatched: dict = {}

        async def fake_dispatch(_agent_id, server_id, _payload):
            dispatched["called"] = True
            dispatched["server_id"] = server_id
            # Simulate the agent completing startup by flipping the row
            async with AsyncSessionMaker() as session:
                row = await session.get(ServerInstance, uuid_module.UUID(server_id))
                row.status = "running"
                row.health_status = "healthy"
                session.add(row)
                await session.commit()

        from app.services import server_startup as ss

        with patch.object(ss, "dispatch_start", new=fake_dispatch):
            result = await ensure_server_ready_by_id(
                str(instance.id), start_timeout=5, poll_interval=0.2
            )
        assert dispatched["called"] is True
        assert dispatched["server_id"] == str(instance.id)
        assert result.status == "running"

    async def test_missing_instance_raises(self) -> None:
        with pytest.raises(ServerStartupError):
            await ensure_server_ready_by_id("00000000-0000-0000-0000-000000000000")
