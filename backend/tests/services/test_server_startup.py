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
    initialize_server,
    pick_primary_filename,
    resolve_model_filenames,
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


class TestResolveModelFilenames:
    def _model(self, **kwargs: object) -> Model:
        base = {
            "name": "m",
            "path": "/models/m.gguf",
            "size_bytes": 1,
            "architecture": "llama",
            "quantization": "Q4_K_M",
            "source": "huggingface",
        }
        base.update(kwargs)
        return Model(**base)  # type: ignore[arg-type]

    def test_explicit_source_files_sorted_by_index(self) -> None:
        model = self._model(
            source_files=[
                "model-00002-of-00003.gguf",
                "model-00001-of-00003.gguf",
                "model-00003-of-00003.gguf",
            ]
        )
        assert resolve_model_filenames(model) == [
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ]

    def test_legacy_json_array_source_file_parsed(self) -> None:
        model = self._model(
            source_file='["b-00002-of-00002.gguf", "a-00001-of-00002.gguf"]'
        )
        assert resolve_model_filenames(model) == [
            "a-00001-of-00002.gguf",
            "b-00002-of-00002.gguf",
        ]

    def test_single_source_file(self) -> None:
        model = self._model(source_file="plain.Q4_K_M.gguf")
        assert resolve_model_filenames(model) == ["plain.Q4_K_M.gguf"]

    def test_falls_back_to_path_basename(self) -> None:
        model = self._model(source_file=None, path="/models/dir/fallback.gguf")
        assert resolve_model_filenames(model) == ["fallback.gguf"]

    def test_malformed_json_array_treated_as_single(self) -> None:
        model = self._model(source_file="[not, json")
        assert resolve_model_filenames(model) == ["[not, json"]

    def test_pick_primary_prefers_first_part(self) -> None:
        names = [
            "model-00002-of-00003.gguf",
            "model-00001-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ]
        assert pick_primary_filename(names) == "model-00001-of-00003.gguf"

    def test_pick_primary_single_returns_it(self) -> None:
        assert pick_primary_filename(["only.gguf"]) == "only.gguf"


class TestBuildStartPayloadSplit:
    def _agent(self, db) -> Agent:
        agent = Agent(
            name=f"split-test-agent-{uuid_module.uuid4().hex[:8]}",
            host="127.0.0.1",
            port=8080,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return agent

    def test_split_payload_carries_all_parts_and_primary(self, db) -> None:
        model = Model(
            name=f"split-model-{uuid_module.uuid4().hex[:8]}.gguf",
            path="/models/split-model.gguf",
            size_bytes=3000,
            architecture="llama",
            quantization="Q4_K_M",
            source="huggingface",
            source_repo_id="test-org/split-repo",
            source_file="split-model-00001-of-00003.gguf",
            source_files=[
                "split-model-00003-of-00003.gguf",
                "split-model-00001-of-00003.gguf",
                "split-model-00002-of-00003.gguf",
            ],
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        instance = ServerInstance(
            model_id=model.id,
            agent_id=self._agent(db).id,
            alias=f"split-alias-{model.id.hex[:8]}",
            process_command="llama-server",
            status="stopped",
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)

        payload = build_start_payload(instance, model)
        assert payload["source"]["filenames"] == [
            "split-model-00001-of-00003.gguf",
            "split-model-00002-of-00003.gguf",
            "split-model-00003-of-00003.gguf",
        ]
        assert payload["source"]["filename"] == "split-model-00001-of-00003.gguf"

    def test_single_file_payload_unchanged(self, db) -> None:
        model = Model(
            name=f"single-model-{uuid_module.uuid4().hex[:8]}.gguf",
            path="/models/single-model.gguf",
            size_bytes=1000,
            architecture="llama",
            quantization="Q4_K_M",
            source="huggingface",
            source_repo_id="test-org/single-repo",
            source_file="single-model.gguf",
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        instance = ServerInstance(
            model_id=model.id,
            agent_id=self._agent(db).id,
            alias=f"single-alias-{model.id.hex[:8]}",
            process_command="llama-server",
            status="stopped",
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)

        payload = build_start_payload(instance, model)
        assert payload["source"]["filename"] == "single-model.gguf"
        assert payload["source"]["filenames"] == ["single-model.gguf"]


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
        instance.health_status = "healthy"
        db.add(instance)
        db.commit()

        result = await ensure_server_ready_by_id(str(instance.id))
        assert result.status == "running"

    async def test_starting_waits_until_running(self, db) -> None:
        """A starting instance is waited on; when the row flips to running
        the waiter returns it."""
        model = _make_model(db, "starting-model.gguf")
        instance = self._instance(db, model, "starting")

        async def flip_to_running():
            instance.status = "running"
            instance.health_status = "healthy"
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


class TestInitializeServer:
    """Initialization must actually launch the process to read its metadata.

    Regression: the benchmark gate short-circuits on instance status, and
    ``initialize_server`` sets ``metadata_gathering`` before dispatching the
    start, so the start was dropped and metadata lookup 404'd.
    """

    def _instance(self, db, status: str = "uninitialized") -> ServerInstance:
        agent = Agent(
            name=f"init-test-agent-{uuid_module.uuid4().hex[:8]}",
            host="127.0.0.1",
            port=8080,
            status="online",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        model = _make_model(db, "init-model.Q4_K_M.gguf")
        model.source_repo_id = "test-org/init-repo"
        db.add(model)
        db.commit()
        instance = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            alias=f"init-alias-{model.id.hex[:8]}",
            process_command="llama-server",
            status=status,
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)
        return instance

    async def test_dispatches_start_during_metadata_gathering(self, db) -> None:
        instance = self._instance(db)
        calls: list[tuple[str, str]] = []

        async def fake_send(_agent_id, method, path, _payload, **_kwargs):
            calls.append((method, path))
            if path == "/servers/start":
                async with AsyncSessionMaker() as session:
                    row = await session.get(ServerInstance, instance.id)
                    assert row.status == "metadata_gathering"
            elif path.endswith(f"/servers/metadata/{instance.id}"):
                return {"data": [{"id": "init-model"}]}
            return {}

        with patch(
            "app.services.server_startup.agent_manager.send_to_agent",
            new=fake_send,
        ):
            await initialize_server(
                str(instance.agent_id),
                str(instance.id),
                {"config": {"id": str(instance.id), "engine": "llamacpp"}},
            )

        assert ("POST", "/servers/start") in calls
        assert ("POST", "/servers/stop") in calls
        async with AsyncSessionMaker() as session:
            row = await session.get(ServerInstance, instance.id)
        assert row.status == "stopped"
        assert row.model_metadata == {"id": "init-model"}
        assert row.error_message is None

    async def test_initialization_failed_when_metadata_lookup_fails(self, db) -> None:
        """When metadata cannot be read the row must not look usable."""
        instance = self._instance(db)

        async def fake_send(_agent_id, _method, path, _payload, **_kwargs):
            if path == "/servers/metadata" + f"/{instance.id}":
                raise RuntimeError("Server is not running")
            return {}

        with patch(
            "app.services.server_startup.agent_manager.send_to_agent",
            new=fake_send,
        ):
            await initialize_server(
                str(instance.agent_id),
                str(instance.id),
                {"config": {"id": str(instance.id), "engine": "llamacpp"}},
            )

        async with AsyncSessionMaker() as session:
            row = await session.get(ServerInstance, instance.id)
        assert row.status == "initialization_failed"
        assert not row.model_metadata
