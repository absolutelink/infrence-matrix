"""DB-backed download-progress tests.

Covers threading an owning ``server_id`` through download.* events so the
ServerInstance row carries an aggregated ``download_progress`` blob that
survives page reloads, while the legacy DownloadJob path stays intact.
"""

import uuid as uuid_module

import pytest

from app.db.session import AsyncSessionMaker
from app.models import Agent, Model, ServerInstance
from app.services.agent_manager import agent_manager
from app.services.server_startup import build_start_payload


def _make_agent(db) -> Agent:
    agent = Agent(
        name=f"dl-agent-{uuid_module.uuid4().hex[:8]}",
        host="127.0.0.1",
        port=8080,
        status="online",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _make_model(db) -> Model:
    model = Model(
        name=f"dl-model-{uuid_module.uuid4().hex[:8]}.gguf",
        path="/models/dl-model.gguf",
        size_bytes=3_000_000_000,
        architecture="llama",
        quantization="Q4_K_M",
        source="huggingface",
        source_repo_id="test-org/dl-repo",
        source_file="dl-model.gguf",
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def _make_instance(db, model: Model, agent: Agent) -> ServerInstance:
    instance = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias=f"dl-alias-{model.id.hex[:8]}",
        process_command="llama-server",
        status="preparing",
    )
    db.add(instance)
    db.commit()
    db.refresh(instance)
    return instance


async def _reload(server_id: uuid_module.UUID) -> ServerInstance | None:
    async with AsyncSessionMaker() as session:
        return await session.get(ServerInstance, server_id)


class TestServerInstanceDownloadProgress:
    @pytest.mark.asyncio
    async def test_progress_updates_server_instance(self, db) -> None:
        agent = _make_agent(db)
        model = _make_model(db)
        instance = _make_instance(db, model, agent)

        await agent_manager._handle_download_progress(
            str(agent.id),
            {
                "job_id": f"server-{instance.id}",
                "server_id": str(instance.id),
                "filename": "dl-model.gguf",
                "progress_percent": 42.5,
                "bytes_downloaded": 1_275_000_000,
                "total_bytes": 3_000_000_000,
                "speed_mbps": 55.0,
            },
            "download.progress",
        )

        reloaded = await _reload(instance.id)
        assert reloaded is not None
        blob = reloaded.download_progress
        assert blob["phase"] == "downloading"
        assert blob["progress_percent"] == 42.5
        assert blob["bytes_downloaded"] == 1_275_000_000
        assert blob["total_bytes"] == 3_000_000_000
        assert blob["speed_mbps"] == 55.0
        assert blob["filename"] == "dl-model.gguf"
        assert "updated_at" in blob

    @pytest.mark.asyncio
    async def test_started_sets_downloading_phase(self, db) -> None:
        agent = _make_agent(db)
        model = _make_model(db)
        instance = _make_instance(db, model, agent)

        await agent_manager._handle_download_progress(
            str(agent.id),
            {
                "job_id": f"server-{instance.id}",
                "server_id": str(instance.id),
                "filename": "dl-model.gguf",
            },
            "download.started",
        )

        reloaded = await _reload(instance.id)
        assert reloaded is not None
        assert reloaded.download_progress["phase"] == "downloading"
        assert reloaded.download_progress["progress_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_completed_sets_hundred_percent(self, db) -> None:
        agent = _make_agent(db)
        model = _make_model(db)
        instance = _make_instance(db, model, agent)

        await agent_manager._handle_download_progress(
            str(agent.id),
            {
                "job_id": f"server-{instance.id}",
                "server_id": str(instance.id),
                "filename": "dl-model.gguf",
                "progress_percent": 99.1,
            },
            "download.completed",
        )

        reloaded = await _reload(instance.id)
        assert reloaded is not None
        assert reloaded.download_progress["phase"] == "completed"
        assert reloaded.download_progress["progress_percent"] == 100.0
        assert "error" not in reloaded.download_progress

    @pytest.mark.asyncio
    async def test_failed_sets_error(self, db) -> None:
        agent = _make_agent(db)
        model = _make_model(db)
        instance = _make_instance(db, model, agent)

        await agent_manager._handle_download_progress(
            str(agent.id),
            {
                "job_id": f"server-{instance.id}",
                "server_id": str(instance.id),
                "filename": "dl-model.gguf",
                "error": "connection reset by peer",
            },
            "download.failed",
        )

        reloaded = await _reload(instance.id)
        assert reloaded is not None
        assert reloaded.download_progress["phase"] == "failed"
        assert reloaded.download_progress["error"] == "connection reset by peer"

    @pytest.mark.asyncio
    async def test_unknown_server_id_does_not_crash(self, db) -> None:
        """A server_id that points at no row must be a no-op (no exception)."""
        await agent_manager._handle_download_progress(
            "agent-1",
            {
                "job_id": "server-11111111-1111-1111-1111-111111111111",
                "server_id": "11111111-1111-1111-1111-111111111111",
                "progress_percent": 10.0,
            },
            "download.progress",
        )

    @pytest.mark.asyncio
    async def test_non_uuid_server_id_is_skipped(self, db) -> None:
        """A non-UUID server_id must not crash and must not touch any row."""
        await agent_manager._handle_download_progress(
            "agent-1",
            {
                "job_id": "server-file-foo.gguf",
                "server_id": "not-a-uuid",
                "progress_percent": 10.0,
            },
            "download.progress",
        )

    @pytest.mark.asyncio
    async def test_synthetic_job_id_without_server_id_does_not_crash(self, db) -> None:
        """Regression: synthetic 'server-<uuid>' job_id with no DownloadJob row
        and no server_id must be skipped by the DownloadJob path (used to crash
        the whole agent WebSocket)."""
        await agent_manager._handle_download_progress(
            "agent-1",
            {
                "job_id": "server-eb1f85c7-4ca2-41e6-bdf9-ce0da7451aa4",
                "progress_percent": 33.0,
                "bytes_downloaded": 1000,
                "speed_mbps": 5.0,
            },
            "download.progress",
        )


class TestBuildStartPayloadServerId:
    def test_source_carries_server_id(self, db) -> None:
        agent = _make_agent(db)
        model = _make_model(db)
        instance = _make_instance(db, model, agent)

        payload = build_start_payload(instance, model)
        assert payload["source"]["server_id"] == str(instance.id)
        assert payload["source"]["job_id"] == f"server-{instance.id}"

    def test_mmproj_and_draft_carry_server_id(self, db) -> None:
        agent = _make_agent(db)
        model = _make_model(db)
        mmproj = Model(
            name=f"mmproj-{uuid_module.uuid4().hex[:8]}.gguf",
            path="/models/mmproj.gguf",
            size_bytes=123456789,
            architecture="clip",
            model_type="mmproj",
            quantization="F16",
            source="huggingface",
            source_repo_id="test-org/mmproj-repo",
        )
        dflash = Model(
            name=f"dflash-{uuid_module.uuid4().hex[:8]}.gguf",
            path="/models/dflash.gguf",
            size_bytes=123456789,
            architecture="qwen",
            model_type="dflash",
            quantization="Q4_K_M",
            source="huggingface",
            source_repo_id="test-org/dflash-repo",
        )
        db.add(mmproj)
        db.add(dflash)
        db.commit()
        db.refresh(mmproj)
        db.refresh(dflash)

        instance = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            alias=f"mm-df-alias-{model.id.hex[:8]}",
            process_command="llama-server",
            mmproj_model_id=mmproj.id,
            dflash_model_id=dflash.id,
            status="stopped",
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)

        payload = build_start_payload(instance, model)
        assert payload["mmproj_source"]["server_id"] == str(instance.id)
        assert payload["mmproj_source"]["job_id"] == f"server-{instance.id}-mmproj"
        assert payload["draft_source"]["server_id"] == str(instance.id)
        assert payload["draft_source"]["job_id"] == f"server-{instance.id}-dflash"
