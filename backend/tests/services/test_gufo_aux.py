"""Tests for Gufo aux-file source resolution in server_startup.

The Gufo engine references auxiliary GGUF files (mmproj, dflash, dspark,
mtp) by their library ``Model.path`` inside ``engine_options``. These must
be resolved to download sources so the agent can fetch them to the exact
path the binary is told to use.
"""

import uuid as uuid_module
from unittest.mock import patch

import pytest

from app.models import Model
from app.services.server_startup import attach_aux_sources


def _aux_model(
    db,
    *,
    name: str,
    repo_id: str | None = "aux-org/aux-repo",
    source_file: str | None = None,
) -> Model:
    model = Model(
        name=name,
        path=f"/models/{repo_id or 'local'}/{name}",
        size_bytes=1000,
        architecture="clip",
        quantization="F16",
        source="huggingface",
        source_repo_id=repo_id,
        source_file=source_file or name,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


class TestAttachAuxSources:
    async def test_gufo_resolves_each_aux_key(self, db) -> None:
        mmproj = _aux_model(db, name="proj.gguf", repo_id="org/v")
        mtp = _aux_model(
            db,
            name="mtp.gguf",
            repo_id="org/mtp",
            source_file="mtp-00001-of-00002.gguf",
        )
        server_id = str(uuid_module.uuid4())
        payload = {
            "config": {
                "id": server_id,
                "engine": "gufo",
                "engine_options": {
                    "mmproj": mmproj.path,
                    "mtp_model": mtp.path,
                },
            }
        }

        await attach_aux_sources(payload)

        by_key = {s["key"]: s for s in payload["aux_sources"]}
        assert set(by_key) == {"mmproj", "mtp_model"}
        assert by_key["mmproj"]["path"] == mmproj.path
        assert by_key["mmproj"]["repo_id"] == "org/v"
        assert by_key["mmproj"]["filename"] == "proj.gguf"
        assert by_key["mmproj"]["filenames"] == ["proj.gguf"]
        assert by_key["mmproj"]["job_id"] == f"server-{server_id}-mmproj"
        assert by_key["mmproj"]["server_id"] == server_id
        # split-aware resolution for the mtp aux model
        assert by_key["mtp_model"]["repo_id"] == "org/mtp"
        assert by_key["mtp_model"]["filename"] == "mtp-00001-of-00002.gguf"
        assert by_key["mtp_model"]["job_id"] == f"server-{server_id}-mtp_model"

    async def test_missing_model_or_no_repo_is_skipped(self, db) -> None:
        no_repo = _aux_model(db, name="plain.gguf", repo_id=None)
        server_id = str(uuid_module.uuid4())
        payload = {
            "config": {
                "id": server_id,
                "engine": "gufo",
                "engine_options": {
                    "mmproj": no_repo.path,
                    "dflash_model": "/models/does/not/exist.gguf",
                },
            }
        }

        await attach_aux_sources(payload)

        assert payload["aux_sources"] == []

    async def test_non_gufo_engine_gets_empty_list(self, db) -> None:
        mmproj = _aux_model(db, name="proj.gguf", repo_id="org/v")
        payload = {
            "config": {
                "id": str(uuid_module.uuid4()),
                "engine": "llamacpp",
                "engine_options": {"mmproj": mmproj.path},
            }
        }

        await attach_aux_sources(payload)

        assert payload["aux_sources"] == []

    async def test_default_and_empty_values_skipped(self, db) -> None:
        payload = {
            "config": {
                "id": str(uuid_module.uuid4()),
                "engine": "gufo",
                "engine_options": {
                    "mmproj": "default",
                    "dspark_model": "   ",
                    "mtp_model": None,
                    "context": 8192,
                },
            }
        }

        await attach_aux_sources(payload)

        assert payload["aux_sources"] == []

    async def test_odd_payload_never_raises(self) -> None:
        for payload in (
            {},
            {"config": None},
            {"config": {}},
            {"config": {"engine": "gufo"}},
            {"config": {"engine": "gufo", "engine_options": None}},
            {"config": {"engine": "gufo", "engine_options": {"mmproj": 123}}},
        ):
            await attach_aux_sources(payload)
            assert payload["aux_sources"] == []

    async def test_idempotent_overwrites_existing(self, db) -> None:
        mmproj = _aux_model(db, name="proj.gguf", repo_id="org/v")
        payload = {
            "config": {
                "id": str(uuid_module.uuid4()),
                "engine": "gufo",
                "engine_options": {"mmproj": mmproj.path},
            },
            "aux_sources": [{"stale": True}],
        }

        await attach_aux_sources(payload)

        assert payload["aux_sources"] and "stale" not in payload["aux_sources"][0]


@pytest.mark.asyncio
async def test_initialize_server_attaches_aux_sources(db) -> None:
    """The prepare payload sent to the agent carries resolved aux_sources."""
    from app.models import Agent, ServerInstance
    from app.services import server_startup as ss

    agent = Agent(
        name=f"gufo-agent-{uuid_module.uuid4().hex[:8]}",
        host="127.0.0.1",
        port=8080,
        status="online",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    mmproj = _aux_model(db, name="vision.gguf", repo_id="org/v")
    model = Model(
        name="base.gguf",
        path="/models/org/base.gguf",
        size_bytes=1000,
        architecture="llama",
        quantization="Q4_K_M",
        source="huggingface",
        source_repo_id="org/base",
    )
    db.add(model)
    db.commit()
    instance = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias=f"gufo-alias-{model.id.hex[:8]}",
        process_command="gufo",
        engine="gufo",
        engine_options={"mmproj": mmproj.path},
        status="uninitialized",
    )
    db.add(instance)
    db.commit()
    db.refresh(instance)

    captured: dict = {}

    async def fake_send(_agent_id, _method, path, payload, **_kwargs):
        if path == "/servers/prepare":
            captured["prepare"] = payload
        if path.endswith(f"/servers/metadata/{instance.id}"):
            return {"data": [{"id": "base"}]}
        return {}

    request = {
        "config": {
            "id": str(instance.id),
            "engine": "gufo",
            "engine_options": {"mmproj": mmproj.path},
        }
    }
    with (
        patch.object(ss.agent_manager, "send_to_agent", new=fake_send),
        patch.object(ss, "_send_start_with_gate", new=_noop_start),
    ):
        await ss.initialize_server(str(agent.id), str(instance.id), request)

    assert captured["prepare"]["aux_sources"]
    assert captured["prepare"]["aux_sources"][0]["key"] == "mmproj"
    assert captured["prepare"]["aux_sources"][0]["path"] == mmproj.path


async def _noop_start(*_args, **_kwargs) -> dict:
    return {}
