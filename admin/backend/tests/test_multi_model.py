"""Phase 25 admin multi-model definitions.

Covers the canonical resolver (``resolve_models`` / ``resolve_served``),
``ProviderModel`` CRUD + cascade, the definition create/PATCH validation rules
(spec §6), the assignment payload + fingerprint, and per-name litellm
registration. The scheduler name->instance behavior lives in
``test_multi_model_scheduler.py`` and the v1 endpoint gates in
``test_multi_model_v1.py``.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.admin.providers import (
    compute_config_fingerprint,
    compute_definition_fingerprint,
)
from app.models import (
    ProviderDefinition,
    ProviderModel,
    ProviderType,
)
from app.services import assignments as asg
from app.services.served_models import (
    resolve_models,
    resolve_served,
    served_models_payload,
)
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

# A permissive multi-model type: serves every modality, requires a `slug` in
# each served model's per-model config (x-served-model-config-schema).
MM_SCHEMA = {
    "type": "object",
    "x-multi-model": True,
    "x-serves-modalities": ["llm", "embedding", "tts", "asr"],
    "x-served-model-config-schema": {
        "type": "object",
        "required": ["slug"],
        "additionalProperties": False,
        "properties": {"slug": {"type": "string", "minLength": 1}},
    },
}


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _seed_types(session: Session) -> None:
    from app.services.hashing import canonical_json_sha256

    for name, schema, multi in (
        ("mock", {"type": "object"}, False),
        ("mm", MM_SCHEMA, True),
    ):
        if (
            session.exec(select(ProviderType).where(ProviderType.name == name)).first()
            is None
        ):
            session.add(
                ProviderType(
                    name=name,
                    schema=schema,
                    schema_fingerprint=canonical_json_sha256(schema),
                    multi_model=multi,
                    serves_modalities=(
                        ["llm", "embedding", "tts", "asr"] if multi else ["llm"]
                    ),
                    status="active",
                )
            )
    session.commit()


def _add_models(session: Session, definition: ProviderDefinition, specs) -> None:
    for spec in specs:
        session.add(
            ProviderModel(
                definition_id=definition.id,
                name=spec["name"],
                modality=spec["modality"],
                backend_config=spec.get("backend_config", {}),
                enabled=spec.get("enabled", True),
            )
        )
    session.commit()


# ---------------------------------------------------------------------------
# resolve_models / resolve_served
# ---------------------------------------------------------------------------
def test_resolve_models_single_model_synthesizes(session: Session) -> None:
    d = make_definition(
        session,
        alias="solo",
        modality="llm",
        backend_config={"model": {"file": "m.gguf"}},
    )
    specs = resolve_models(session, d)
    assert len(specs) == 1
    assert specs[0].name == "solo"
    assert specs[0].modality == "llm"
    assert specs[0].backend_config == {"model": {"file": "m.gguf"}}
    assert specs[0].enabled is True


def test_resolve_models_multi_returns_rows_incl_disabled(session: Session) -> None:
    d = make_definition(session, alias="mmdef", provider_type="mm", modality="tts")
    _add_models(
        session,
        d,
        [
            {"name": "b-name", "modality": "asr", "backend_config": {"slug": "b"}},
            {"name": "a-name", "modality": "tts", "backend_config": {"slug": "a"}},
            {
                "name": "c-off",
                "modality": "tts",
                "backend_config": {"slug": "c"},
                "enabled": False,
            },
        ],
    )
    specs = resolve_models(session, d)
    # Deterministic order by name; disabled rows included so callers can gate.
    assert [s.name for s in specs] == ["a-name", "b-name", "c-off"]
    assert [s.enabled for s in specs] == [True, True, False]


def test_resolve_served_name_to_def_and_spec(session: Session) -> None:
    d = make_definition(session, alias="mmdef", provider_type="mm", modality="tts")
    _add_models(
        session,
        d,
        [
            {"name": "tts-x", "modality": "tts", "backend_config": {"slug": "x"}},
            {"name": "asr-y", "modality": "asr", "backend_config": {"slug": "y"}},
        ],
    )
    resolved = resolve_served(session, "asr-y")
    assert resolved is not None
    defn, spec = resolved
    assert defn.id == d.id
    assert spec.modality == "asr"
    assert spec.backend_config == {"slug": "y"}


def test_resolve_served_disabled_name_is_none(session: Session) -> None:
    d = make_definition(session, alias="mmdef", provider_type="mm", modality="tts")
    _add_models(
        session,
        d,
        [
            {
                "name": "off-z",
                "modality": "tts",
                "backend_config": {"slug": "z"},
                "enabled": False,
            }
        ],
    )
    assert resolve_served(session, "off-z") is None


def test_resolve_served_unknown_is_none(session: Session) -> None:
    assert resolve_served(session, "nope") is None


def test_resolve_served_single_model_alias(session: Session) -> None:
    d = make_definition(session, alias="solo2", modality="embedding")
    resolved = resolve_served(session, "solo2")
    assert resolved is not None
    defn, spec = resolved
    assert defn.id == d.id
    assert spec.modality == "embedding"


# ---------------------------------------------------------------------------
# ProviderModel CRUD + cascade
# ---------------------------------------------------------------------------
def test_provider_model_cascade_on_definition_delete(session: Session) -> None:
    d = make_definition(session, alias="mmdef", provider_type="mm", modality="tts")
    _add_models(
        session,
        d,
        [
            {"name": "c1", "modality": "tts", "backend_config": {"slug": "c1"}},
            {"name": "c2", "modality": "asr", "backend_config": {"slug": "c2"}},
        ],
    )
    assert len(session.exec(select(ProviderModel)).all()) == 2
    session.delete(d)
    session.commit()
    assert session.exec(select(ProviderModel)).all() == []


def test_provider_model_unique_name_index(session: Session) -> None:
    d1 = make_definition(session, alias="mmdef1", provider_type="mm", modality="tts")
    d2 = make_definition(session, alias="mmdef2", provider_type="mm", modality="tts")
    _add_models(session, d1, [{"name": "dup", "modality": "tts"}])
    from sqlalchemy.exc import IntegrityError

    session.add(ProviderModel(definition_id=d2.id, name="dup", modality="asr"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# ---------------------------------------------------------------------------
# definitions create API — served_models
# ---------------------------------------------------------------------------
def _mm_create(client: TestClient, **overrides) -> dict:
    body = {
        "alias": "ignored-alias",
        "provider_type": "mm",
        "modality": "tts",
        "backend_config": {"engine": {"foo": 1}},
        "served_models": [
            {"name": "tts-a", "modality": "tts", "backend_config": {"slug": "a"}},
            {"name": "asr-b", "modality": "asr", "backend_config": {"slug": "b"}},
        ],
    }
    body.update(overrides)
    resp = client.post("/admin/api/definitions", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_create_multi_model_happy(client: TestClient, session: Session) -> None:
    created = _mm_create(client)
    # alias synced to first enabled served name (submission order).
    assert created["alias"] == "tts-a"
    names = {m["name"] for m in created["served_models"]}
    assert names == {"tts-a", "asr-b"}
    rows = session.exec(
        select(ProviderModel).where(
            ProviderModel.definition_id == uuid.UUID(created["id"])
        )
    ).all()
    assert {r.name for r in rows} == {"tts-a", "asr-b"}


def test_create_served_models_rejected_on_non_multi_type(
    client: TestClient,
) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "solo-mm",
            "provider_type": "mock",
            "modality": "llm",
            "backend_config": {},
            "served_models": [{"name": "x", "modality": "llm"}],
        },
    )
    assert resp.status_code == 422
    assert "multi_model" in resp.json()["detail"]


def test_create_served_models_empty_rejected(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "mm-empty",
            "provider_type": "mm",
            "modality": "tts",
            "backend_config": {},
            "served_models": [],
        },
    )
    assert resp.status_code == 422


def test_create_served_models_dup_name_vs_alias(
    client: TestClient, session: Session
) -> None:
    make_definition(session, alias="taken", provider_type="mock", modality="llm")
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "mm-dup",
            "provider_type": "mm",
            "modality": "tts",
            "backend_config": {},
            "served_models": [
                {"name": "taken", "modality": "tts", "backend_config": {"slug": "t"}}
            ],
        },
    )
    assert resp.status_code == 422
    assert (
        "collides" in resp.json()["detail"] or "already in use" in resp.json()["detail"]
    )


def test_create_served_models_bad_modality(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "mm-badmod",
            "provider_type": "mm",
            "modality": "tts",
            "backend_config": {},
            "served_models": [{"name": "n1", "modality": "video"}],
        },
    )
    assert resp.status_code == 422


def test_create_served_models_missing_slug_per_model_schema(client: TestClient) -> None:
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "mm-badcfg",
            "provider_type": "mm",
            "modality": "tts",
            "backend_config": {},
            "served_models": [{"name": "n1", "modality": "tts", "backend_config": {}}],
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["error"] == "served_model_config_schema_validation_failed"


def test_create_single_model_unchanged(client: TestClient, session: Session) -> None:
    """No served_models sent -> behaves exactly as before (no ProviderModel rows)."""
    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "plain",
            "provider_type": "mock",
            "modality": "llm",
            "backend_config": {"model": {"file": "m.gguf"}},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["alias"] == "plain"
    # served_models present in the response as the one synthesized entry.
    assert [m["name"] for m in body["served_models"]] == ["plain"]
    rows = session.exec(
        select(ProviderModel).where(
            ProviderModel.definition_id == uuid.UUID(body["id"])
        )
    ).all()
    assert rows == []


# ---------------------------------------------------------------------------
# definitions PATCH — immutability + alias sync
# ---------------------------------------------------------------------------
def test_patch_served_names_immutable_while_attached(
    client: TestClient, session: Session
) -> None:
    created = _mm_create(client)
    did = uuid.UUID(created["id"])
    machine = get_or_create_machine(session, uid="mm-patch", total_vram=1000)
    agent = make_agent(session, machine, provider_type="mm", connected=True)
    make_instance(session, agent, session.get(ProviderDefinition, did))
    # Rename a served name while an instance is attached -> 409.
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={
            "served_models": [
                {"name": "tts-a", "modality": "tts", "backend_config": {"slug": "a"}},
                {
                    "name": "asr-renamed",
                    "modality": "asr",
                    "backend_config": {"slug": "b"},
                },
            ]
        },
    )
    assert resp.status_code == 409


def test_patch_served_toggle_enabled_syncs_alias(
    client: TestClient, session: Session
) -> None:
    created = _mm_create(client)
    # Disable tts-a (the current alias) -> alias syncs to the next enabled name.
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={
            "served_models": [
                {
                    "name": "tts-a",
                    "modality": "tts",
                    "backend_config": {"slug": "a"},
                    "enabled": False,
                },
                {"name": "asr-b", "modality": "asr", "backend_config": {"slug": "b"}},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["alias"] == "asr-b"
    session.expunge_all()
    d = session.get(ProviderDefinition, uuid.UUID(created["id"]))
    assert d.alias == "asr-b"
    row = session.exec(
        select(ProviderModel).where(ProviderModel.name == "tts-a")
    ).first()
    assert row is not None and row.enabled is False


def _attach_instance(session: Session, definition_id: str, uid: str) -> None:
    """Attach one backend on a connected agent to a definition (so the
    immutability gates fire)."""
    machine = get_or_create_machine(session, uid=uid, total_vram=1000)
    agent = make_agent(session, machine, provider_type="mm", connected=True)
    make_instance(session, agent, session.get(ProviderDefinition, uuid.UUID(definition_id)))


def test_patch_served_modality_immutable_while_attached(
    client: TestClient, session: Session
) -> None:
    """MEDIUM-2 regression: a served name's modality is immutable while
    instances are attached (409) — flipping it would desync the admin routing
    key from the live engine (spec §6.3 / §3)."""
    created = _mm_create(client)
    _attach_instance(session, created["id"], "mm-mod-imm")
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={
            "served_models": [
                # Same names, but tts-a flipped to asr.
                {"name": "tts-a", "modality": "asr", "backend_config": {"slug": "a"}},
                {"name": "asr-b", "modality": "asr", "backend_config": {"slug": "b"}},
            ]
        },
    )
    assert resp.status_code == 409
    assert "modality" in resp.json()["detail"]


def test_patch_served_enabled_and_config_allowed_while_attached(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Same names + modalities but toggled enabled + changed per-model config is
    allowed while attached (spec §6.3)."""
    from app.services.connection_manager import manager
    from app.services.wire import Frame

    async def ok_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": True, "detail": {}})

    monkeypatch.setattr(manager, "send_command", ok_send_command)

    created = _mm_create(client)
    _attach_instance(session, created["id"], "mm-edit-imm")
    resp = client.patch(
        f"/admin/api/definitions/{created['id']}",
        json={
            "served_models": [
                {
                    "name": "tts-a",
                    "modality": "tts",
                    "backend_config": {"slug": "a2"},
                    "enabled": False,
                },
                {"name": "asr-b", "modality": "asr", "backend_config": {"slug": "b2"}},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    session.expunge_all()
    row = session.exec(select(ProviderModel).where(ProviderModel.name == "tts-a")).first()
    assert row is not None and row.enabled is False
    assert row.backend_config == {"slug": "a2"}


# ---------------------------------------------------------------------------
# assignment payload + fingerprint
# ---------------------------------------------------------------------------
def test_build_assignment_entry_carries_served_models(session: Session) -> None:
    d = make_definition(session, alias="mmdef", provider_type="mm", modality="tts")
    _add_models(
        session,
        d,
        [
            {"name": "tts-a", "modality": "tts", "backend_config": {"slug": "a"}},
            {"name": "asr-b", "modality": "asr", "backend_config": {"slug": "b"}},
        ],
    )
    machine = get_or_create_machine(session, uid="mm-asg", total_vram=1000)
    agent = make_agent(session, machine, provider_type="mm", connected=True)
    inst = make_instance(session, agent, d)
    entry = asg.build_assignment_entry(session, inst, d)
    assert entry["served_models"] == served_models_payload(session, d)
    assert {m["name"] for m in entry["served_models"]} == {"tts-a", "asr-b"}


def test_fingerprint_single_model_matches_legacy(session: Session) -> None:
    d = make_definition(
        session, alias="solo-fp", backend_config={"model": {"file": "m.gguf"}}
    )
    assert compute_definition_fingerprint(session, d) == compute_config_fingerprint(
        d.backend_config
    )


def test_fingerprint_changes_on_per_model_edit(session: Session) -> None:
    d = make_definition(session, alias="mmdef", provider_type="mm", modality="tts")
    _add_models(
        session,
        d,
        [
            {"name": "tts-a", "modality": "tts", "backend_config": {"slug": "a"}},
            {"name": "asr-b", "modality": "asr", "backend_config": {"slug": "b"}},
        ],
    )
    fp_before = compute_definition_fingerprint(session, d)
    # Change ONE per-model config; the shared backend_config is untouched.
    row = session.exec(
        select(ProviderModel).where(ProviderModel.name == "asr-b")
    ).first()
    row.backend_config = {"slug": "b2"}
    session.add(row)
    session.commit()
    fp_after = compute_definition_fingerprint(session, d)
    assert fp_before != fp_after
    # The legacy backend_config fingerprint is unchanged (shared config same).
    assert compute_config_fingerprint(d.backend_config) == compute_config_fingerprint(
        {}
    )


# ---------------------------------------------------------------------------
# per-name litellm registration
# ---------------------------------------------------------------------------
def test_alias_registry_registers_llm_embedding_skips_audio(
    client: TestClient, monkeypatch
) -> None:
    from app.api.admin import definitions as definitions_mod

    warmed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        definitions_mod.alias_registry,
        "ensure_registered",
        lambda alias, mode="chat": warmed.append((alias, mode)),
    )
    client.post(
        "/admin/api/definitions",
        json={
            "alias": "x",
            "provider_type": "mm",
            "modality": "llm",
            "backend_config": {},
            "served_models": [
                {"name": "chat-1", "modality": "llm", "backend_config": {"slug": "c"}},
                {
                    "name": "emb-1",
                    "modality": "embedding",
                    "backend_config": {"slug": "e"},
                },
                {"name": "tts-1", "modality": "tts", "backend_config": {"slug": "t"}},
                {"name": "asr-1", "modality": "asr", "backend_config": {"slug": "a"}},
            ],
        },
    )
    registered = dict(warmed)
    assert registered.get("chat-1") == "chat"
    assert registered.get("emb-1") == "embedding"
    assert "tts-1" not in registered
    assert "asr-1" not in registered
