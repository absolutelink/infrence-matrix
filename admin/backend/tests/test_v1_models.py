"""GET /v1/models derived from enabled ProviderDefinitions (Phase 7)."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import ProviderDefinition


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _mk_def(session: Session, alias: str, **kwargs) -> ProviderDefinition:
    definition = ProviderDefinition(
        alias=alias,
        provider_type=kwargs.pop("provider_type", "mock"),
        registration_token=f"tok-{alias}-{uuid.uuid4().hex[:6]}",
        # Phase 14: default to an authored config (the listed flows assume
        # a configured definition; shells are covered by their own tests).
        backend_config=kwargs.pop("backend_config", {"model": {"file": "m.gguf"}}),
        **kwargs,
    )
    session.add(definition)
    session.commit()
    session.refresh(definition)
    return definition


def test_empty_list_when_none(client: TestClient) -> None:
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json() == {"object": "list", "data": []}


def test_enabled_listed_fields_and_metadata_merge(
    client: TestClient, session: Session
) -> None:
    definition = _mk_def(
        session,
        "mm-alpha",
        provider_type="llama-cpp",
        model_metadata={
            "context_length": 8192,
            "capabilities": ["tools"],
            "owned_by": "should-not-win",
        },
    )
    resp = client.get("/v1/models")
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["id"] == "mm-alpha"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "llama-cpp"
    assert entry["created"] == int(definition.created_at.timestamp())
    assert entry["context_length"] == 8192
    assert entry["capabilities"] == ["tools"]


def test_disabled_excluded(client: TestClient, session: Session) -> None:
    _mk_def(session, "mm-on")
    _mk_def(session, "mm-off", enabled=False)
    resp = client.get("/v1/models")
    ids = [d["id"] for d in resp.json()["data"]]
    assert ids == ["mm-on"]


def test_ordering_alias_asc(client: TestClient, session: Session) -> None:
    for alias in ("zz-model", "aa-model", "mm-model"):
        _mk_def(session, alias)
    resp = client.get("/v1/models")
    ids = [d["id"] for d in resp.json()["data"]]
    assert ids == ["aa-model", "mm-model", "zz-model"]


def test_shell_definition_not_listed(client, session: Session) -> None:
    """Phase 14: a shell (backend_config NULL) is invisible to /v1/models
    even when enabled, and appears once config is authored."""
    _mk_def(session, "shell-model", backend_config=None)
    resp = client.get("/v1/models")
    assert [m["id"] for m in resp.json()["data"]] == []

    shell = session.exec(_query_def("shell-model")).one()
    shell.backend_config = {}
    session.add(shell)
    session.commit()
    resp = client.get("/v1/models")
    assert [m["id"] for m in resp.json()["data"]] == ["shell-model"]


def _query_def(alias: str):
    from sqlmodel import select

    return select(ProviderDefinition).where(ProviderDefinition.alias == alias)
