"""Phase 12 registration-body tests: AdminClient.register() sends the
provider schema and surfaces the schema-gate 409s distinctly."""

import json
from typing import Any

import pytest

from provider_lib.admin_client import (
    AdminClient,
    RegistrationError,
    SchemaPendingError,
)
from provider_lib.config import ProviderSettings

OK_BODY = {
    "instance_id": "11111111-2222-3333-4444-555555555555",
    "instance_secret": "s3cret",
    "machine": {"uid": "m1"},
    "provider_definition": {"alias": "a1", "config_fingerprint": "fp"},
}

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"stream": {"type": "object"}},
}


class _Resp:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> Any:
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; records the posted body."""

    last: dict[str, Any] = {}
    response: _Resp = _Resp(200, OK_BODY)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, json: dict[str, Any] | None = None) -> _Resp:  # noqa: A002
        _FakeAsyncClient.last = {"url": url, "body": json or {}}
        return _FakeAsyncClient.response


@pytest.fixture
def client(tmp_path: Any, monkeypatch: Any) -> AdminClient:
    settings = ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="m1",
        PROVIDER_REGISTRATION_TOKEN="tok",
        ADMIN_BASE_URL="http://admin:8000",
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
    )
    monkeypatch.setattr("provider_lib.admin_client.httpx.AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.last = {}
    _FakeAsyncClient.response = _Resp(200, OK_BODY)
    return AdminClient(settings)


async def _register(client: AdminClient, **kwargs: Any) -> Any:
    return await client.register(
        provider_type="mock",
        version="dev",
        port=8081,
        hardware={},
        **kwargs,
    )


async def test_register_includes_schema_in_body(client: AdminClient) -> None:
    result = await _register(client, schema=SCHEMA)
    assert result.instance_id == OK_BODY["instance_id"]
    body = _FakeAsyncClient.last["body"]
    assert body["schema"] == SCHEMA
    assert body["provider_type"] == "mock"


async def test_register_omits_schema_when_none(client: AdminClient) -> None:
    await _register(client)
    assert "schema" not in _FakeAsyncClient.last["body"]


async def test_schema_pending_surfaces_structured_error(client: AdminClient) -> None:
    detail = {
        "error": "schema_pending",
        "provider_type": "mock",
        "committed_fingerprint": "c0ffee",
        "pending_fingerprint": "beef00",
        "voted": ["id-1"],
        "waiting_on": ["id-2", "id-3"],
    }
    _FakeAsyncClient.response = _Resp(409, {"detail": detail})
    with pytest.raises(SchemaPendingError) as exc_info:
        await _register(client, schema=SCHEMA)
    err = exc_info.value
    assert err.status_code == 409
    assert err.error == "schema_pending"
    assert err.voted == ["id-1"]
    assert err.waiting_on == ["id-2", "id-3"]
    # Readable operator message per the Phase 12 requirement.
    assert "waiting for all agents to update schema" in str(err)
    assert "1 voted, waiting on 2" in str(err)
    assert "['id-2', 'id-3']" in str(err)
    # Still a RegistrationError so run_forever-style retry keeps working.
    assert isinstance(err, RegistrationError)


async def test_schema_conflict_surfaces_structured_error(client: AdminClient) -> None:
    detail = {
        "error": "schema_conflict",
        "provider_type": "mock",
        "committed_fingerprint": "c0ffee",
        "pending_fingerprint": "beef00",
        "voted": ["id-1"],
        "waiting_on": ["id-2"],
    }
    _FakeAsyncClient.response = _Resp(409, {"detail": detail})
    with pytest.raises(SchemaPendingError) as exc_info:
        await _register(client, schema=SCHEMA)
    assert exc_info.value.error == "schema_conflict"
    assert "schema conflict: another schema is pending consensus" in str(exc_info.value)
    assert "waiting for all agents" not in str(exc_info.value)


async def test_unknown_structured_409_not_mislabeled(client: AdminClient) -> None:
    detail = {
        "error": "schema_zombie",
        "provider_type": "mock",
        "voted": [],
        "waiting_on": [],
    }
    _FakeAsyncClient.response = _Resp(409, {"detail": detail})
    with pytest.raises(SchemaPendingError) as exc_info:
        await _register(client, schema=SCHEMA)
    assert exc_info.value.error == "schema_zombie"
    assert str(exc_info.value) == "registration refused: schema_zombie"
    assert "conflict" not in str(exc_info.value)
    assert "waiting for all agents" not in str(exc_info.value)


async def test_missing_error_key_not_defaulted_to_pending(client: AdminClient) -> None:
    detail = {"provider_type": "mock", "voted": ["id-1"]}
    _FakeAsyncClient.response = _Resp(409, {"detail": detail})
    with pytest.raises(SchemaPendingError) as exc_info:
        await _register(client, schema=SCHEMA)
    assert exc_info.value.error == ""
    assert str(exc_info.value) == "registration refused: unknown schema error"


async def test_plain_409_stays_registration_error(client: AdminClient) -> None:
    _FakeAsyncClient.response = _Resp(409, {"detail": "version mismatch"})
    with pytest.raises(RegistrationError) as exc_info:
        await _register(client, schema=SCHEMA)
    assert not isinstance(exc_info.value, SchemaPendingError)
    assert "version mismatch" in str(exc_info.value)
