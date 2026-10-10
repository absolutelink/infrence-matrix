"""Phase 25 S1 mock: a single backend serving MULTIPLE model names end to end.

The mock's ``build_registry`` / ``make_handle`` parse ``served_models`` via the
shared ``models_from_entry`` helper, so one lifecycle advertises and serves every
enabled name on the agent's single ``/v1`` surface. This drives the real mock
driver + app to prove:

* both served names route (chat + responses surfaces) and echo the requested
  name back;
* a disabled name 404s;
* ``/v1/models`` lists every enabled served name;
* the legacy single-model path (alias only, no ``served_models``) is unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from provider_lib.admin_client import AdminClient, RegistrationResult
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendRegistry
from provider_lib.wire import Frame

from provider_mock.main import build_registry, install_command_handlers


def _settings(tmp_path: Any) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="mock-mm",
        MACHINE_SECRET="tok",
        AGENT_ID="mock-mm-agent",
        ADMIN_BASE_URL="http://localhost:9999",
        PROVIDER_PORT=8081,
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
    )


def _result(definition: dict[str, Any]) -> RegistrationResult:
    return RegistrationResult(
        {
            "agent_id": "mock-mm-agent",
            "agent_secret": "secret",
            "machine": {},
            "backends": [{"instance_id": "i1", "definition": definition}],
        }
    )


async def _app_for(tmp_path: Any, definition: dict[str, Any]) -> httpx.AsyncClient:
    client = AdminClient(_settings(tmp_path))
    registry = build_registry(client, _result(definition))
    for handle in registry.handles():
        await handle.lifecycle.start()
    app = create_provider_app(
        client.settings,
        BackendOverrides(provider_type="mock", version="dev", registry=registry),
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://provider"
    )


async def test_multi_name_backend_serves_all_names(tmp_path: Any) -> None:
    definition = {
        "alias": "a",  # admin keeps alias == first enabled name
        "backend_config": {},
        "config_fingerprint": "fp",
        "served_models": [
            {"name": "a", "modality": "llm"},
            {"name": "b", "modality": "llm"},
            {"name": "c", "modality": "llm", "enabled": False},
        ],
    }
    async with await _app_for(tmp_path, definition) as v1:
        # Both enabled names route on the chat surface and echo the name back.
        for name in ("a", "b"):
            resp = await v1.post(
                "/v1/chat/completions",
                json={"model": name, "messages": [{"role": "user", "content": "x"}]},
            )
            assert resp.status_code == 200, name
            assert resp.json()["model"] == name
        # The responses surface too.
        r = await v1.post("/v1/responses", json={"model": "b", "input": "go"})
        assert r.status_code == 200
        assert r.json()["model"] == "b"
        # A disabled served name is a clean 404.
        assert (
            await v1.post("/v1/responses", json={"model": "c", "input": "go"})
        ).status_code == 404
        # /v1/models lists every enabled served name (not the disabled one).
        ids = {m["id"] for m in (await v1.get("/v1/models")).json()["data"]}
        assert ids == {"a", "b"}


async def test_legacy_single_model_unchanged(tmp_path: Any) -> None:
    # No served_models: the legacy trio synthesizes one spec; behavior identical.
    definition = {
        "alias": "solo",
        "modality": "llm",
        "backend_config": {},
        "config_fingerprint": "fp",
    }
    async with await _app_for(tmp_path, definition) as v1:
        resp = await v1.post(
            "/v1/chat/completions",
            json={"model": "solo", "messages": [{"role": "user", "content": "x"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "solo"
        assert (
            await v1.post("/v1/responses", json={"model": "other", "input": "go"})
        ).status_code == 404
        ids = {m["id"] for m in (await v1.get("/v1/models")).json()["data"]}
        assert ids == {"solo"}


async def _dispatch(client: AdminClient, type_: str, payload: dict) -> dict[str, Any]:
    await client._dispatch(Frame(type=type_, id="cmd-1", epoch=1, payload=payload))
    for _ in range(80):
        frame = await asyncio.wait_for(client._outgoing.get(), timeout=5.0)
        if frame.type == "ack" and frame.reply_to == "cmd-1":
            return frame.payload
    raise AssertionError("no ack produced")


async def test_make_handle_parses_served_models(tmp_path: Any) -> None:
    """The mock's ``make_handle`` (via an assignments push) uses the shared
    helper: served_models reach the handle AND the driver."""
    client = AdminClient(_settings(tmp_path))
    registry = BackendRegistry()
    install_command_handlers(client, registry)

    entry = {
        "instance_id": "i9",
        "alias": "a",
        "backend_config": {},
        "config_fingerprint": "fp-i9",
        "capacity": 1,
        "served_models": [{"name": "a"}, {"name": "b", "modality": "asr"}],
    }
    ack = await _dispatch(client, "agent.assignments.update", {"assignments": [entry]})
    assert ack["added"] == ["i9"]
    h = registry.get("i9")
    assert h is not None
    assert [m.name for m in h.models] == ["a", "b"]
    assert h.alias == "a"
    assert registry.resolve_by_model("b") is h
    # The mock driver adopted the list (set_models) and mirrors first name.
    assert h.lifecycle.driver.model == "a"
    assert h.lifecycle.driver.set_models_calls == 1
