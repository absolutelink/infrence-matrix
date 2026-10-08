"""Port model overhaul (mock reference cutover): a two-backend mock agent serves
BOTH aliases on ONE ``/v1`` surface, routed by the request's ``model`` field.

The mock now builds its ``BackendRegistry`` with each handle carrying its
definition ``alias`` (no per-backend port). This drives the multiplexing app
(``create_provider_app(registry=...)``) directly — the same app
``AgentServer`` binds to ``PROVIDER_PORT`` — and asserts:

* a request for each hosted alias is served (200), an unknown alias 404s;
* ``/v1/models`` aggregates the model list across both running backends.
"""

from __future__ import annotations

from typing import Any

import httpx
from provider_lib.admin_client import AdminClient
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendHandle, BackendRegistry

from provider_mock.main import make_lifecycle


def _settings(tmp_path: Any) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="mock-mux",
        MACHINE_SECRET="tok",
        AGENT_ID="mock-mux-agent",
        ADMIN_BASE_URL="http://localhost:9999",
        PROVIDER_PORT=8081,
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
    )


async def _two_backend_app(tmp_path: Any) -> httpx.AsyncClient:
    client = AdminClient(_settings(tmp_path))
    registry = BackendRegistry()
    for iid, alias in (("i1", "alpha"), ("i2", "beta")):
        lifecycle = make_lifecycle(client, instance_id=iid)
        lifecycle.driver.model = alias  # what the mock advertises as its model
        await lifecycle.start()
        registry.add(BackendHandle(iid, lifecycle, None, alias=alias))
    app = create_provider_app(
        client.settings,
        BackendOverrides(provider_type="mock", version="dev", registry=registry),
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://provider"
    )


async def test_two_backends_served_on_one_port(tmp_path: Any) -> None:
    async with await _two_backend_app(tmp_path) as v1:
        for alias in ("alpha", "beta"):
            resp = await v1.post(
                "/v1/responses", json={"model": alias, "input": "go", "stream": True}
            )
            assert resp.status_code == 200, alias
            assert resp.headers["content-type"].startswith("text/event-stream")
        # An alias this agent does not host is a clean 404 (no fallback).
        assert (
            await v1.post("/v1/responses", json={"model": "gamma", "input": "go"})
        ).status_code == 404


async def test_models_aggregate_across_backends(tmp_path: Any) -> None:
    async with await _two_backend_app(tmp_path) as v1:
        body = (await v1.get("/v1/models")).json()
        ids = {m["id"] for m in body["data"]}
        assert ids == {"alpha", "beta"}
