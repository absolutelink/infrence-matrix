"""Serve guards for the mock package.

`uvicorn.Config` has no `base_port` kwarg and takes no `**kwargs`, so a
`base_port=` there crashes the container at startup. The port model overhaul
serves the agent's single ``/v1`` surface on ``PROVIDER_PORT`` via
:class:`provider_lib.serve.AgentServer` (one uvicorn listener, routing by model). This test (a) proves `base_port` is
invalid and `port` is valid, (b) asserts the shared serve module binds with
`port=` and never `base_port=`, and (c) asserts the mock's `run_async` delegates
serving to `AgentServer` (no stray `base_port=`).
"""

from __future__ import annotations

import inspect

import pytest
import uvicorn


def test_uvicorn_config_rejects_base_port() -> None:
    with pytest.raises(TypeError):
        uvicorn.Config(object(), host="0.0.0.0", base_port=8081)  # type: ignore[call-arg]
    # The correct single-backend serve kwarg is `port`.
    cfg = uvicorn.Config(object(), host="0.0.0.0", port=8081, log_level="info")
    assert cfg.port == 8081


def test_serve_binds_on_port_not_base_port() -> None:
    from provider_lib import serve

    src = inspect.getsource(serve)
    assert "base_port=" not in src
    assert "port=port" in src


def test_run_async_serves_via_agent_server() -> None:
    import provider_mock.main as m

    src = inspect.getsource(m.run_async)
    assert "base_port=settings.PROVIDER_PORT" not in src
    assert "AgentServer" in src
