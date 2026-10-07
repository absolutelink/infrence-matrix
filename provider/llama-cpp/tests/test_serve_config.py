"""Slice 6 serve guard: uvicorn binds each backend on a single ``port``.

`uvicorn.Config` has no `base_port` kwarg and takes no `**kwargs`, so a
`base_port=` there crashes the container at startup. Slice 6 moved per-backend
HTTP serving into :class:`provider_lib.serve.MultiPortServer` (one uvicorn
listener per hosted backend, each on its assignment port). This test (a) proves
`base_port` is invalid and `port` is valid, (b) asserts the shared server binds
with `port=` and never `base_port=`, and (c) asserts the package's `run_async`
delegates serving to `MultiPortServer` (no stray `base_port=`).
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


def test_multi_port_server_binds_on_port_not_base_port() -> None:
    from provider_lib import serve

    src = inspect.getsource(serve)
    assert "base_port=" not in src
    assert "port=port" in src


def test_run_async_serves_via_multi_port_server() -> None:
    import provider_llama_cpp.main as m

    src = inspect.getsource(m.run_async)
    assert "base_port=settings.PROVIDER_PORT" not in src
    assert "MultiPortServer" in src
