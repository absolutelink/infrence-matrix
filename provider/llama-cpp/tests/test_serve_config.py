"""B1 guard: the provider serves uvicorn on a single `port`.

`uvicorn.Config` has no `base_port` kwarg and takes no `**kwargs`, so a
`base_port=` there crashes the container at startup. This test (a) proves
`base_port` is invalid and `port` is valid, and (b) asserts the package's
`run_async` source uses `port=settings.PROVIDER_PORT` and never
`base_port=settings.PROVIDER_PORT`.
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


def test_run_async_serves_on_port_not_base_port() -> None:
    import provider_llama_cpp.main as m

    src = inspect.getsource(m.run_async)
    assert "base_port=settings.PROVIDER_PORT" not in src
    assert "port=settings.PROVIDER_PORT" in src
