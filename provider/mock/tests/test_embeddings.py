"""Phase 18 slice 6: the mock serves deterministic fake embeddings.

Driver-level tests assert the spec shape + determinism of
``MockBackend.embeddings``; an app-level test proves the inherited
``POST /v1/embeddings`` route (Slice 4) reaches the driver end to end.
"""

from typing import Any

import httpx
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings

from provider_mock.backend import MockBackend
from provider_mock.main import build_app


def _settings() -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="mock-embed",
        MACHINE_SECRET="tok",
        AGENT_ID="mock-embed-agent",
        ADMIN_BASE_URL="http://127.0.0.1:1",
        PROVIDER_PORT=8081,
        CACHE_DIR="/tmp/mock_embed_test_cache",
        MODELS_DIR="/tmp/mock_embed_test_models",
    )


async def test_embeddings_single_string_shape() -> None:
    driver = MockBackend(model="mock-model")
    result = await driver.embeddings({"input": "hello world"})
    assert result["object"] == "list"
    assert result["model"] == "mock-model"
    data = result["data"]
    assert len(data) == 1
    assert data[0]["object"] == "embedding"
    assert data[0]["index"] == 0
    vec = data[0]["embedding"]
    assert isinstance(vec, list)
    assert len(vec) == MockBackend.EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vec)
    assert all(-1.0 <= x <= 1.0 for x in vec)
    assert result["usage"]["prompt_tokens"] >= 1
    assert result["usage"]["total_tokens"] == result["usage"]["prompt_tokens"]
    assert driver.embeddings_calls == 1


async def test_embeddings_is_deterministic() -> None:
    driver = MockBackend()
    first = await driver.embeddings({"input": "same text"})
    second = await driver.embeddings({"input": "same text"})
    assert first["data"][0]["embedding"] == second["data"][0]["embedding"]
    different = await driver.embeddings({"input": "other text"})
    assert different["data"][0]["embedding"] != first["data"][0]["embedding"]


async def test_embeddings_list_input() -> None:
    driver = MockBackend()
    result = await driver.embeddings({"input": ["a", "b"]})
    data = result["data"]
    assert len(data) == 2
    assert [item["index"] for item in data] == [0, 1]
    assert all(len(item["embedding"]) == MockBackend.EMBEDDING_DIM for item in data)
    # Distinct inputs -> distinct vectors.
    assert data[0]["embedding"] != data[1]["embedding"]
    # Token estimate sums over inputs.
    assert result["usage"]["prompt_tokens"] >= 2


async def test_embeddings_empty_input_is_one_empty_text() -> None:
    driver = MockBackend()
    for req in ({"input": ""}, {"input": []}, {}):
        result = await driver.embeddings(req)
        assert len(result["data"]) == 1
        assert result["usage"]["prompt_tokens"] >= 1


async def test_embeddings_uses_request_model() -> None:
    driver = MockBackend(model="mock-model")
    result = await driver.embeddings({"input": "hi", "model": "alias-embed"})
    assert result["model"] == "alias-embed"


async def test_app_v1_embeddings_reaches_driver() -> None:
    driver = MockBackend(model="mock-model")
    lifecycle = BackendLifecycle(driver, capacity=1)
    await lifecycle.start()
    app = build_app(lifecycle, _settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://provider"
    ) as client:
        resp = await client.post(
            "/v1/embeddings", json={"input": "hello world", "model": "mock-model"}
        )
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert len(body["data"][0]["embedding"]) == MockBackend.EMBEDDING_DIM
        assert body["usage"]["prompt_tokens"] >= 1
    assert driver.embeddings_calls == 1
    # Slot released after the non-streaming call.
    assert lifecycle.in_flight == 0
