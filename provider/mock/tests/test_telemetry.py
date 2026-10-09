"""Phase 23 S4: the mock emits rate + latency telemetry on terminal usage.

Both streaming surfaces must carry ``usage.completion_tokens_details``
(``prompt_per_second``/``predicted_per_second``/``prompt_time``/
``prediction_time``) so dev/smoke exercises the admin ``persist_turn``
mapping into ``TokenUsageSample``.
"""

from typing import Any

from provider_mock.backend import MockBackend

EXPECTED_DETAILS = {
    "prompt_per_second": 1412.21,
    "predicted_per_second": 36.2534,
    "prompt_time": 0.12,
    "prediction_time": 3.4,
}


async def _collect(aiter) -> list[dict[str, Any]]:
    return [frame async for frame in aiter]


async def test_responses_terminal_carries_completion_tokens_details() -> None:
    driver = MockBackend(model="mock-model")
    frames = await _collect(
        driver.stream_responses({"model": "mock-model", "input": "hi"})
    )
    completed = next(f for f in frames if f["type"] == "response.completed")
    details = completed["response"]["usage"]["completion_tokens_details"]
    assert details == EXPECTED_DETAILS


async def test_chat_final_usage_carries_completion_tokens_details() -> None:
    driver = MockBackend(model="mock-model")
    chunks = await _collect(
        driver.stream_chat_completions({"model": "mock-model", "messages": []})
    )
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    details = final["usage"]["completion_tokens_details"]
    assert details == EXPECTED_DETAILS
