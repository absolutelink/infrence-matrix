"""Phase 22 S1: the mock emits a canned chat.completion tool_call turn.

Driver-level tests assert ``MockBackend.stream_chat_completions`` folds a
tool call into the chat.completion.chunk delta sequence when ``tools`` are
present (mirroring the responses ``_stream_tool_call`` path) and still emits
plain text when they are not.
"""

from typing import Any

from provider_mock.backend import MockBackend

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    }
]


async def _collect(driver: MockBackend, request: dict[str, Any]) -> list[dict]:
    return [chunk async for chunk in driver.stream_chat_completions(request)]


async def test_chat_with_tools_emits_tool_call() -> None:
    driver = MockBackend(model="mock-model")
    chunks = await _collect(
        driver, {"model": "mock-model", "messages": [], "tools": TOOLS}
    )
    assert chunks, "tool request must produce chunks"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)

    # A delta carries the tool call id/name, a later one the arguments.
    tool_deltas = [
        tc
        for c in chunks
        for ch in c["choices"]
        for tc in (ch.get("delta") or {}).get("tool_calls") or []
    ]
    assert tool_deltas, "expected tool_call deltas"
    assert tool_deltas[0].get("id")
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    assert any(tc["function"].get("arguments") for tc in tool_deltas)

    # Terminal choice carries finish_reason == "tool_calls" + usage.
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "tool_calls"
    assert last.get("usage", {}).get("prompt_tokens") == 1


async def test_chat_without_tools_emits_text() -> None:
    driver = MockBackend(model="mock-model")
    chunks = await _collect(driver, {"model": "mock-model", "messages": []})
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    content = "".join(
        (ch.get("delta") or {}).get("content") or ""
        for c in chunks
        for ch in c["choices"]
    )
    assert content.startswith("mock-token-")
    assert not any(
        (ch.get("delta") or {}).get("tool_calls") for c in chunks for ch in c["choices"]
    )
