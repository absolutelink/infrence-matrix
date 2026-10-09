"""Reasoning-event conformance mapping for the halogen-flash driver.

The Flash backend emits non-spec OpenWebUI dual-name reasoning events
(``response.reasoning_text.delta``/``.done``). The driver's
``_stream_native_responses`` relay must rename them to the OpenResponses
spec types (``response.reasoning.delta``/``.done``), fill ``content_index``
when the backend omits it, leave the already spec-shaped
``response.reasoning_summary_text.*`` events untouched, and keep counting
reasoning-delta tokens for the usage override.
"""

from typing import Any

from conftest import NPU_UNAVAILABLE, make_settings

from provider_halogen_flash.driver import HalogenFlashBackend

# Expected reasoning tokens the driver counts from the fake's deltas:
# summary delta ("sum"*4 = 12 chars) + two reasoning_text deltas
# ("x"*40, "y"*40) → (12 + 40 + 40) // 4 == 23.
_EXPECTED_REASONING_TOKENS = (12 + 40 + 40) // 4


def _driver(tmp_path: Any, binary: str, artifacts: dict[str, str]):
    settings = make_settings(tmp_path, binary)
    return HalogenFlashBackend(
        settings,
        {
            "artifacts": {
                "model": {"path": artifacts["checkpoint"]},
                "tokenizer": {"path": artifacts["tokenizer"]},
            },
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )


async def test_reasoning_text_events_are_mapped_to_spec(
    tmp_path, fake_flash_binary, local_artifacts, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_REASONING", "1")
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
    finally:
        await driver.aclose()

    types = [e["type"] for e in events]
    # No non-spec reasoning_text.* events survive the relay.
    assert not any(t.startswith("response.reasoning_text.") for t in types)
    assert types.count("response.reasoning.delta") == 2
    assert types.count("response.reasoning.done") == 1

    for event in events:
        if event["type"] == "response.reasoning.delta":
            assert event["item_id"] == "rs_1"
            assert event["output_index"] == 0
            assert isinstance(event["content_index"], int)
            assert isinstance(event["delta"], str)
        elif event["type"] == "response.reasoning.done":
            assert event["item_id"] == "rs_1"
            assert event["output_index"] == 0
            assert isinstance(event["content_index"], int)
            assert isinstance(event["text"], str)


async def test_content_index_filled_when_absent(
    tmp_path, fake_flash_binary, local_artifacts, monkeypatch
) -> None:
    """The fake emits the second delta and the done event WITHOUT
    content_index; the driver must fill it with 0 (spec-required)."""
    monkeypatch.setenv("FAKE_REASONING", "1")
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
    finally:
        await driver.aclose()

    deltas = [e for e in events if e["type"] == "response.reasoning.delta"]
    done = [e for e in events if e["type"] == "response.reasoning.done"]
    # First delta carried content_index 0 from the backend; the second was
    # filled by the driver.
    assert deltas[0]["content_index"] == 0
    assert deltas[1]["content_index"] == 0
    assert deltas[1]["delta"] == "y" * 40
    assert done[0]["content_index"] == 0


async def test_summary_events_pass_through_unchanged(
    tmp_path, fake_flash_binary, local_artifacts, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_REASONING", "1")
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
    finally:
        await driver.aclose()

    summary_delta = [
        e for e in events if e["type"] == "response.reasoning_summary_text.delta"
    ]
    summary_done = [
        e for e in events if e["type"] == "response.reasoning_summary_text.done"
    ]
    assert len(summary_delta) == 1
    assert len(summary_done) == 1
    # Spec summary events keep their summary_index and are not renamed.
    assert summary_delta[0]["summary_index"] == 0
    assert summary_delta[0]["delta"] == "sum" * 4
    assert summary_done[0]["summary_index"] == 0
    assert "content_index" not in summary_delta[0]


async def test_usage_still_counts_reasoning_tokens(
    tmp_path, fake_flash_binary, local_artifacts, monkeypatch
) -> None:
    """The usage override must keep counting reasoning deltas after the
    rename (it now keys off the mapped spec type)."""
    monkeypatch.setenv("FAKE_REASONING", "1")
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
    finally:
        await driver.aclose()

    terminal = events[-1]
    assert terminal["type"] == "response.completed"
    usage = terminal["response"]["usage"]
    assert usage["output_tokens_details"]["reasoning_tokens"] == (
        _EXPECTED_REASONING_TOKENS
    )


async def test_reasoning_items_sanitized_on_item_events(
    tmp_path, fake_flash_binary, local_artifacts, monkeypatch
) -> None:
    """output_item.added/done reasoning items must drop a null
    encrypted_content and gain a summary array."""
    monkeypatch.setenv("FAKE_REASONING", "1")
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
    finally:
        await driver.aclose()

    item_events = [
        e
        for e in events
        if e["type"] in ("response.output_item.added", "response.output_item.done")
    ]
    assert len(item_events) == 2
    for event in item_events:
        item = event["item"]
        assert item["type"] == "reasoning"
        # null encrypted_content removed entirely.
        assert "encrypted_content" not in item
        # summary present (added when missing, kept when provided).
        assert isinstance(item["summary"], list)


async def test_reasoning_items_sanitized_in_terminal_output(
    tmp_path, fake_flash_binary, local_artifacts, monkeypatch
) -> None:
    """The terminal response.output reasoning items are sanitized: null
    encrypted_content dropped, missing summary defaulted to [], a string
    encrypted_content preserved, and non-reasoning items untouched."""
    monkeypatch.setenv("FAKE_REASONING", "1")
    driver = _driver(tmp_path, fake_flash_binary, local_artifacts)
    try:
        await driver.start()
        events = [e async for e in driver.stream_responses({"input": "go"})]
    finally:
        await driver.aclose()

    output = events[-1]["response"]["output"]
    by_id = {item["id"]: item for item in output}

    # rs_1: null encrypted_content removed, summary defaulted to [].
    rs1 = by_id["rs_1"]
    assert "encrypted_content" not in rs1
    assert rs1["summary"] == []

    # rs_2: string encrypted_content preserved, existing summary kept.
    rs2 = by_id["rs_2"]
    assert rs2["encrypted_content"] == "secret"
    assert rs2["summary"] == []

    # Non-reasoning message item untouched (no keys injected).
    msg = by_id["msg_1"]
    assert msg["type"] == "message"
    assert "summary" not in msg
    assert "encrypted_content" not in msg
