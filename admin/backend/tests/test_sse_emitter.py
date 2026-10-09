"""SSEEmitter: monotonic sequence numbers, admin-owned response id
replacement on lifecycle frames, non-canonical event passthrough, [DONE],
and synthesized response.failed frames."""

import json

from app.services.sse import SSEEmitter

CLIENT_ID = "resp_" + "c" * 32
WRAPPED_ID = "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfdG95MTIz"


def parse_frame(frame: str) -> tuple[str, dict]:
    """Split an 'event: X\\ndata: {...}\\n\\n' frame into (type, payload)."""
    assert frame.endswith("\n\n")
    event_line, data_line = frame[:-2].split("\n")
    assert event_line.startswith("event: ")
    assert data_line.startswith("data: ")
    return event_line[len("event: ") :], json.loads(data_line[len("data: ") :])


def test_sequence_numbers_are_monotonic_and_overwrite_upstream() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    frames = [
        emitter.frame({"type": "response.created", "sequence_number": 99}),
        emitter.frame({"type": "response.output_text.delta", "sequence_number": 7}),
        emitter.frame({"type": "response.completed", "sequence_number": 123}),
    ]
    seqs = [parse_frame(f)[1]["sequence_number"] for f in frames]
    assert seqs == [0, 1, 2]


def test_response_id_replaced_on_all_lifecycle_events() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    lifecycle = (
        "response.created",
        "response.in_progress",
        "response.queued",
        "response.completed",
        "response.failed",
        "response.incomplete",
    )
    for event_type in lifecycle:
        frame = emitter.frame(
            {
                "type": event_type,
                "response": {"id": WRAPPED_ID, "status": "x", "output": []},
            }
        )
        parsed_type, payload = parse_frame(frame)
        assert parsed_type == event_type
        assert payload["response"]["id"] == CLIENT_ID
        # Other response fields untouched.
        assert payload["response"]["output"] == []


def test_non_lifecycle_and_noncanonical_events_pass_through_intact() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    delta = {
        "type": "response.reasoning_text.delta",
        "item_id": "rs_1",
        "delta": "thinking...",
        "sequence_number": 555,
    }
    frame = emitter.frame(dict(delta))
    parsed_type, payload = parse_frame(frame)
    assert parsed_type == "response.reasoning_text.delta"
    assert payload["item_id"] == "rs_1"
    assert payload["delta"] == "thinking..."
    # Only sequence_number reassigned; nothing else mutated.
    assert payload["sequence_number"] == 0

    # A message output_item.done with no response object: passes through.
    item_done = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"id": "msg_1", "type": "message", "role": "assistant"},
    }
    parsed_type, payload = parse_frame(emitter.frame(dict(item_done)))
    assert parsed_type == "response.output_item.done"
    assert payload["item"] == item_done["item"]


def test_completed_preserves_usage_and_output() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 4},
        "output_tokens_details": {"reasoning_tokens": 2},
    }
    output = [{"id": "msg_1", "type": "message", "role": "assistant"}]
    frame = emitter.frame(
        {
            "type": "response.completed",
            "response": {
                "id": WRAPPED_ID,
                "status": "completed",
                "output": output,
                "usage": usage,
            },
        }
    )
    _, payload = parse_frame(frame)
    assert payload["response"]["id"] == CLIENT_ID
    assert payload["response"]["usage"] == usage
    assert payload["response"]["output"] == output


def test_done_terminator() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    assert emitter.done() == "data: [DONE]\n\n"


def test_failed_synthesizes_spec_frames_with_our_id() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    # Consume a sequence number first so failed() continues counting.
    emitter.frame({"type": "response.created", "response": {"id": WRAPPED_ID}})
    error = {"type": "server_error", "code": "upstream_failed", "message": "boom"}
    frames = emitter.failed(error)
    assert len(frames) == 2

    failed_type, failed_payload = parse_frame(frames[0])
    assert failed_type == "response.failed"
    assert failed_payload["response"]["id"] == CLIENT_ID
    assert failed_payload["response"]["status"] == "failed"
    assert failed_payload["response"]["error"] == {**error, "param": None}
    assert failed_payload["sequence_number"] == 1

    error_type, error_payload = parse_frame(frames[1])
    assert error_type == "error"
    assert error_payload["error"] == {**error, "param": None}


def test_to_dict_handles_objects_with_model_dump() -> None:
    class FakeEvent:
        def model_dump(self, exclude_none: bool = False) -> dict:
            return {"type": "response.created", "response": {"id": WRAPPED_ID}}

    emitter = SSEEmitter(CLIENT_ID)
    frame = emitter.frame(FakeEvent())
    _, payload = parse_frame(frame)
    assert payload["response"]["id"] == CLIENT_ID
    assert payload["sequence_number"] == 0


def test_enum_type_is_unwrapped_to_dotted_value() -> None:
    """Regression: litellm returns some events with `type` as the
    ResponsesAPIStreamEvents (str, Enum) member, whose str() is the
    class-qualified name (e.g. 'ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA').
    The SSE `event:` line and the data payload must carry the dotted value."""
    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    emitter = SSEEmitter(CLIENT_ID)
    for member in (
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
    ):
        # str() of the member is NOT the dotted value (that's the bug).
        assert str(member).startswith("ResponsesAPIStreamEvents.")
        frame = emitter.frame({"type": member, "delta": "x"})
        parsed_type, payload = parse_frame(frame)
        assert parsed_type == member.value
        assert payload["type"] == member.value
        assert "ResponsesAPIStreamEvents" not in frame


def test_lifecycle_event_with_enum_type_still_gets_id_replaced() -> None:
    """Even if a lifecycle event arrives as an enum member, the id swap and
    dotted type must both hold."""
    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    emitter = SSEEmitter(CLIENT_ID)
    member = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    frame = emitter.frame(
        {"type": member, "response": {"id": WRAPPED_ID, "output": []}}
    )
    parsed_type, payload = parse_frame(frame)
    assert parsed_type == "response.completed"
    assert payload["response"]["id"] == CLIENT_ID


def test_position_fields_filled_when_provider_omits_them() -> None:
    """llama.cpp-style events carry no output_index/content_index; the
    emitter must fill them from tracked position."""
    emitter = SSEEmitter(CLIENT_ID)
    _, added = parse_frame(
        emitter.frame({"type": "response.output_item.added", "item": {"id": "msg_1"}})
    )
    _, part = parse_frame(
        emitter.frame(
            {
                "type": "response.content_part.added",
                "part": {"type": "output_text", "text": ""},
            }
        )
    )
    _, delta = parse_frame(
        emitter.frame({"type": "response.output_text.delta", "delta": "hi"})
    )
    _, done = parse_frame(
        emitter.frame({"type": "response.output_text.done", "text": "hi"})
    )
    assert added["output_index"] == 0
    assert part["output_index"] == 0 and part["content_index"] == 0
    assert part["item_id"] == "msg_1"
    assert delta["output_index"] == 0 and delta["content_index"] == 0
    assert delta["item_id"] == "msg_1"
    assert done["output_index"] == 0 and done["content_index"] == 0
    assert done["item_id"] == "msg_1"

    # Second item advances output_index and resets content_index.
    _, added2 = parse_frame(
        emitter.frame(
            {
                "type": "response.output_item.added",
                "item": {"id": "msg_2"},
            }
        )
    )
    _, delta2 = parse_frame(
        emitter.frame({"type": "response.output_text.delta", "delta": "x"})
    )
    assert added2["output_index"] == 1
    assert delta2["output_index"] == 1 and delta2["content_index"] == 0
    assert delta2["item_id"] == "msg_2"


def test_position_fields_never_overwrite_provider_values() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    frame = emitter.frame(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_x",
            "output_index": 5,
            "content_index": 2,
            "delta": "z",
        }
    )
    _, payload = parse_frame(frame)
    assert payload["item_id"] == "msg_x"
    assert payload["output_index"] == 5
    assert payload["content_index"] == 2


def test_reasoning_events_get_position_fields() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    _, added = parse_frame(
        emitter.frame(
            {
                "type": "response.output_item.added",
                "item": {"id": "rs_1", "type": "reasoning"},
            }
        )
    )
    assert added["output_index"] == 0
    _, part = parse_frame(
        emitter.frame(
            {
                "type": "response.reasoning_summary_part.added",
                "part": {"type": "summary_text", "text": ""},
            }
        )
    )
    _, summ = parse_frame(
        emitter.frame({"type": "response.reasoning_summary_text.delta", "delta": "s"})
    )
    assert part["item_id"] == "rs_1"
    assert part["output_index"] == 0 and part["summary_index"] == 0
    assert summ["item_id"] == "rs_1" and summ["summary_index"] == 0
    _, rdelta = parse_frame(
        emitter.frame({"type": "response.reasoning.delta", "delta": "r"})
    )
    assert rdelta["output_index"] == 0 and rdelta["content_index"] == 0
    assert rdelta["item_id"] == "rs_1"


def test_failed_accepts_normalized_response_base() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    error = {"type": "server_error", "code": "x", "message": "y"}
    frames = emitter.failed(
        error,
        {"status": "failed", "output": [], "tools": [], "usage": None},
    )
    _, payload = parse_frame(frames[0])
    assert payload["response"]["id"] == CLIENT_ID
    assert payload["response"]["status"] == "failed"
    assert payload["response"]["error"] == {**error, "param": None}
    assert payload["response"]["error"]["param"] is None
    assert payload["response"]["tools"] == []
    # Every synthesized frame carries a gapless sequence number.
    assert payload["sequence_number"] == 0
    _, err_frame = parse_frame(frames[1])
    assert err_frame["sequence_number"] == 1
    # Back-compat: without a base, the minimal frame still holds.
    _, minimal = parse_frame(
        emitter.failed({"type": "server_error", "code": "z", "message": ""})[0]
    )
    assert minimal["response"]["id"] == CLIENT_ID
    assert minimal["response"]["status"] == "failed"


def test_annotation_events_get_position_fields() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    emitter.frame(
        {
            "type": "response.output_item.added",
            "item": {"id": "msg_1"},
        }
    )
    emitter.frame(
        {
            "type": "response.content_part.added",
            "part": {"type": "output_text", "text": "x"},
        }
    )
    _, ann = parse_frame(
        emitter.frame(
            {
                "type": "response.output_text.annotation.added",
                "annotation": {"type": "url_citation", "url": "u"},
            }
        )
    )
    assert ann["item_id"] == "msg_1"
    assert ann["output_index"] == 0
    assert ann["content_index"] == 0
    assert ann["annotation_index"] == 0


def test_content_part_added_gets_annotations_array() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    frame = emitter.frame(
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        }
    )
    _, payload = parse_frame(frame)
    assert payload["part"]["annotations"] == []
    # Provider-sent annotations are never overwritten.
    frame = emitter.frame(
        {
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "x",
                "annotations": [{"type": "url_citation"}],
            },
        }
    )
    _, payload = parse_frame(frame)
    assert payload["part"]["annotations"] == [{"type": "url_citation"}]
