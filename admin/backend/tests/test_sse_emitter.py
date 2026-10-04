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
    assert failed_payload["response"]["error"] == error
    assert failed_payload["sequence_number"] == 1

    error_type, error_payload = parse_frame(frames[1])
    assert error_type == "error"
    assert error_payload["error"] == error


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
