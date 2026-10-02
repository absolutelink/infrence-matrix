"""Unit tests for the gufo native /v1/responses translation helpers."""

import pytest

from app.api.routes.v1.responses import events as ev
from app.api.routes.v1.responses.schemas import CreateResponseBody
from app.api.routes.v1.responses.translator import (
    StreamState,
    TranslationError,
    decode_gufo_stream_event,
    gufo_buffered_to_output_items,
    gufo_native_payload,
    gufo_usage_to_spec,
    input_items_to_gufo_native,
)


def _req(**kwargs: object) -> CreateResponseBody:
    return CreateResponseBody.model_validate({"model": "qwen", "input": "hi", **kwargs})


class TestNativeInputItems:
    def test_string_input_becomes_user_message(self) -> None:
        native = input_items_to_gufo_native(_req(input="hello"), [])
        assert native == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ]

    def test_assistant_message_uses_output_text(self) -> None:
        native = input_items_to_gufo_native(
            _req(
                input=[
                    {"type": "message", "role": "assistant", "content": "prior answer"}
                ]
            ),
            [],
        )
        assert native[0]["content"] == [{"type": "output_text", "text": "prior answer"}]

    def test_image_part_preserved(self) -> None:
        native = input_items_to_gufo_native(
            _req(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "look:"},
                            {
                                "type": "input_image",
                                "image_url": "https://x/img.png",
                                "detail": "high",
                            },
                        ],
                    }
                ]
            ),
            [],
        )
        parts = native[0]["content"]
        assert parts[0] == {"type": "input_text", "text": "look:"}
        assert parts[1] == {
            "type": "input_image",
            "image_url": "https://x/img.png",
            "detail": "high",
        }

    def test_input_file_rejected(self) -> None:
        with pytest.raises(TranslationError):
            input_items_to_gufo_native(
                _req(
                    input=[
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_file", "file_url": "https://x/f"}
                            ],
                        }
                    ]
                ),
                [],
            )

    def test_item_reference_rejected(self) -> None:
        with pytest.raises(TranslationError):
            input_items_to_gufo_native(
                _req(input=[{"type": "item_reference", "id": "item_1"}]), []
            )

    def test_reasoning_replayed_as_summary(self) -> None:
        # Broker reasoning item carries text in content (reasoning_text); gufo
        # wants it folded into summary parts and rejects a content key.
        native = input_items_to_gufo_native(
            _req(input="go"),
            history_items=[
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "thinking"}],
                }
            ],
        )
        assert native[0] == {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thinking"}],
        }
        assert "content" not in native[0]
        assert "encrypted_content" not in native[0]

    def test_function_call_replayed_verbatim(self) -> None:
        native = input_items_to_gufo_native(
            _req(input="go"),
            history_items=[
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                }
            ],
        )
        assert native[0] == {
            "type": "function_call",
            "call_id": "c1",
            "name": "get_weather",
            "arguments": '{"city":"SF"}',
        }

    def test_function_call_requires_call_id(self) -> None:
        with pytest.raises(TranslationError):
            input_items_to_gufo_native(
                _req(input="go"),
                history_items=[
                    {"type": "function_call", "name": "f", "arguments": "{}"}
                ],
            )

    def test_function_output_string_and_parts(self) -> None:
        native = input_items_to_gufo_native(
            _req(input="go"),
            history_items=[
                {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
                {
                    "type": "function_call_output",
                    "call_id": "c2",
                    "output": [{"type": "input_text", "text": "42"}],
                },
            ],
        )
        assert native[0] == {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "sunny",
        }
        assert native[1]["output"] == [{"type": "input_text", "text": "42"}]

    def test_compaction_replayed_as_assistant(self) -> None:
        native = input_items_to_gufo_native(
            _req(input="go"),
            history_items=[
                {"type": "compaction", "id": "cmp_1", "encrypted_content": "sum"}
            ],
        )
        assert native[0] == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "sum"}],
        }


class TestNativePayload:
    def test_store_forced_false_and_no_background(self) -> None:
        payload = gufo_native_payload(_req(input="hi"), [], stream=False)
        assert payload["store"] is False
        assert "background" not in payload
        assert "model" not in payload  # agent injects served_model_name
        assert payload["stream"] is False

    def test_instructions_and_max_output(self) -> None:
        payload = gufo_native_payload(
            _req(input="hi", instructions="Be terse.", max_output_tokens=64),
            [],
            stream=False,
        )
        assert payload["instructions"] == "Be terse."
        assert payload["max_output_tokens"] == 64

    def test_sampling_forwarded(self) -> None:
        payload = gufo_native_payload(
            _req(
                input="hi",
                temperature=0.3,
                top_p=0.9,
                presence_penalty=0.1,
                frequency_penalty=0.2,
            ),
            [],
            stream=False,
        )
        assert payload["temperature"] == 0.3
        assert payload["top_p"] == 0.9
        assert payload["presence_penalty"] == 0.1
        assert payload["frequency_penalty"] == 0.2

    def test_reasoning_effort(self) -> None:
        payload = gufo_native_payload(
            _req(input="hi"), [], stream=False, reasoning_effort="high"
        )
        assert payload["reasoning"] == {"effort": "high"}

    def test_tools_flat_shape(self) -> None:
        payload = gufo_native_payload(
            _req(
                input="hi",
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    }
                ],
            ),
            [],
            stream=False,
        )
        assert payload["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            }
        ]
        assert payload["tool_choice"] == "auto"

    def test_named_function_tool_choice(self) -> None:
        payload = gufo_native_payload(
            _req(
                input="hi",
                tools=[{"type": "function", "name": "f"}],
                tool_choice={"type": "function", "name": "f"},
            ),
            [],
            stream=False,
        )
        assert payload["tool_choice"] == {"type": "function", "name": "f"}

    def test_allowed_tools_hard_errors(self) -> None:
        with pytest.raises(TranslationError):
            gufo_native_payload(
                _req(
                    input="hi",
                    tools=[{"type": "function", "name": "a"}],
                    tool_choice={
                        "type": "allowed_tools",
                        "tools": [{"type": "function", "name": "a"}],
                    },
                ),
                [],
                stream=False,
            )

    def test_text_json_object_format(self) -> None:
        payload = gufo_native_payload(
            _req(input="hi", text={"format": {"type": "json_object"}}), [], stream=False
        )
        assert payload["text"] == {"format": {"type": "json_object"}}

    def test_text_json_schema_format(self) -> None:
        payload = gufo_native_payload(
            _req(
                input="hi",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "out",
                        "schema": {"type": "object"},
                        "strict": True,
                    }
                },
            ),
            [],
            stream=False,
        )
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["name"] == "out"
        assert payload["text"]["format"]["schema"] == {"type": "object"}
        assert payload["text"]["format"]["strict"] is True


class TestUsageMapping:
    def test_maps_input_output_cached_reasoning(self) -> None:
        usage = gufo_usage_to_spec(
            {
                "input_tokens": 120,
                "output_tokens": 34,
                "total_tokens": 154,
                "input_tokens_details": {"cached_tokens": 90},
                "output_tokens_details": {"reasoning_tokens": 10},
            }
        )
        assert usage["input_tokens"] == 120
        assert usage["output_tokens"] == 34
        assert usage["total_tokens"] == 154
        assert usage["input_tokens_details"]["cached_tokens"] == 90
        assert usage["output_tokens_details"]["reasoning_tokens"] == 10

    def test_empty_usage_defaults_zero(self) -> None:
        usage = gufo_usage_to_spec({})
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0


class TestBufferedPassthrough:
    def test_items_pass_through_without_none(self) -> None:
        items = gufo_buffered_to_output_items(
            {
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hi",
                                "annotations": [],
                                "logprobs": None,
                            }
                        ],
                        "phase": None,
                    }
                ]
            }
        )
        assert items[0]["type"] == "message"
        assert "phase" not in items[0]
        assert "logprobs" not in items[0]["content"][0]


def _state() -> tuple[StreamState, ev.SSEmitter]:
    seq = ev.SSEmitter()
    return StreamState(seq, "qwen"), seq


class TestStreamDecoder:
    def test_text_delta(self) -> None:
        state, _ = _state()
        calls: dict = {}
        terminal, usage, incomplete = decode_gufo_stream_event(
            state, {"type": "response.output_text.delta", "delta": "Hello"}, calls
        )
        assert (terminal, usage, incomplete) == (False, None, False)
        state.finish_message()
        assert state.output_items[0]["content"][0]["text"] == "Hello"

    def test_reasoning_delta(self) -> None:
        state, _ = _state()
        decode_gufo_stream_event(
            state,
            {"type": "response.reasoning_summary_text.delta", "delta": "think"},
            {},
        )
        state.finish_reasoning()
        assert state.output_items[0]["type"] == "reasoning"

    def test_function_call_flow(self) -> None:
        state, _ = _state()
        calls: dict = {}
        decode_gufo_stream_event(
            state,
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                },
            },
            calls,
        )
        decode_gufo_stream_event(
            state,
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"city":',
            },
            calls,
        )
        decode_gufo_stream_event(
            state,
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '"SF"}',
            },
            calls,
        )
        state.finish_calls()
        assert state.output_items[0]["type"] == "function_call"
        assert state.output_items[0]["name"] == "get_weather"
        assert state.output_items[0]["arguments"] == '{"city":"SF"}'

    def test_completed_is_terminal_with_usage(self) -> None:
        state, _ = _state()
        terminal, usage, incomplete = decode_gufo_stream_event(
            state,
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 5, "output_tokens": 7},
                    "timings": {"prompt_ms": 1},
                },
            },
            {},
        )
        assert terminal is True
        assert incomplete is False
        assert usage["input_tokens"] == 5
        assert usage["timings"] == {"prompt_ms": 1}

    def test_incomplete_is_terminal_flagged(self) -> None:
        state, _ = _state()
        terminal, usage, incomplete = decode_gufo_stream_event(
            state,
            {
                "type": "response.incomplete",
                "response": {"usage": {"input_tokens": 5, "output_tokens": 7}},
            },
            {},
        )
        assert terminal is True
        assert incomplete is True

    def test_failed_raises(self) -> None:
        state, _ = _state()
        with pytest.raises(RuntimeError):
            decode_gufo_stream_event(
                state,
                {
                    "type": "response.failed",
                    "response": {"error": {"message": "boom"}},
                },
                {},
            )
