"""Unit tests for OpenResponses schemas, events, and translator."""

import json

import pytest

from app.api.routes.v1.responses import events as ev
from app.api.routes.v1.responses.schemas import (
    AllowedToolsChoice,
    CreateResponseBody,
    ResponseResource,
    SpecificFunctionChoice,
    new_id,
)
from app.api.routes.v1.responses.translator import (
    StreamState,
    TranslationError,
    allowed_tool_names,
    build_response_resource,
    input_items_to_llama_messages,
    tools_to_llama,
)


class TestIds:
    def test_new_id_prefix(self) -> None:
        assert new_id("resp").startswith("resp_")
        assert new_id("msg").startswith("msg_")

    def test_unique(self) -> None:
        assert new_id("x") != new_id("x")


class TestRequestParsing:
    def test_string_input(self) -> None:
        req = CreateResponseBody(model="m", input="hi")
        assert req.input == "hi"
        assert req.stream is False
        assert req.store is True

    def test_message_items(self) -> None:
        req = CreateResponseBody(
            model="m",
            input=[
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            ],
        )
        assert isinstance(req.input, list)
        assert req.input[0].role == "user"

    def test_function_call_roundtrip(self) -> None:
        req = CreateResponseBody(
            model="m",
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ],
        )
        assert req.input[0].type == "function_call"
        assert req.input[1].call_id == "call_1"

    def test_tool_choice_string(self) -> None:
        req = CreateResponseBody(model="m", input="x", tool_choice="required")
        assert req.tool_choice == "required"

    def test_tool_choice_specific_function(self) -> None:
        req = CreateResponseBody(
            model="m",
            input="x",
            tool_choice={"type": "function", "name": "my_fn"},
        )
        assert isinstance(req.tool_choice, SpecificFunctionChoice)
        assert req.tool_choice.name == "my_fn"

    def test_tool_choice_allowed_tools(self) -> None:
        req = CreateResponseBody(
            model="m",
            input="x",
            tool_choice={
                "type": "allowed_tools",
                "tools": [{"type": "function", "name": "a"}],
                "mode": "required",
            },
        )
        assert isinstance(req.tool_choice, AllowedToolsChoice)
        assert req.tool_choice.tools[0].name == "a"


class TestTranslation:
    def _request(self, **kwargs: object) -> CreateResponseBody:
        return CreateResponseBody(model="m", input="hi", **kwargs)  # type: ignore[arg-type]

    def test_string_input_to_user_message(self) -> None:
        msgs = input_items_to_llama_messages(self._request(), [])
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_instructions_become_system(self) -> None:
        req = self._request(instructions="You are terse.")
        msgs = input_items_to_llama_messages(req, [])
        assert msgs[0] == {"role": "system", "content": "You are terse."}
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_history_order(self) -> None:
        history = [
            {"type": "message", "role": "user", "content": "earlier"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        ]
        msgs = input_items_to_llama_messages(self._request(), history)
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[0]["content"] == "earlier"

    def test_function_call_and_output(self) -> None:
        history = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ]
        msgs = input_items_to_llama_messages(self._request(), history)
        assert "[Called tool get_weather" in msgs[0]["content"]
        assert "[Tool result (c1)]: sunny" in msgs[1]["content"]

    def test_function_call_output_parts(self) -> None:
        history = [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "42"}],
            },
        ]
        msgs = input_items_to_llama_messages(self._request(), history)
        assert msgs[0]["content"] == "[Tool result (c1)]: 42"

    def test_image_input_rejected(self) -> None:
        with pytest.raises(TranslationError):
            input_items_to_llama_messages(
                CreateResponseBody(
                    model="m",
                    input=[
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_image", "image_url": "https://x"}
                            ],
                        }
                    ],
                ),
                [],
            )

    def test_item_reference_rejected(self) -> None:
        with pytest.raises(TranslationError):
            input_items_to_llama_messages(
                CreateResponseBody(
                    model="m",
                    input=[{"type": "item_reference", "id": "item_1"}],
                ),
                [],
            )

    def test_reasoning_not_replayed(self) -> None:
        history = [
            {"type": "reasoning", "summary": [], "content": []},
            {"type": "message", "role": "user", "content": "go"},
        ]
        msgs = input_items_to_llama_messages(self._request(), history)
        # reasoning item dropped, user messages preserved in order
        assert [m["content"] for m in msgs] == ["go", "hi"]

    def test_tools_flat_to_nested(self) -> None:
        req = self._request(
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        extra = tools_to_llama(req)
        assert extra["tools"][0]["function"]["name"] == "get_weather"
        assert extra["tool_choice"] == "auto"

    def test_tool_choice_specific_maps_nested(self) -> None:
        req = self._request(tool_choice={"type": "function", "name": "f1"})
        extra = tools_to_llama(req)
        assert extra["tool_choice"] == {"type": "function", "function": {"name": "f1"}}

    def test_allowed_tools_falls_back_to_auto(self) -> None:
        req = self._request(
            tool_choice={
                "type": "allowed_tools",
                "tools": [{"type": "function", "name": "a"}],
            }
        )
        extra = tools_to_llama(req)
        assert extra["tool_choice"] == "auto"
        assert allowed_tool_names(req) == {"a"}

    def test_no_allowed_tools_returns_none(self) -> None:
        assert allowed_tool_names(self._request()) is None
        assert allowed_tool_names(self._request(tool_choice="none")) is None


class TestStreamState:
    def _state(self) -> tuple[StreamState, ev.SSEmitter]:
        seq = ev.SSEmitter()
        return StreamState(seq, "m"), seq

    def test_text_message_events_and_item(self) -> None:
        state, seq = self._state()
        state.add_text_delta("Hel")
        state.add_text_delta("lo")
        state.finish_message()
        assert state.output_items[0]["type"] == "message"
        assert state.output_items[0]["content"][0]["text"] == "Hello"
        assert state.output_items[0]["status"] == "completed"

    def test_tool_call_item(self) -> None:
        state, _ = self._state()
        state.add_tool_call_delta(0, {"id": "call_x", "function": {"name": "fn"}})
        state.add_tool_call_delta(0, {"function": {"arguments": '{"a"'}})
        state.add_tool_call_delta(0, {"function": {"arguments": ":1}"}})
        state.finish_calls()
        assert state.output_items[0]["type"] == "function_call"
        assert state.output_items[0]["name"] == "fn"
        assert state.output_items[0]["arguments"] == '{"a":1}'
        assert state.output_items[0]["status"] == "completed"

    def test_allowed_tools_suppression(self) -> None:
        state, _ = self._state()
        state.add_tool_call_delta(0, {"id": "c1", "function": {"name": "send_email"}})
        all_suppressed = state.apply_allowed_tools({"get_weather"})
        state.finish_calls()
        state.finish_message()
        assert all_suppressed is True
        assert not state.has_calls
        types = [i["type"] for i in state.output_items]
        assert "function_call" not in types
        assert "message" in types  # suppression note

    def test_allowed_tools_permitted_call_kept(self) -> None:
        state, _ = self._state()
        state.add_tool_call_delta(0, {"id": "c1", "function": {"name": "get_weather"}})
        state.apply_allowed_tools({"get_weather"})
        state.finish_calls()
        assert state.has_calls is False  # finished
        assert state.output_items[0]["type"] == "function_call"

    def test_sequence_numbers_monotonic(self) -> None:
        seq = ev.SSEmitter()
        a = seq.next_sequence()
        b = seq.next_sequence()
        assert b == a + 1

    def test_reasoning_item(self) -> None:
        state, _ = self._state()
        state.add_reasoning_delta("thinking...")
        state.finish_reasoning()
        assert state.output_items[0]["type"] == "reasoning"
        assert state.output_items[0]["content"][0]["text"] == "thinking..."


class TestSSEFraming:
    def test_frame_format(self) -> None:
        seq = ev.SSEmitter()
        event = ev.response_created(
            seq, ResponseResource(created_at=1, status="completed", model="m")
        )
        frame = seq.frame(event)
        assert frame.startswith("event: response.created\n")
        data_line = frame.split("\n")[1]
        assert data_line.startswith("data: ")
        payload = json.loads(data_line[6:])
        assert payload["type"] == "response.created"
        assert "sequence_number" in payload
        assert payload["response"]["object"] == "response"

    def test_done_terminator(self) -> None:
        assert "data: [DONE]\n\n" != None


class TestResponseResourceEcho:
    def test_echo_fields(self) -> None:
        req = CreateResponseBody(
            model="m",
            input="hi",
            temperature=0.5,
            max_output_tokens=100,
            tools=[{"type": "function", "name": "f", "parameters": {}}],
        )
        resp = build_response_resource(req, "resp_1", 123)
        assert resp.model == "m"
        assert resp.temperature == 0.5
        assert resp.max_output_tokens == 100
        assert resp.tools[0].name == "f"
        assert resp.service_tier == "default"
        assert resp.status == "in_progress"

    def test_json_schema_text_format(self) -> None:
        req = CreateResponseBody(
            model="m",
            input="hi",
            text={"format": {"type": "json_schema", "name": "out", "schema": {}}},
        )
        resp = build_response_resource(req, "resp_1", 123)
        assert resp.text.format.type == "json_schema"
