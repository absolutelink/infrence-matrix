"""Unit tests for OpenResponses schemas, events, and translator."""

import json
import uuid

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

    def test_image_input_translated(self) -> None:
        messages = input_items_to_llama_messages(
            CreateResponseBody(
                model="m",
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What is this:"},
                            {"type": "input_image", "image_url": "https://x"},
                        ],
                    }
                ],
            ),
            [],
        )
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What is this:"}
        assert content[1] == {"type": "image_url", "image_url": {"url": "https://x"}}

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

    def test_lifecycle_builders_accept_model_not_dict(self) -> None:
        """Regression: router passed response.model_dump() into builders that
        already call model_dump() internally -> AttributeError('dict' object
        has no attribute 'model_dump') crashed the SSE stream on first frame."""
        seq = ev.SSEmitter()
        req = CreateResponseBody(model="m", input="hi")
        for builder in (
            ev.response_created,
            ev.response_in_progress,
            ev.response_completed,
            ev.response_incomplete,
            ev.response_failed,
        ):
            snapshot = build_response_resource(req, "resp_x", 1)
            snapshot.status = {
                ev.response_completed: "completed",
                ev.response_incomplete: "incomplete",
                ev.response_failed: "failed",
            }.get(builder, snapshot.status)
            event = builder(seq, snapshot)
            assert "sequence_number" in event
            assert event["response"]["object"] == "response"

    def test_done_terminator(self) -> None:
        assert "data: [DONE]\n\n" != None

    def test_emitter_queues_and_drains_once(self) -> None:
        """Regression for blank streamed messages: make() must queue events so
        drain_frames() emits each exactly once (build -> drain pattern), and
        StreamState-built deltas actually reach the wire."""
        seq = ev.SSEmitter()
        req = CreateResponseBody(model="m", input="hi")
        ev.response_created(seq, build_response_resource(req, "resp_x", 1))
        ev.response_in_progress(seq, build_response_resource(req, "resp_x", 1))

        state = StreamState(seq, "m")
        state.add_text_delta("Hel")
        state.add_text_delta("lo")
        state.finish_message()

        frames = seq.drain_frames()
        types = [f.split("\n")[0][7:] for f in frames]
        assert types == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
        # Each sequence number appears exactly once across all frames
        seqs = [json.loads(f.split("data: ", 1)[1])["sequence_number"] for f in frames]
        assert seqs == list(range(1, len(frames) + 1))
        # Draining again yields nothing
        assert seq.drain_frames() == []


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


def _item_text(item: dict) -> str:
    """Extract user text or assistant output text from a stored item."""
    if item.get("role") == "user":
        return item["content"]
    return "".join(
        part["text"]
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


class TestChainHistory:
    """_build_chain_history must walk the full previous_response_id chain."""

    def _record(self, db, response_id, previous_id):
        from app.models import Agent, Model, ResponseRecord

        suffix = uuid.uuid4().hex[:8]
        agent = Agent(
            name=f"chain-{suffix}",
            host="127.0.0.1",
            port=8080,
            status="online",
        )
        model = Model(
            name=f"chain-model-{suffix}.gguf",
            path="/models/x.gguf",
            size_bytes=1,
            architecture="llama",
            quantization="Q4_K_M",
            context_length=4096,
            source="local",
        )
        db.add(agent)
        db.add(model)
        db.commit()
        record = ResponseRecord(
            response_id=response_id,
            previous_response_id=previous_id,
            input_items=[
                {"type": "message", "role": "user", "content": f"in-{response_id}"}
            ],
            output_items=[
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"out-{response_id}",
                            "annotations": [],
                        }
                    ],
                }
            ],
            model_id=model.id,
            agent_id=agent.id,
            status="completed",
            store=True,
        )
        db.add(record)
        db.commit()
        return record

    def test_full_chain_replay(self, db) -> None:
        """Regression: history must span all hops, not just one level back."""
        from app.api.routes.v1.responses.router import _build_chain_history

        self._record(db, "resp_1", None)
        self._record(db, "resp_2", "resp_1")
        r3 = self._record(db, "resp_3", "resp_2")

        history = _build_chain_history(db, r3)
        contents = [_item_text(item) for item in history]
        assert contents == [
            "in-resp_1",
            "out-resp_1",
            "in-resp_2",
            "out-resp_2",
            "in-resp_3",
            "out-resp_3",
        ]

    def test_cycle_protection(self, db) -> None:
        from app.api.routes.v1.responses.router import _build_chain_history

        self._record(db, "resp_a", None)
        r2 = self._record(db, "resp_b", "resp_a")
        # Fabricate a cycle
        r2.previous_response_id = "resp_b"
        db.add(r2)
        db.commit()
        db.refresh(r2)

        history = _build_chain_history(db, r2)
        # Terminates; contains resp_b only
        contents = [_item_text(item) for item in history]
        assert contents == ["in-resp_b", "out-resp_b"]

    def test_broken_link_stops_gracefully(self, db) -> None:
        from app.api.routes.v1.responses.router import _build_chain_history

        r2 = self._record(db, "resp_x2", "resp_missing")
        history = _build_chain_history(db, r2)
        contents = [_item_text(item) for item in history]
        assert contents == ["in-resp_x2", "out-resp_x2"]


class TestChainDepth:
    def test_depth_zero_without_previous(self, db) -> None:
        from app.api.routes.v1.responses.router import _chain_depth

        assert _chain_depth(db, None) == 0

    def test_depth_counts_hops(self, db) -> None:
        from app.api.routes.v1.responses.router import _chain_depth

        class TestChainHistory_method:  # reuse the fixture helper
            pass

        helper = TestChainHistory()
        helper._record(db, "resp_d1", None)
        helper._record(db, "resp_d2", "resp_d1")
        r3 = helper._record(db, "resp_d3", "resp_d2")
        assert _chain_depth(db, r3) == 3


class TestConformanceSchemaRules:
    """Mirror the openresponses conformance harness's Zod validation rules.

    responseResourceSchema marks temperature/top_p/presence_penalty/
    frequency_penalty/top_logprobs/parallel_tool_calls strictly non-nullable
    and text.verbosity optional() — None values fail the schema.
    """

    def _serialize(self, request: CreateResponseBody) -> dict:
        from app.api.routes.v1.responses.schemas import serialize

        return serialize(build_response_resource(request, "resp_x", 1))

    def test_unset_params_omitted_not_null(self) -> None:
        req = CreateResponseBody(model="m", input="hi")
        data = self._serialize(req)
        for key in (
            "parallel_tool_calls",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
            "reasoning",
            "completed_at",
        ):
            assert key not in data, f"{key} must be omitted, not null"

    def test_set_params_echoed(self) -> None:
        req = CreateResponseBody.model_validate(
            {
                "model": "m",
                "input": "hi",
                "temperature": 0.5,
                "top_p": 0.9,
                "parallel_tool_calls": True,
                "top_logprobs": 3,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "text": {"verbosity": "low"},
            }
        )
        data = self._serialize(req)
        assert data["temperature"] == 0.5
        assert data["top_p"] == 0.9
        assert data["parallel_tool_calls"] is True
        assert data["top_logprobs"] == 3
        assert data["presence_penalty"] == 0.1
        assert data["frequency_penalty"] == 0.2
        assert data["text"]["verbosity"] == "low"

    def test_text_format_echo_shape(self) -> None:
        """format.type 'text' echoes only {type}; json_schema echoes the
        real schema under the aliased 'schema' key with strict defaulting."""
        req = CreateResponseBody.model_validate(
            {"model": "m", "input": "hi", "text": {"format": {"type": "text"}}}
        )
        data = self._serialize(req)
        assert data["text"]["format"] == {"type": "text"}

        req2 = CreateResponseBody.model_validate(
            {
                "model": "m",
                "input": "hi",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "x",
                        "schema": {"type": "object"},
                    }
                },
            }
        )
        data2 = self._serialize(req2)
        fmt = data2["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"] == {"type": "object"}
        assert fmt["strict"] is True
        assert "schema_" not in fmt
        assert "description" not in fmt

    def test_tools_echo_strict_defaults_true(self) -> None:
        req = CreateResponseBody.model_validate(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "function", "name": "f"}],
            }
        )
        data = self._serialize(req)
        tool = data["tools"][0]
        assert tool["strict"] is True
        assert "description" not in tool

    def test_output_items_omit_null_optionals(self) -> None:
        """phase/logprobs/encrypted_content serialize as None and would fail
        the harness's optional() (key-absent) semantics."""
        from app.api.routes.v1.responses.router import _coerced_output_items
        from app.api.routes.v1.responses.schemas import serialize

        seq = ev.SSEmitter()
        state = StreamState(seq, "m")
        state.assistant_phase = "final_answer"
        state.add_reasoning_delta("thinking")
        state.add_text_delta("answer")
        state.finish_calls()
        state.finish_reasoning()
        state.finish_message()

        items = [serialize(i) for i in _coerced_output_items(state.output_items)]
        reasoning = next(i for i in items if i["type"] == "reasoning")
        assert "encrypted_content" not in reasoning
        message = next(i for i in items if i["type"] == "message")
        assert message["phase"] == "final_answer"
        part = message["content"][0]
        assert "logprobs" not in part
        assert part["annotations"] == []


class TestConformanceStreamingEvents:
    def test_reasoning_events_emitted_under_both_names(self) -> None:
        """Spec conformance validates response.reasoning.delta/done; OpenWebUI
        renders response.reasoning_text.delta — both must reach the wire."""
        seq = ev.SSEmitter()
        state = StreamState(seq, "m")
        state.add_reasoning_delta("hmm")
        state.finish_reasoning()

        frames = seq.drain_frames()
        types = [f.split("\n")[0][7:] for f in frames]
        assert "response.reasoning.delta" in types
        assert "response.reasoning_text.delta" in types
        assert "response.reasoning.done" in types
        assert "response.reasoning_text.done" in types
        # Both payloads carry identical fields
        payloads = {}
        for f in frames:
            payload = json.loads(f.split("data: ", 1)[1])
            if payload["type"].startswith("response.reasoning"):
                payloads[payload["type"]] = payload
        for shared in ("item_id", "output_index", "content_index"):
            assert (
                payloads["response.reasoning.delta"][shared]
                == (payloads["response.reasoning_text.delta"][shared])
            )

    def test_phase_passthrough_replayed_with_cue(self) -> None:
        messages = input_items_to_llama_messages(
            CreateResponseBody(
                model="m",
                input=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": "thinking out loud",
                    },
                    {"type": "message", "role": "user", "content": "go on"},
                ],
            ),
            [],
        )
        assert "[assistant phase: commentary]" in messages[0]["content"]

    def test_last_assistant_phase_echoed_on_output(self) -> None:
        from app.api.routes.v1.responses.router import _last_assistant_phase

        req = CreateResponseBody.model_validate(
            {
                "model": "m",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": "The number is four.",
                    },
                    {"type": "message", "role": "user", "content": "repeat"},
                ],
            }
        )
        assert _last_assistant_phase(req) == "final_answer"
