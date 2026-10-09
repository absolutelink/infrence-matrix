"""Route integration for POST /v1/responses with litellm monkeypatched.

A connected provider instance is seeded directly in the UTF8 test DB
(websocket_connected=True, backend_status=running) so the real
InferenceScheduler admits against it. ``litellm.aresponses`` is replaced
with a fake async iterator emitting a realistic OpenResponses event
sequence (including litellm's affinity-wrapped response id), letting us
assert the full admin-side contract: our resp_ id on every frame,
monotonic sequence numbers, [DONE], persistence, slot release, the
response.failed path, and previous_response_id reconstruction.
"""

import asyncio
import copy
import json
import uuid
from typing import Any

import litellm
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.db import engine
from app.models import ProviderDefinition, ProviderInstance, ResponseRecord
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

CLIENT_ID_PREFIX = "resp_"
WRAPPED_ID = (
    "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZA"
    "pOb25lO3Jlc3BvbnNlX2lkOnJlc3BfdG95MTIz"
)


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def seed_instance(
    session: Session,
    *,
    alias: str,
    machine_uid: str,
    backend_status: str = "running",
    capacity: int = 1,
    vram_required: int = 0,
    enabled: bool = True,
) -> ProviderInstance:
    machine = get_or_create_machine(session, uid=machine_uid, host="127.0.0.1")
    agent = make_agent(session, machine, connected=True)
    definition = make_definition(
        session,
        alias=alias,
        capacity=capacity,
        vram_required=vram_required,
        enabled=enabled,
        backend_config={"model": {"file": "m.gguf"}},
    )
    instance = make_instance(session, agent, definition, backend_status=backend_status)
    return instance


def completed_event(output: list[dict], usage: dict) -> dict:
    return {
        "type": "response.completed",
        "response": {
            "id": WRAPPED_ID,
            "object": "response",
            "status": "completed",
            "output": output,
            "usage": usage,
        },
    }


def make_fake_stream(events: list[dict], captured: dict[str, Any]):
    """Build a fake ``litellm.aresponses``: awaitable returning an async
    iterator over ``events``, recording the kwargs it was called with."""

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)

        async def gen():
            for event in events:
                yield event

        return gen()

    return fake_aresponses


def parse_sse(text: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        event_type = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if data == "[DONE]":
            frames.append(("__done__", {}))
        elif event_type is not None:
            frames.append((event_type, json.loads(data)))
    return frames


# ---------------------------------------------------------------------------
# Happy path streaming
# ---------------------------------------------------------------------------
def test_stream_happy_path(client: TestClient, session: Session, monkeypatch) -> None:
    instance = seed_instance(session, alias="rt-a", machine_uid="rt-m")

    output = [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello there"}],
        }
    ]
    usage = {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 3},
        "output_tokens_details": {"reasoning_tokens": 1},
    }
    events = [
        {
            "type": "response.created",
            "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
        },
        {
            "type": "response.in_progress",
            "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello ",
        },
        {
            "type": "response.reasoning_text.delta",
            "item_id": "rs_1",
            "delta": "thinking",
        },
        completed_event(output, usage),
    ]
    captured: dict[str, Any] = {}
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, captured))

    resp = client.post(
        "/v1/responses",
        json={"model": "rt-a", "input": "hi", "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse(resp.text)

    assert frames[-1][0] == "__done__"
    body_frames = frames[:-1]

    # Our resp_ id on every lifecycle frame, never litellm's wrapped id.
    for _event_type, payload in body_frames:
        if "response" in payload and isinstance(payload["response"], dict):
            assert payload["response"]["id"].startswith(CLIENT_ID_PREFIX)
            assert payload["response"]["id"] != WRAPPED_ID

    # Monotonic sequence numbers starting at 0.
    seqs = [p["sequence_number"] for _, p in body_frames]
    assert seqs == list(range(len(body_frames)))

    # Event types preserved, including non-canonical reasoning passthrough.
    types = [t for t, _ in body_frames]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_text.delta",
        "response.reasoning_text.delta",
        "response.completed",
    ]
    reasoning = body_frames[3][1]
    assert reasoning["delta"] == "thinking"

    # litellm was called with our base_url + /v1 and no previous_response_id.
    assert captured["api_base"] == "http://127.0.0.1:8081/v1"
    assert captured["custom_llm_provider"] == "openai"
    assert captured["stream"] is True
    assert "previous_response_id" not in captured

    # Persisted ResponseRecord.
    client_id = body_frames[0][1]["response"]["id"]
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == client_id)
        .first()
    )
    assert record is not None
    assert record.status == "completed"
    assert record.output_items == output
    assert record.provider_definition_id == instance.provider_definition_id
    assert record.provider_instance_id == instance.id
    assert record.input_tokens == 11
    assert record.output_tokens == 4
    assert record.total_tokens == 15
    assert record.parameters["litellm_response_id"] == WRAPPED_ID

    # TokenUsageSample with cached tokens.
    from app.models import TokenUsageSample

    sample = (
        session.query(TokenUsageSample)
        .filter(TokenUsageSample.provider_instance_id == instance.id)
        .first()
    )
    assert sample is not None
    assert sample.prompt_tokens == 11
    assert sample.cached_tokens == 3
    assert sample.completion_tokens == 4

    # Scheduler slot released after the stream finished.
    scheduler = client.app.state.scheduler
    assert scheduler.active_count("rt-a") == 0
    assert scheduler._redis is not None


def test_active_set_emptied_after_stream(
    client: TestClient, session: Session, monkeypatch, clean_redis
) -> None:
    seed_instance(session, alias="rt-rel", machine_uid="rt-rel-m")
    events = [
        {
            "type": "response.created",
            "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
        },
        completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, {}))

    resp = client.post(
        "/v1/responses", json={"model": "rt-rel", "input": "x", "stream": True}
    )
    assert resp.status_code == 200
    # im:sched:active drained back to empty for the alias.
    assert clean_redis.smembers("im:sched:active:rt-rel") == set()
    assert clean_redis.hgetall("im:vram:used:rt-rel-m") == {}


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------
def test_midstream_failure_yields_failed_frame_and_persists(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="rt-fail", machine_uid="rt-fail-m")

    async def failing_aresponses(*args, **kwargs):  # noqa: ARG001
        async def gen():
            yield {
                "type": "response.created",
                "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
            }
            raise litellm.exceptions.MidStreamFallbackError(
                "provider reported failure",
                "rt-fail",
                "openai",
                original_exception=litellm.exceptions.InternalServerError(
                    message="boom", model="rt-fail", llm_provider="openai"
                ),
            )

        return gen()

    monkeypatch.setattr(litellm, "aresponses", failing_aresponses)

    resp = client.post(
        "/v1/responses", json={"model": "rt-fail", "input": "x", "stream": True}
    )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    types = [t for t, _ in frames]
    assert "response.failed" in types
    assert types[-1] == "__done__"

    failed = next(p for t, p in frames if t == "response.failed")
    assert failed["response"]["id"].startswith(CLIENT_ID_PREFIX)
    assert failed["response"]["id"] != WRAPPED_ID
    assert failed["response"]["status"] == "failed"
    assert failed["response"]["error"]["code"] == "upstream_failed"
    # Failed frames are validated against the full ResponseResource schema.
    body = failed["response"]
    assert isinstance(body["created_at"], int)
    assert body["completed_at"] is None
    assert body["output"] == []
    assert body["tools"] == []
    assert body["tool_choice"] == "auto"
    assert body["store"] is True
    assert body["service_tier"] == "default"
    assert body["model"] == "rt-fail"

    created = next(p for t, p in frames if t == "response.created")
    client_id = created["response"]["id"]
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == client_id)
        .first()
    )
    assert record is not None
    assert record.status == "failed"
    assert record.error_code == "upstream_failed"

    # Slot released even on failure.
    scheduler = client.app.state.scheduler
    assert scheduler.active_count("rt-fail") == 0


# ---------------------------------------------------------------------------
# Client disconnect releases the slot
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# previous_response_id reconstruction
# ---------------------------------------------------------------------------
def test_previous_response_id_reconstructs_input(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="rt-cont", machine_uid="rt-cont-m")

    prior_input = [{"role": "user", "content": "first question"}]
    prior_output = [
        {
            "id": "msg_prev",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "first answer"}],
        }
    ]
    prior = ResponseRecord(
        response_id="resp_" + uuid.uuid4().hex,
        previous_response_id=None,
        input_items=prior_input,
        output_items=prior_output,
        provider_definition_id=session.query(ProviderDefinition)
        .filter(ProviderDefinition.alias == "rt-cont")
        .first()
        .id,
        status="completed",
    )
    session.add(prior)
    session.commit()
    prior_id = prior.response_id

    events = [
        {
            "type": "response.created",
            "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
        },
        completed_event([], {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}),
    ]
    captured: dict[str, Any] = {}
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, captured))

    resp = client.post(
        "/v1/responses",
        json={
            "model": "rt-cont",
            "input": "second question",
            "previous_response_id": prior_id,
            "stream": True,
        },
    )
    assert resp.status_code == 200

    handed_input = captured["input"]
    assert isinstance(handed_input, list)
    # Prior input + prior output + this turn's new user message.
    assert handed_input[0] == {"role": "user", "content": "first question"}
    assert handed_input[1] == prior_output[0]
    assert handed_input[-1] == {"role": "user", "content": "second question"}
    # We never pass previous_response_id to litellm.
    assert "previous_response_id" not in captured

    # The new record chains to the prior id.
    frames = parse_sse(resp.text)
    new_id = next(p for t, p in frames if t == "response.created")["response"]["id"]
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == new_id)
        .first()
    )
    assert record.previous_response_id == prior_id
    # input_items stores the accumulated conversation (prior input +
    # prior output + this turn's new user message), so the next
    # continuation reconstructs in O(1) from this record alone.
    assert record.input_items == handed_input


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------
def test_unknown_alias_404(client: TestClient) -> None:
    resp = client.post("/v1/responses", json={"model": "nope", "input": "x"})
    assert resp.status_code == 404


def test_disabled_alias_404(client: TestClient, session: Session) -> None:
    seed_instance(session, alias="rt-off", machine_uid="rt-off-m", enabled=False)
    resp = client.post("/v1/responses", json={"model": "rt-off", "input": "x"})
    assert resp.status_code == 404


def test_no_connected_provider_503(client: TestClient, session: Session) -> None:
    machine = get_or_create_machine(session, uid="rt-np", host="127.0.0.1")
    agent = make_agent(session, machine, connected=False)
    definition = make_definition(
        session, alias="rt-np", backend_config={"model": {"file": "m.gguf"}}
    )
    make_instance(session, agent, definition, backend_status="stopped")
    resp = client.post("/v1/responses", json={"model": "rt-np", "input": "x"})
    assert resp.status_code == 503


def test_non_stream_queue_cleared_503(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """QueueCleared on the non-stream path -> 503 JSON (never a stream)."""
    from app.services.scheduler import QueueCleared

    seed_instance(session, alias="rt-qc", machine_uid="rt-qc-m")
    scheduler = client.app.state.scheduler

    async def cleared_acquire(alias, request_id):  # noqa: ARG001
        raise QueueCleared("the 'rt-qc' queue was cleared by an operator")

    monkeypatch.setattr(scheduler, "acquire", cleared_acquire)
    resp = client.post(
        "/v1/responses", json={"model": "rt-qc", "input": "x", "stream": False}
    )
    assert resp.status_code == 503


def test_missing_fields_400(client: TestClient) -> None:
    resp = client.post("/v1/responses", json={"model": "x"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Non-stream path
# ---------------------------------------------------------------------------
def test_non_stream_returns_json(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="rt-ns", machine_uid="rt-ns-m")
    output = [{"id": "msg_1", "type": "message", "role": "assistant"}]
    final = {
        "id": WRAPPED_ID,
        "object": "response",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "input_tokens_details": {"cached_tokens": 1},
        },
    }

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        assert kwargs.get("stream") is False
        return final

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    resp = client.post(
        "/v1/responses", json={"model": "rt-ns", "input": "hi", "stream": False}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"].startswith(CLIENT_ID_PREFIX)
    assert body["id"] != WRAPPED_ID
    assert body["output"] == output

    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == body["id"])
        .first()
    )
    assert record is not None
    assert record.status == "completed"
    assert record.parameters["litellm_response_id"] == WRAPPED_ID
    assert record.input_tokens == 5
    assert record.output_tokens == 2

    scheduler = client.app.state.scheduler
    assert scheduler.active_count("rt-ns") == 0


def llama_cpp_sparse_response() -> dict[str, Any]:
    """Terminal response shaped like llama.cpp's: required fields null or
    absent, usage details full of nulls."""
    return {
        "id": WRAPPED_ID,
        "object": "response",
        "created_at": 1791512825,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "model": "m.gguf",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "parallel_tool_calls": None,
        "temperature": None,
        "tool_choice": None,
        "tools": None,
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": None,
        "truncation": None,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 3,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
            },
            "output_tokens": 46,
            "output_tokens_details": None,
            "total_tokens": 56,
            "cost": None,
        },
        "user": None,
        "store": None,
    }


def assert_spec_normalized(body: dict[str, Any]) -> None:
    assert isinstance(body["completed_at"], int)
    assert body["tools"] == []
    assert body["tool_choice"] == "auto"
    assert body["truncation"] == "disabled"
    assert body["parallel_tool_calls"] is True
    assert body["text"] == {"format": {"type": "text"}}
    assert body["top_p"] == 1
    assert body["temperature"] == 1
    assert body["presence_penalty"] == 0
    assert body["frequency_penalty"] == 0
    assert body["top_logprobs"] == 0
    assert body["store"] is True
    assert body["background"] is False
    assert body["service_tier"] == "default"
    assert body["max_tool_calls"] is None
    assert body["safety_identifier"] is None
    assert body["prompt_cache_key"] is None
    usage = body["usage"]
    assert usage["input_tokens_details"] == {"cached_tokens": 3}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 0}


def test_non_stream_normalizes_sparse_provider_response(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="rt-sparse", machine_uid="rt-sparse-m")
    final = llama_cpp_sparse_response()

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        return final

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    resp = client.post("/v1/responses", json={"model": "rt-sparse", "input": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"].startswith(CLIENT_ID_PREFIX)
    assert_spec_normalized(body)


def test_stream_terminal_frame_is_normalized(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="rt-sparse-s", machine_uid="rt-sparse-sm")
    events = [completed_event([], llama_cpp_sparse_response()["usage"])]
    events[0]["response"].update(
        {k: v for k, v in llama_cpp_sparse_response().items() if k != "id"}
    )
    pristine = copy.deepcopy(events)
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, {}))

    resp = client.post(
        "/v1/responses",
        json={"model": "rt-sparse-s", "input": "hi", "stream": True},
    )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    terminal = next(data for etype, data in frames if etype == "response.completed")
    assert_spec_normalized(terminal["response"])
    # The route must not mutate litellm's own event dicts.
    assert events == pristine


def test_stream_lifecycle_frames_are_normalized(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """The suite validates created/in_progress frames against the full
    ResponseResource schema — bare skeletons must be filled."""
    seed_instance(session, alias="rt-lc", machine_uid="rt-lc-m")
    skeleton = {
        "id": WRAPPED_ID,
        "object": "response",
        "status": "in_progress",
    }
    events = [
        {"type": "response.created", "response": dict(skeleton)},
        {"type": "response.in_progress", "response": dict(skeleton)},
        completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, {}))

    resp = client.post(
        "/v1/responses",
        json={"model": "rt-lc", "input": "hi", "stream": True},
    )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    for etype in ("response.created", "response.in_progress"):
        frame = next(data for t, data in frames if t == etype)
        body = frame["response"]
        assert body["id"].startswith(CLIENT_ID_PREFIX)
        assert body["status"] == "in_progress"
        assert body["output"] == []
        assert body["completed_at"] is None
        assert body["tools"] == []
        assert body["tool_choice"] == "auto"
        assert body["temperature"] == 1
        assert body["store"] is True
        assert body["background"] is False
        assert body["service_tier"] == "default"
        assert body["usage"] is None


def test_non_stream_echoed_tools_get_required_keys(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """functionToolSchema requires description/parameters/strict
    (nullable) — echoed request tools must carry them."""
    seed_instance(session, alias="rt-tools", machine_uid="rt-tools-m")
    final = llama_cpp_sparse_response()
    final["tools"] = [{"type": "function", "name": "get_weather"}]

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        return final

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "rt-tools",
            "input": "weather?",
            "tools": [{"type": "function", "name": "get_weather"}],
        },
    )
    assert resp.status_code == 200
    tool = resp.json()["tools"][0]
    assert tool["name"] == "get_weather"
    assert tool["strict"] is None
    assert tool["description"] is None
    assert tool["parameters"] is None


def test_stream_duplicate_terminal_frames_dropped(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Exactly one terminal frame may reach the client (FINDINGS §2)."""
    seed_instance(session, alias="rt-dup", machine_uid="rt-dup-m")
    ev = completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    events = [dict(ev), dict(ev)]
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, {}))

    resp = client.post(
        "/v1/responses",
        json={"model": "rt-dup", "input": "hi", "stream": True},
    )
    frames = parse_sse(resp.text)
    assert [t for t, _ in frames].count("response.completed") == 1


def test_stream_enum_typed_terminal_frame_is_normalized(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """litellm may send `type` as an enum member — the route must still
    recognize and normalize the terminal frame."""
    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    seed_instance(session, alias="rt-enum", machine_uid="rt-enum-m")
    ev = completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    ev["type"] = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream([ev], {}))

    resp = client.post(
        "/v1/responses",
        json={"model": "rt-enum", "input": "hi", "stream": True},
    )
    frames = parse_sse(resp.text)
    terminal = next(data for etype, data in frames if etype == "response.completed")
    assert terminal["response"]["tools"] == []
    assert isinstance(terminal["response"]["completed_at"], int)


def test_stream_upstream_failed_frame_is_normalized(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """A raw response.failed passing through (litellm normally raises
    instead) must not reach the client as a sparse skeleton."""
    seed_instance(session, alias="rt-uf", machine_uid="rt-uf-m")
    events = [
        {
            "type": "response.failed",
            "response": {
                "id": WRAPPED_ID,
                "status": "failed",
                "error": {"type": "server_error", "code": "c", "message": "m"},
            },
        }
    ]
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, {}))

    resp = client.post(
        "/v1/responses",
        json={"model": "rt-uf", "input": "x", "stream": True},
    )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    failed = next(p for t, p in frames if t == "response.failed")
    assert failed["response"]["id"].startswith(CLIENT_ID_PREFIX)
    assert isinstance(failed["response"]["created_at"], int)
    assert failed["response"]["tools"] == []
    assert failed["response"]["error"]["code"] == "c"

    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == failed["response"]["id"])
        .first()
    )
    assert record is not None
    assert record.status == "failed"
    assert record.error_code == "c"


def test_stream_incomplete_frame_and_persistence(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="rt-inc", machine_uid="rt-inc-m")
    event = {
        "type": "response.incomplete",
        "response": {
            "id": WRAPPED_ID,
            "object": "response",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        },
    }
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream([event], {}))

    resp = client.post(
        "/v1/responses",
        json={"model": "rt-inc", "input": "hi", "stream": True},
    )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    terminal = next(data for etype, data in frames if etype == "response.incomplete")
    body = terminal["response"]
    assert body["status"] == "incomplete"
    assert body["completed_at"] is None
    assert body["store"] is True

    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == body["id"])
        .first()
    )
    assert record is not None
    assert record.status == "incomplete"
    assert record.output_tokens == 1


def test_non_stream_echoes_requested_params(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Provider-null fields fall back to the REQUEST values, not constants."""
    seed_instance(session, alias="rt-req", machine_uid="rt-req-m")
    final = llama_cpp_sparse_response()

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        return final

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "rt-req",
            "input": "hi",
            "temperature": 0.7,
            "top_p": 0.9,
            "tools": [{"type": "function", "name": "x", "parameters": {}}],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "store": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9
    assert body["tools"] == [
        {
            "type": "function",
            "name": "x",
            "parameters": {},
            "description": None,
            "strict": None,
        }
    ]
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    assert body["store"] is False


def test_normalize_noop_on_spec_complete_response() -> None:
    from app.api.v1.responses import _normalize_response

    data: dict[str, Any] = {
        "id": "resp_x",
        "object": "response",
        "created_at": 1,
        "completed_at": 2,
        "status": "completed",
        "incomplete_details": None,
        "model": "m",
        "previous_response_id": None,
        "instructions": None,
        "output": [],
        "error": None,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
        "parallel_tool_calls": True,
        "text": {"format": {"type": "text"}},
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_logprobs": 0,
        "temperature": 1,
        "reasoning": None,
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "max_output_tokens": None,
        "max_tool_calls": None,
        "store": True,
        "background": False,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": None,
        "prompt_cache_key": None,
    }
    before = copy.deepcopy(data)
    _normalize_response(data, store=True)
    assert data == before


def test_normalize_defaults_missing_identity_and_rejects_wrong_types() -> None:
    from app.api.v1.responses import _normalize_response

    # Provider omits status/model/object entirely and sends wrong-typed
    # values for tools/temperature/truncation; client echoes a string
    # temperature (invalid too).
    data: dict[str, Any] = {
        "output": [],
        "tools": {},
        "temperature": "0.7",
        "truncation": 123,
    }
    _normalize_response(
        data, store=True, requested={"model": "rt-x", "temperature": "0.7"}
    )
    assert data["status"] == "completed"
    assert data["model"] == "rt-x"
    assert data["object"] == "response"
    assert data["tools"] == []
    assert data["temperature"] == 1  # request tier gated out too
    assert data["truncation"] == "disabled"
    assert isinstance(data["created_at"], int)
    # status defaults BEFORE completed_at logic: defaulted-completed
    # responses still get a concrete completed_at.
    assert isinstance(data["completed_at"], int)


def test_normalize_coerces_usage_token_counters() -> None:
    from app.api.v1.responses import _normalize_response

    data: dict[str, Any] = {
        "status": "completed",
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": 56,
        },
    }
    _normalize_response(data, store=True)
    assert data["usage"]["input_tokens"] == 0
    assert data["usage"]["output_tokens"] == 0
    assert data["usage"]["total_tokens"] == 56

    data = {"status": "completed", "usage": {"input_tokens": 3, "output_tokens": 4}}
    _normalize_response(data, store=True)
    assert data["usage"]["total_tokens"] == 7


def test_normalize_incomplete_keeps_null_completed_at() -> None:
    from app.api.v1.responses import _normalize_response

    data: dict[str, Any] = {"status": "incomplete"}
    _normalize_response(data, store=True)
    assert data["completed_at"] is None
    assert data["usage"] is None
    assert isinstance(data["created_at"], int)

    # Missing status on an incomplete terminal frame derives from the frame.
    data = {}
    _normalize_response(data, store=True, default_status="incomplete")
    assert data["status"] == "incomplete"
    assert data["completed_at"] is None


# ---------------------------------------------------------------------------
# Generator-level cancellation-safe release (drives _stream_response directly)
# ---------------------------------------------------------------------------
async def test_stream_generator_aclose_releases_and_persists_failed(
    session, monkeypatch
) -> None:
    """Closing the SSE generator mid-stream (client disconnect) must run the
    cancellation-safe finally: release the scheduler slot and persist the
    turn as failed. TestClient can't reliably interrupt a running generator,
    so we drive the route generator directly on the event loop."""
    from app.api.v1 import responses as route_mod
    from app.services.scheduler import InferenceScheduler

    instance = seed_instance(session, alias="rt-gen", machine_uid="rt-gen-m")
    definition_id = instance.provider_definition_id

    async def hang_aresponses(*args, **kwargs):  # noqa: ARG001
        async def gen():
            yield {
                "type": "response.created",
                "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
            }
            await asyncio.sleep(120)  # client disconnects long before this

        return gen()

    monkeypatch.setattr(litellm, "aresponses", hang_aresponses)

    import redis.asyncio as aioredis

    aredis = aioredis.from_url(
        __import__("os").environ["TEST_REDIS_URL"], decode_responses=True
    )
    scheduler = InferenceScheduler(aredis)
    # Phase 6 keepalive redesign: the generator owns the acquire (it must,
    # so it can emit keepalives while a cold boot blocks). No pre-admission
    # here — the Admission is created inside the generator.
    agen = route_mod._stream_response(
        scheduler=scheduler,
        alias="rt-gen",
        request_id="req-gen",
        client_response_id="resp_gen000",
        previous_response_id=None,
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=definition_id,
        litellm_input=[{"role": "user", "content": "x"}],
        passthrough={},
        base_params={"model": "rt-gen", "provider_type": "mock"},
        store=True,
    )
    # Consume the first frame, then simulate a client disconnect.
    first = await agen.__anext__()
    assert "response.created" in first
    assert scheduler.active_count("rt-gen") == 1

    await agen.aclose()

    # Cancellation-safe finally landed: slot released + failed record.
    assert scheduler.active_count("rt-gen") == 0
    assert await aredis.smembers("im:sched:active:rt-gen") == set()
    with Session(engine) as check:
        rec = check.exec(
            select(ResponseRecord).where(ResponseRecord.response_id == "resp_gen000")
        ).first()
        assert rec is not None
        assert rec.status == "failed"
    await aredis.aclose()


# ---------------------------------------------------------------------------
# Cold-boot SSE keepalive + in-stream error contract (Phase 6)
# ---------------------------------------------------------------------------
# Chosen contract (documented in ARCHITECTURE.md §7):
#   * Fast pre-stream checks stay HTTP: unknown/disabled alias -> 404,
#     missing fields -> 400, and the cheap zero-candidates emptiness check
#     -> 503 (never open a stream that can never carry a response).
#   * The SLOW part (cold boot + FIFO queue wait) runs INSIDE the stream,
#     emitting `: keep-alive` SSE comments so proxies don't drop it.
#   * Scheduler errors that arrive inside the stream (QueueTimeout, or a
#     NoProviderAvailable after the pre-check raced away) surface as the
#     spec response.failed + error frames + [DONE] and a failed
#     ResponseRecord — no HTTP status is possible once bytes are sent.
#   * Non-stream path is unchanged: 503/504 JSON.
# ---------------------------------------------------------------------------
def _short_keepalive(monkeypatch, interval: float = 0.05) -> None:
    from app.api.v1 import responses as route_mod

    monkeypatch.setattr(route_mod, "KEEPALIVE_INTERVAL_SECONDS", interval)


async def test_stream_emits_keepalive_during_slow_acquire(session, monkeypatch) -> None:
    """A slow acquire yields `: keep-alive` comments before real events."""
    import redis.asyncio as aioredis

    from app.api.v1 import responses as route_mod
    from app.services.scheduler import InferenceScheduler

    _short_keepalive(monkeypatch)
    instance = seed_instance(session, alias="ka-a", machine_uid="ka-m")

    aredis = aioredis.from_url(
        __import__("os").environ["TEST_REDIS_URL"], decode_responses=True
    )
    scheduler = InferenceScheduler(aredis)

    release = asyncio.Event()
    original_acquire = scheduler.acquire

    async def slow_acquire(alias, request_id):
        adm = await original_acquire(alias, request_id)
        await release.wait()  # hold so the keepalive loop must spin
        return adm

    monkeypatch.setattr(scheduler, "acquire", slow_acquire)

    events = [
        {
            "type": "response.created",
            "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
        },
        completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, {}))

    agen = route_mod._stream_response(
        scheduler=scheduler,
        alias="ka-a",
        request_id="req-ka",
        client_response_id="resp_ka000",
        previous_response_id=None,
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=instance.provider_definition_id,
        litellm_input=[{"role": "user", "content": "x"}],
        passthrough={},
        base_params={"model": "ka-a", "provider_type": "mock"},
        store=True,
    )

    # The generator is pull-driven: each __anext__ blocks up to the
    # keepalive interval and returns a comment while acquire is stuck on
    # `release`. Unblock it after a short delay and collect everything.
    async def unblock() -> None:
        await asyncio.sleep(0.18)
        release.set()

    unblock_task = asyncio.create_task(unblock())
    chunks: list[str] = []
    while True:
        chunk = await agen.__anext__()
        chunks.append(chunk)
        if "response.created" in chunk:
            break
    await unblock_task
    # Drain the remaining terminal frames.
    rest = [chunk async for chunk in agen]
    body = "".join([*chunks, *rest])

    # Bytes flowed (keepalives) before admission resolved.
    assert body.count(": keep-alive\n\n") >= 2
    assert chunks[0] == ": keep-alive\n\n"
    assert "response.created" in body
    assert body.endswith("data: [DONE]\n\n")
    # Slot released at terminal.
    assert scheduler.active_count("ka-a") == 0
    await aredis.aclose()


async def test_stream_queue_timeout_surfaces_as_failed_frames(
    session, monkeypatch
) -> None:
    """QueueTimeout inside the stream -> response.failed + error + [DONE],
    failed record persisted (no HTTP status possible once streaming)."""
    import redis.asyncio as aioredis

    from app.api.v1 import responses as route_mod
    from app.services.scheduler import InferenceScheduler, QueueTimeout

    seed_instance(session, alias="qt-a", machine_uid="qt-m")
    definition_id = (
        session.query(ProviderDefinition)
        .filter(ProviderDefinition.alias == "qt-a")
        .first()
        .id
    )
    aredis = aioredis.from_url(
        __import__("os").environ["TEST_REDIS_URL"], decode_responses=True
    )
    scheduler = InferenceScheduler(aredis)

    async def timeout_acquire(alias, request_id):  # noqa: ARG001
        raise QueueTimeout("timed out waiting for a 'qt-a' slot")

    monkeypatch.setattr(scheduler, "acquire", timeout_acquire)

    agen = route_mod._stream_response(
        scheduler=scheduler,
        alias="qt-a",
        request_id="req-qt",
        client_response_id="resp_qt000",
        previous_response_id=None,
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=definition_id,
        litellm_input=[{"role": "user", "content": "x"}],
        passthrough={},
        base_params={"model": "qt-a", "provider_type": "mock"},
        store=True,
    )
    body = "".join([chunk async for chunk in agen])
    frames = parse_sse(body)
    types = [t for t, _ in frames]
    assert "response.failed" in types
    assert types[-1] == "__done__"
    failed = next(p for t, p in frames if t == "response.failed")
    assert failed["response"]["id"] == "resp_qt000"
    assert failed["response"]["error"]["code"] == "queue_timeout"
    assert "error" in types
    with Session(engine) as check:
        rec = check.exec(
            select(ResponseRecord).where(ResponseRecord.response_id == "resp_qt000")
        ).first()
        assert rec is not None
        assert rec.status == "failed"
        assert rec.error_code == "queue_timeout"
    await aredis.aclose()


async def test_stream_noprovider_after_precheck_surfaces_as_failed(
    session, monkeypatch
) -> None:
    """NoProviderAvailable raised inside the stream (candidates raced away
    after the pre-stream check) -> response.failed with code no_provider."""
    import redis.asyncio as aioredis

    from app.api.v1 import responses as route_mod
    from app.services.scheduler import InferenceScheduler, NoProviderAvailable

    seed_instance(session, alias="np-a", machine_uid="np-m")
    definition_id = (
        session.query(ProviderDefinition)
        .filter(ProviderDefinition.alias == "np-a")
        .first()
        .id
    )
    aredis = aioredis.from_url(
        __import__("os").environ["TEST_REDIS_URL"], decode_responses=True
    )
    scheduler = InferenceScheduler(aredis)

    async def np_acquire(alias, request_id):  # noqa: ARG001
        raise NoProviderAvailable("no connected provider instance for alias 'np-a'")

    monkeypatch.setattr(scheduler, "acquire", np_acquire)

    agen = route_mod._stream_response(
        scheduler=scheduler,
        alias="np-a",
        request_id="req-np",
        client_response_id="resp_np000",
        previous_response_id=None,
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=definition_id,
        litellm_input=[{"role": "user", "content": "x"}],
        passthrough={},
        base_params={"model": "np-a", "provider_type": "mock"},
        store=True,
    )
    body = "".join([chunk async for chunk in agen])
    frames = parse_sse(body)
    failed = next(p for t, p in frames if t == "response.failed")
    assert failed["response"]["error"]["code"] == "no_provider"
    assert [t for t, _ in frames][-1] == "__done__"
    await aredis.aclose()


async def test_stream_queue_cleared_surfaces_as_failed_frames(
    session, monkeypatch
) -> None:
    """QueueCleared inside the stream -> response.failed with code
    queue_cleared + failed record persisted (no HTTP status once streaming)."""
    import redis.asyncio as aioredis

    from app.api.v1 import responses as route_mod
    from app.services.scheduler import InferenceScheduler, QueueCleared

    seed_instance(session, alias="qc-a", machine_uid="qc-m")
    definition_id = (
        session.query(ProviderDefinition)
        .filter(ProviderDefinition.alias == "qc-a")
        .first()
        .id
    )
    aredis = aioredis.from_url(
        __import__("os").environ["TEST_REDIS_URL"], decode_responses=True
    )
    scheduler = InferenceScheduler(aredis)

    async def cleared_acquire(alias, request_id):  # noqa: ARG001
        raise QueueCleared("the 'qc-a' queue was cleared by an operator")

    monkeypatch.setattr(scheduler, "acquire", cleared_acquire)

    agen = route_mod._stream_response(
        scheduler=scheduler,
        alias="qc-a",
        request_id="req-qc",
        client_response_id="resp_qc000",
        previous_response_id=None,
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=definition_id,
        litellm_input=[{"role": "user", "content": "x"}],
        passthrough={},
        base_params={"model": "qc-a", "provider_type": "mock"},
        store=True,
    )
    body = "".join([chunk async for chunk in agen])
    frames = parse_sse(body)
    types = [t for t, _ in frames]
    assert "response.failed" in types
    assert types[-1] == "__done__"
    failed = next(p for t, p in frames if t == "response.failed")
    assert failed["response"]["id"] == "resp_qc000"
    assert failed["response"]["error"]["code"] == "queue_cleared"
    with Session(engine) as check:
        rec = check.exec(
            select(ResponseRecord).where(ResponseRecord.response_id == "resp_qc000")
        ).first()
        assert rec is not None
        assert rec.status == "failed"
        assert rec.error_code == "queue_cleared"
    await aredis.aclose()


async def test_client_disconnect_during_keepalive_releases_slot(
    session, monkeypatch
) -> None:
    """Disconnecting while the keepalive loop blocks on a queued acquire
    must not leak the waiter or the slot."""
    import redis.asyncio as aioredis

    from app.api.v1 import responses as route_mod
    from app.services.scheduler import InferenceScheduler

    _short_keepalive(monkeypatch)
    seed_instance(session, alias="dl-a", machine_uid="dl-m", capacity=1)
    aredis = aioredis.from_url(
        __import__("os").environ["TEST_REDIS_URL"], decode_responses=True
    )
    scheduler = InferenceScheduler(aredis)
    # Occupy the only slot so the generator's acquire genuinely queues and
    # the keepalive phase is what the client disconnects during.
    await scheduler.acquire("dl-a", "holder")
    definition_id = (
        session.query(ProviderDefinition)
        .filter(ProviderDefinition.alias == "dl-a")
        .first()
        .id
    )

    agen = route_mod._stream_response(
        scheduler=scheduler,
        alias="dl-a",
        request_id="req-dl",
        client_response_id="resp_dl000",
        previous_response_id=None,
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=definition_id,
        litellm_input=[{"role": "user", "content": "x"}],
        passthrough={},
        base_params={"model": "dl-a", "provider_type": "mock"},
        store=True,
    )
    # Consume one keepalive (proves we are in the admission-wait phase),
    # then disconnect.
    first = await agen.__anext__()
    assert first == ": keep-alive\n\n"
    assert "req-dl" in scheduler._states["dl-a"].waiters
    await agen.aclose()

    # No leaked waiter; only the original holder slot remains.
    assert list(scheduler._states["dl-a"].waiters) == []
    assert scheduler.active_count("dl-a") == 1
    await scheduler.release("dl-a", "holder")
    assert scheduler.active_count("dl-a") == 0
    await aredis.aclose()


async def test_stream_aresponses_call_time_error_is_terminal(
    session, monkeypatch
) -> None:
    """S1: ``litellm.aresponses`` raising at CALL time (connection
    refused / APIError before the first event) must produce the same
    terminal SSE contract as a mid-stream failure: response.failed +
    error + [DONE], not a bare truncation after the keepalives."""
    import redis.asyncio as aioredis

    from app.api.v1 import responses as route_mod
    from app.services.scheduler import InferenceScheduler

    instance = seed_instance(session, alias="ct-a", machine_uid="ct-m")

    async def raise_at_call(*args, **kwargs):  # noqa: ARG001
        raise litellm.exceptions.APIError(
            status_code=500,
            message="connection refused",
            llm_provider="openai",
            model="ct-a",
        )

    monkeypatch.setattr(litellm, "aresponses", raise_at_call)

    aredis = aioredis.from_url(
        __import__("os").environ["TEST_REDIS_URL"], decode_responses=True
    )
    scheduler = InferenceScheduler(aredis)
    agen = route_mod._stream_response(
        scheduler=scheduler,
        alias="ct-a",
        request_id="req-ct",
        client_response_id="resp_ct000",
        previous_response_id=None,
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=instance.provider_definition_id,
        litellm_input=[{"role": "user", "content": "x"}],
        passthrough={},
        base_params={"model": "ct-a", "provider_type": "mock"},
        store=True,
    )
    body = "".join([chunk async for chunk in agen])
    frames = parse_sse(body)
    types = [t for t, _ in frames]
    assert "response.failed" in types
    assert "error" in types
    assert types[-1] == "__done__"
    failed = next(p for t, p in frames if t == "response.failed")
    assert failed["response"]["id"] == "resp_ct000"
    assert failed["response"]["status"] == "failed"
    assert failed["response"]["error"]["code"] == "upstream_failed"

    with Session(engine) as check:
        rec = check.exec(
            select(ResponseRecord).where(ResponseRecord.response_id == "resp_ct000")
        ).first()
        assert rec is not None
        assert rec.status == "failed"
        assert rec.error_code == "upstream_failed"
    # Slot released; the exception was NOT swallowed as a SchedulerError.
    assert scheduler.active_count("ct-a") == 0
    await aredis.aclose()


def test_non_stream_queue_timeout_still_http_504(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Non-stream path keeps the HTTP 504 (unchanged by the keepalive work)."""
    from app.services.scheduler import QueueTimeout

    seed_instance(session, alias="ns504", machine_uid="ns504-m")
    scheduler = client.app.state.scheduler

    async def timeout_acquire(alias, request_id):  # noqa: ARG001
        raise QueueTimeout("nope")

    monkeypatch.setattr(scheduler, "acquire", timeout_acquire)
    resp = client.post(
        "/v1/responses", json={"model": "ns504", "input": "x", "stream": False}
    )
    assert resp.status_code == 504


def test_non_stream_noprovider_still_http_503(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Non-stream path keeps the HTTP 503 (unchanged)."""
    from app.services.scheduler import NoProviderAvailable

    seed_instance(session, alias="ns503", machine_uid="ns503-m")
    scheduler = client.app.state.scheduler

    async def np_acquire(alias, request_id):  # noqa: ARG001
        raise NoProviderAvailable("none")

    monkeypatch.setattr(scheduler, "acquire", np_acquire)
    resp = client.post(
        "/v1/responses", json={"model": "ns503", "input": "x", "stream": False}
    )
    assert resp.status_code == 503


def test_stream_zero_candidates_precheck_is_http_503(
    client: TestClient, session: Session
) -> None:
    """Streaming with zero connected candidates keeps the cheap pre-stream
    HTTP 503 (never opens a keepalive-only stream that can't respond)."""
    machine = get_or_create_machine(session, uid="sc503", host="127.0.0.1")
    agent = make_agent(session, machine, connected=False)
    definition = make_definition(
        session, alias="sc503", backend_config={"model": {"file": "m.gguf"}}
    )
    make_instance(session, agent, definition, backend_status="stopped")
    resp = client.post(
        "/v1/responses", json={"model": "sc503", "input": "x", "stream": True}
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Phase 23 S4: rate + latency telemetry end-to-end into TokenUsageSample
# ---------------------------------------------------------------------------
def test_stream_persists_rate_and_latency_telemetry(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Terminal completion_tokens_details (rates + seconds) map into the
    persisted TokenUsageSample (rates verbatim, *_time seconds -> *_ms)."""
    instance = seed_instance(session, alias="rt-tel", machine_uid="rt-tel-m")

    output = [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        }
    ]
    usage = {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 3},
        "output_tokens_details": {"reasoning_tokens": 1},
        "completion_tokens_details": {
            "prompt_per_second": 1412.21,
            "predicted_per_second": 36.2534,
            "prompt_time": 0.12,
            "prediction_time": 3.4,
        },
    }
    events = [
        {
            "type": "response.created",
            "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
        },
        completed_event(output, usage),
    ]
    captured: dict[str, Any] = {}
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, captured))

    resp = client.post(
        "/v1/responses", json={"model": "rt-tel", "input": "hi", "stream": True}
    )
    assert resp.status_code == 200

    from app.models import TokenUsageSample

    sample = (
        session.query(TokenUsageSample)
        .filter(TokenUsageSample.provider_instance_id == instance.id)
        .first()
    )
    assert sample is not None
    assert sample.prompt_per_second > 0
    assert sample.predicted_per_second > 0
    assert sample.prompt_per_second == pytest.approx(1412.21)
    assert sample.predicted_per_second == pytest.approx(36.2534)
    # seconds -> ms (admin x1000).
    assert sample.predicted_ms == pytest.approx(3400.0)
    assert sample.prompt_ms == pytest.approx(120.0)


def test_non_stream_persists_rate_and_latency_telemetry(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """The non-streaming path maps the same completion_tokens_details."""
    instance = seed_instance(session, alias="rt-nstel", machine_uid="rt-nstel-m")
    output = [{"id": "msg_1", "type": "message", "role": "assistant"}]
    final = {
        "id": WRAPPED_ID,
        "object": "response",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "input_tokens_details": {"cached_tokens": 1},
            "completion_tokens_details": {
                "prompt_per_second": 1412.21,
                "predicted_per_second": 36.2534,
                "prompt_time": 0.12,
                "prediction_time": 3.4,
            },
        },
    }

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        assert kwargs.get("stream") is False
        return final

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    resp = client.post(
        "/v1/responses", json={"model": "rt-nstel", "input": "hi", "stream": False}
    )
    assert resp.status_code == 200

    from app.models import TokenUsageSample

    sample = (
        session.query(TokenUsageSample)
        .filter(TokenUsageSample.provider_instance_id == instance.id)
        .first()
    )
    assert sample is not None
    assert sample.prompt_per_second > 0
    assert sample.predicted_per_second > 0
    assert sample.predicted_ms == pytest.approx(3400.0)
    assert sample.prompt_ms == pytest.approx(120.0)


def test_litellm_preserves_completion_tokens_details_extras() -> None:
    """S3 finding lock: on a realistic terminal frame (with ``created_at``)
    litellm parses usage into the typed ``ResponseAPIUsage`` model, whose
    ``extra="allow"`` config carries the provider rate/latency extras
    through to ``to_dict`` so ``persist_turn`` can read them.

    Discriminating: without ``created_at`` litellm leaves ``usage`` unset
    (model_construct path), and if the typed model dropped extras the
    dumped ``completion_tokens_details`` would be absent — either way the
    assertions below fail.
    """
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.types.llms.openai import ResponseAPIUsage

    from app.services.sse import to_dict

    chunk = {
        "type": "response.completed",
        "response": {
            "id": "r",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "completion_tokens_details": {
                    "predicted_per_second": 36.25,
                    "prediction_time": 3.4,
                },
            },
        },
    }
    evt = OpenAIResponsesAPIConfig().transform_streaming_response(
        model="m", parsed_chunk=chunk, logging_obj=None
    )
    # Realistic path: usage is the typed model, not a raw dict.
    assert isinstance(evt.response.usage, ResponseAPIUsage)
    data = to_dict(evt)
    ctd = data["response"]["usage"]["completion_tokens_details"]
    assert ctd["predicted_per_second"] == 36.25
    assert ctd["prediction_time"] == 3.4
