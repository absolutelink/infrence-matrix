"""Phase 21 S1: transport-agnostic turn pipeline.

Covers the SSEEmitter split (``event_data``/``failed_events`` are the dict
half of ``frame``/``failed``) and the shared ``stream_turn_events`` runner
that both the SSE formatter and the Phase 21 WebSocket endpoint consume.
The runner yields normalized event *dicts* (already sequence-numbered by
the emitter) plus the ``KEEPALIVE`` sentinel during a slow acquire.
"""

import asyncio
import json
import os
from typing import Any

import litellm
import pytest
import redis.asyncio as aioredis
from sqlmodel import Session

from app.api.v1 import responses as route_mod
from app.api.v1.responses import KEEPALIVE, Keepalive, stream_turn_events
from app.services.scheduler import InferenceScheduler
from app.services.sse import SSEEmitter, _sse
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

CLIENT_ID = "resp_" + "a" * 32
WRAPPED_ID = "resp_bGl0ZWxsbQp0b3kxMjM"


def _serialize(event: dict[str, Any]) -> str:
    """Mirror the SSE formatter: ``_sse(type, dict)`` (no renumbering)."""
    return _sse(event["type"], event)


def seed_instance(session: Session, *, alias: str, machine_uid: str, capacity: int = 1):
    machine = get_or_create_machine(session, uid=machine_uid, host="127.0.0.1")
    agent = make_agent(session, machine, connected=True)
    definition = make_definition(
        session,
        alias=alias,
        capacity=capacity,
        vram_required=0,
        enabled=True,
        backend_config={"model": {"file": "m.gguf"}},
    )
    return make_instance(session, agent, definition, backend_status="running")


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
    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)

        async def gen():
            for event in events:
                yield event

        return gen()

    return fake_aresponses


# ---------------------------------------------------------------------------
# (b) event_data + serialize == frame
# ---------------------------------------------------------------------------
def test_event_data_serialize_equals_frame() -> None:
    """The dict half (``event_data``) + ``_sse`` reproduces ``frame`` byte
    for byte, including id replacement, position fill, and sequence."""
    events = [
        {"type": "response.created", "response": {"id": WRAPPED_ID, "output": []}},
        {"type": "response.in_progress", "response": {"id": WRAPPED_ID}},
        {"type": "response.output_item.added", "item": {"id": "msg_1"}},
        {
            "type": "response.content_part.added",
            "part": {"type": "output_text", "text": ""},
        },
        {"type": "response.output_text.delta", "delta": "hi"},
        {
            "type": "response.completed",
            "response": {"id": WRAPPED_ID, "output": [], "usage": None},
        },
    ]
    frame_emitter = SSEEmitter(CLIENT_ID)
    data_emitter = SSEEmitter(CLIENT_ID)
    for event in events:
        # Independent emitters, identical input order -> identical counters.
        assert _serialize(data_emitter.event_data(dict(event))) == frame_emitter.frame(
            dict(event)
        )
    assert data_emitter._seq == frame_emitter._seq


def test_event_data_does_not_mutate_caller_response() -> None:
    emitter = SSEEmitter(CLIENT_ID)
    data = {"type": "response.created", "response": {"id": WRAPPED_ID}}
    out = emitter.event_data(data)
    assert out["response"]["id"] == CLIENT_ID
    # The caller's nested response keeps the wrapped id (needed for capture).
    assert data["response"]["id"] == WRAPPED_ID


# ---------------------------------------------------------------------------
# (c) failed_events + serialize == failed
# ---------------------------------------------------------------------------
def test_failed_events_serialize_equals_failed() -> None:
    error = {"type": "server_error", "code": "upstream_failed", "message": "boom"}
    base = {"status": "failed", "output": [], "tools": [], "usage": None}
    str_emitter = SSEEmitter(CLIENT_ID)
    dict_emitter = SSEEmitter(CLIENT_ID)
    # Consume a sequence number on both so failed() continues counting.
    str_emitter.frame({"type": "response.created", "response": {"id": WRAPPED_ID}})
    dict_emitter.event_data(
        {"type": "response.created", "response": {"id": WRAPPED_ID}}
    )
    strs = str_emitter.failed(error, base)
    dicts = dict_emitter.failed_events(error, base)
    assert len(strs) == len(dicts) == 2
    for serialized, produced in zip(strs, dicts, strict=True):
        assert _serialize(produced) == serialized
    assert [d["type"] for d in dicts] == ["response.failed", "error"]
    assert dicts[0]["response"]["id"] == CLIENT_ID
    assert dicts[0]["response"]["status"] == "failed"
    assert dicts[0]["sequence_number"] == 1
    assert dicts[1]["sequence_number"] == 2


# ---------------------------------------------------------------------------
# (a) stream_turn_events yields event dicts + KEEPALIVE
# ---------------------------------------------------------------------------
async def test_stream_turn_events_yields_event_dicts(session, monkeypatch) -> None:
    instance = seed_instance(session, alias="st-a", machine_uid="st-m")
    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    scheduler = InferenceScheduler(aredis)

    output = [{"id": "msg_1", "type": "message", "role": "assistant"}]
    events = [
        {
            "type": "response.created",
            "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hi",
        },
        completed_event(
            output, {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        ),
    ]
    captured: dict[str, Any] = {}
    monkeypatch.setattr(litellm, "aresponses", make_fake_stream(events, captured))

    emitter = SSEEmitter("resp_st000")
    items = [
        item
        async for item in stream_turn_events(
            emitter=emitter,
            scheduler=scheduler,
            alias="st-a",
            request_id="req-st",
            client_response_id="resp_st000",
            previous_response_id=None,
            input_items_for_record=[{"role": "user", "content": "x"}],
            definition_id=instance.provider_definition_id,
            litellm_input=[{"role": "user", "content": "x"}],
            passthrough={},
            base_params={"model": "st-a", "provider_type": "mock"},
            store=True,
        )
    ]

    # All items are dicts (no keepalive on a fast acquire), never strings.
    assert all(isinstance(it, dict) for it in items)
    assert [it["type"] for it in items] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    # Sequence numbers are assigned by the shared emitter, monotonic from 0.
    assert [it["sequence_number"] for it in items] == [0, 1, 2]
    # Admin owns the client id on lifecycle frames.
    assert items[0]["response"]["id"] == "resp_st000"
    assert items[-1]["response"]["id"] == "resp_st000"
    # Terminal dict carries the normalized completed response.
    assert items[-1]["response"]["status"] == "completed"
    assert items[-1]["response"]["output"] == output
    # The emitter's counter advanced exactly len(items) times.
    assert emitter._seq == len(items)
    # Slot released at terminal.
    assert scheduler.active_count("st-a") == 0
    await aredis.aclose()


async def test_stream_turn_events_emits_keepalive_on_slow_acquire(
    session, monkeypatch
) -> None:
    """A slow acquire yields the KEEPALIVE sentinel before the event dicts
    (mirror of the SSE keepalive test, at the runner level)."""
    monkeypatch.setattr(route_mod, "KEEPALIVE_INTERVAL_SECONDS", 0.05)
    instance = seed_instance(session, alias="st-ka", machine_uid="st-ka-m")
    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    scheduler = InferenceScheduler(aredis)

    release = asyncio.Event()
    original_acquire = scheduler.acquire

    async def slow_acquire(alias, request_id):
        adm = await original_acquire(alias, request_id)
        await release.wait()
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

    emitter = SSEEmitter("resp_stka000")
    runner = stream_turn_events(
        emitter=emitter,
        scheduler=scheduler,
        alias="st-ka",
        request_id="req-stka",
        client_response_id="resp_stka000",
        previous_response_id=None,
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=instance.provider_definition_id,
        litellm_input=[{"role": "user", "content": "x"}],
        passthrough={},
        base_params={"model": "st-ka", "provider_type": "mock"},
        store=True,
    )

    async def unblock() -> None:
        await asyncio.sleep(0.18)
        release.set()

    unblock_task = asyncio.create_task(unblock())
    items: list[Any] = []
    async for item in runner:
        items.append(item)
        if isinstance(item, dict) and item["type"] == "response.created":
            break
    await unblock_task
    # Drain the rest.
    async for item in runner:
        items.append(item)

    keepalives = [it for it in items if it is KEEPALIVE]
    dicts = [it for it in items if isinstance(it, dict)]
    assert len(keepalives) >= 2
    assert all(isinstance(it, Keepalive) for it in keepalives)
    # Keepalives precede the first event dict.
    assert items.index(keepalives[0]) < items.index(dicts[0])
    assert [d["type"] for d in dicts] == ["response.created", "response.completed"]
    assert scheduler.active_count("st-ka") == 0
    await aredis.aclose()


async def test_stream_turn_events_failed_surfaces_as_dicts(
    session, monkeypatch
) -> None:
    """A mid-stream litellm failure yields the synthesized response.failed +
    error *dicts* (no [DONE] — that is the SSE formatter's job)."""
    instance = seed_instance(session, alias="st-f", machine_uid="st-f-m")
    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    scheduler = InferenceScheduler(aredis)

    async def raise_stream(*args, **kwargs):  # noqa: ARG001
        async def gen():
            yield {
                "type": "response.created",
                "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
            }
            raise litellm.exceptions.APIError(
                status_code=500,
                message="boom",
                llm_provider="openai",
                model="st-f",
            )

        return gen()

    monkeypatch.setattr(litellm, "aresponses", raise_stream)

    emitter = SSEEmitter("resp_stf000")
    items = [
        item
        async for item in stream_turn_events(
            emitter=emitter,
            scheduler=scheduler,
            alias="st-f",
            request_id="req-stf",
            client_response_id="resp_stf000",
            previous_response_id=None,
            input_items_for_record=[{"role": "user", "content": "x"}],
            definition_id=instance.provider_definition_id,
            litellm_input=[{"role": "user", "content": "x"}],
            passthrough={},
            base_params={"model": "st-f", "provider_type": "mock"},
            store=True,
        )
    ]
    types = [it["type"] for it in items]
    assert types == ["response.created", "response.failed", "error"]
    assert items[1]["response"]["error"]["code"] == "upstream_failed"
    assert [it["sequence_number"] for it in items] == [0, 1, 2]
    await aredis.aclose()


def test_keepalive_sentinel_identity_and_repr() -> None:
    assert repr(KEEPALIVE) == "<KEEPALIVE>"
    assert KEEPALIVE is route_mod.KEEPALIVE
    assert isinstance(KEEPALIVE, Keepalive)


# ---------------------------------------------------------------------------
# End-to-end byte-identity: the SSE formatter over the runner matches the
# emitter.frame path exactly (guards the split against drift).
# ---------------------------------------------------------------------------
def test_formatter_output_is_json_valid() -> None:
    """Sanity: serialized event dicts round-trip as the SSE data payload."""
    emitter = SSEEmitter(CLIENT_ID)
    data = emitter.event_data(
        {"type": "response.output_text.delta", "item_id": "m", "delta": "x"}
    )
    frame = _serialize(data)
    assert frame.startswith("event: response.output_text.delta\ndata: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["delta"] == "x"
    assert payload["sequence_number"] == 0


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.created", "response": {"id": WRAPPED_ID}},
        {"type": "response.output_text.delta", "delta": "z"},
        {"type": "response.completed", "response": {"id": WRAPPED_ID, "output": []}},
    ],
)
def test_event_data_serialize_equals_frame_parametrized(event: dict) -> None:
    f = SSEEmitter(CLIENT_ID)
    d = SSEEmitter(CLIENT_ID)
    assert _serialize(d.event_data(dict(event))) == f.frame(dict(event))
