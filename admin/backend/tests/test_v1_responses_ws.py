"""Phase 21 S2: Responses-over-WebSocket transport (``WS /v1/responses``).

Drives the endpoint through the starlette ``TestClient`` websocket with the
same fake-litellm / seeded-instance fixtures the SSE route tests use, and
asserts the locked-decision contract: dict events with monotonic sequence and
no ``[DONE]``, sequential turns, the per-connection continuation cache
(store=false) vs the store=true Postgres fallback, call_id validation +
eviction, cold-boot lifecycle synthesis with provider-duplicate drop, the
error-envelope taxonomy, and the 60-minute limit.
"""

import asyncio
import base64
import json
import uuid
from typing import Any

import litellm
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.v1 import responses as route_mod
from app.api.v1 import responses_ws as route_mod_ws
from app.api.v1.responses import build_litellm_input
from app.models import ResponseRecord
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

WRAPPED_ID = "resp_bGl0ZWxsbQp0b3kxMjM"
TERMINAL_TYPES = ("response.completed", "response.incomplete", "response.failed")


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
    capacity: int = 1,
    enabled: bool = True,
) -> None:
    machine = get_or_create_machine(session, uid=machine_uid, host="127.0.0.1")
    agent = make_agent(session, machine, connected=True)
    definition = make_definition(
        session,
        alias=alias,
        capacity=capacity,
        vram_required=0,
        enabled=enabled,
        backend_config={"model": {"file": "m.gguf"}},
    )
    make_instance(session, agent, definition, backend_status="running")


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


def created_event() -> dict:
    return {
        "type": "response.created",
        "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
    }


def make_recording_stream(events: list[dict], calls: list[dict[str, Any]]):
    """Fake ``litellm.aresponses`` that appends its kwargs per call and yields
    ``events`` (so a two-turn socket can assert the second call's input)."""

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        calls.append(dict(kwargs))

        async def gen():
            for event in events:
                yield event

        return gen()

    return fake_aresponses


def drain_turn(ws: TestClient, *, extra: int = 0) -> list[dict]:
    """Read frames until the first terminal/error event (plus ``extra`` more).

    The app blocks on ``receive`` after a turn ends, so we must stop at the
    terminal frame; ``extra`` lets a caller also pull the trailing ``error``
    event that follows a ``response.failed``.
    """
    events: list[dict] = []
    while True:
        msg = ws.receive_json()
        events.append(msg)
        etype = msg.get("type")
        if etype in TERMINAL_TYPES or etype == "error":
            break
    for _ in range(extra):
        events.append(ws.receive_json())
    return events


def _seqs(events: list[dict]) -> list[int]:
    return [e["sequence_number"] for e in events if "sequence_number" in e]


# ---------------------------------------------------------------------------
# (1) happy single turn
# ---------------------------------------------------------------------------
def test_ws_happy_single_turn(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-a", machine_uid="ws-a-m")
    output = [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        }
    ]
    events = [
        created_event(),
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
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, []))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-a", "input": "hello"})
        frames = drain_turn(ws)

    # Every frame is a dict (never the SSE "[DONE]" string).
    assert all(isinstance(f, dict) for f in frames)
    types = [f["type"] for f in frames]
    assert types == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    # Monotonic sequence from 0.
    assert _seqs(frames) == [0, 1, 2]
    # Admin owns the client id on lifecycle frames.
    assert frames[0]["response"]["id"].startswith("resp_")
    assert frames[-1]["response"]["id"] == frames[0]["response"]["id"]
    assert frames[-1]["response"]["status"] == "completed"


# ---------------------------------------------------------------------------
# (2) sequential two turns on one socket
# ---------------------------------------------------------------------------
def test_ws_sequential_two_turns(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-2", machine_uid="ws-2-m")
    events = [
        created_event(),
        completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, []))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-2", "input": "one"})
        first = drain_turn(ws)
        ws.send_json({"type": "response.create", "model": "ws-2", "input": "two"})
        second = drain_turn(ws)

    assert first[-1]["type"] == "response.completed"
    assert second[-1]["type"] == "response.completed"
    # Each turn restarts its own emitter sequence at 0.
    assert _seqs(first)[0] == 0
    assert _seqs(second)[0] == 0


# ---------------------------------------------------------------------------
# (3) store=false continuation via the per-connection cache
# ---------------------------------------------------------------------------
def test_ws_store_false_continuation_from_cache(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-c", machine_uid="ws-c-m")
    prior_output = [
        {
            "id": "msg_p",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "code word: ember"}],
        }
    ]
    events = [
        created_event(),
        completed_event(
            prior_output, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        ),
    ]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, calls))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-c",
                "store": False,
                "input": "what is the code word?",
            }
        )
        first = drain_turn(ws)
        prior_id = first[-1]["response"]["id"]

        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-c",
                "store": False,
                "previous_response_id": prior_id,
                "input": "reply with only the code word",
            }
        )
        drain_turn(ws)

    # The store=false turn was never resolvable from Postgres; the cache made
    # the continuation work, and the second litellm call carried the prior
    # turn's accumulated input + output prepended.
    assert len(calls) == 2
    second_input = calls[1]["input"]
    assert second_input[0] == {"role": "user", "content": "what is the code word?"}
    assert second_input[1] == prior_output[0]
    assert second_input[-1] == {
        "role": "user",
        "content": "reply with only the code word",
    }


# ---------------------------------------------------------------------------
# (4) store=false unknown id -> previous_response_not_found, socket usable
# ---------------------------------------------------------------------------
def test_ws_store_false_unknown_id_then_fresh_turn(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-u", machine_uid="ws-u-m")
    events = [
        created_event(),
        completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, []))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-u",
                "store": False,
                "previous_response_id": "resp_" + "f" * 32,
                "input": "continue",
            }
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["status"] == 404
        assert err["error"]["code"] == "previous_response_not_found"
        assert err["error"]["param"] == "previous_response_id"

        # Socket stays open: a fresh turn (no previous id) completes.
        ws.send_json({"type": "response.create", "model": "ws-u", "input": "hello"})
        frames = drain_turn(ws)
    assert frames[-1]["type"] == "response.completed"


# ---------------------------------------------------------------------------
# (5) store=true continuation works from the DB after a reconnect
# ---------------------------------------------------------------------------
def test_ws_store_true_continuation_across_sockets(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-db", machine_uid="ws-db-m")
    prior_output = [
        {
            "id": "msg_d",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "stored"}],
        }
    ]
    events = [
        created_event(),
        completed_event(
            prior_output, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        ),
    ]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, calls))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-db",
                "store": True,
                "input": "remember this",
            }
        )
        first = drain_turn(ws)
        prior_id = first[-1]["response"]["id"]

    # New socket: the store=true record is resolved from Postgres.
    with client.websocket_connect("/v1/responses") as ws2:
        ws2.send_json(
            {
                "type": "response.create",
                "model": "ws-db",
                "store": True,
                "previous_response_id": prior_id,
                "input": "continue",
            }
        )
        drain_turn(ws2)

    assert calls[1]["input"][0] == {"role": "user", "content": "remember this"}
    assert calls[1]["input"][1] == prior_output[0]


# ---------------------------------------------------------------------------
# (5b) REGRESSION: store=true persist survives an immediate post-terminal
# disconnect (the CI race). Without the terminal_sent guard the racer cancels
# the turn while it is parked in the runner's shielded release, skipping
# persist_turn_logged -> the next socket's Postgres lookup misses.
# ---------------------------------------------------------------------------
def test_ws_store_true_persist_survives_immediate_disconnect(
    client: TestClient, session: Session, monkeypatch
) -> None:
    import time as _time

    from sqlalchemy import select as _select
    from sqlmodel import Session as _Session

    from app.core.db import engine

    def _record_present(rid: str) -> bool:
        with _Session(engine) as s:
            return (
                s.exec(
                    _select(ResponseRecord).where(ResponseRecord.response_id == rid)
                ).first()
                is not None
            )

    seed_instance(session, alias="ws-race", machine_uid="ws-race-m")
    scheduler = client.app.state.scheduler

    prior_output = [
        {
            "id": "msg_r",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "stored before disconnect"}],
        }
    ]
    events = [
        created_event(),
        completed_event(
            prior_output, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        ),
    ]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, calls))

    # Widen the terminal->persist window deterministically: the runner's
    # finally awaits scheduler.release (shielded) BEFORE persist_turn_logged,
    # so a slow release parks the committed turn exactly where a cancel would
    # orphan the cleanup and drop the record.
    original_release = scheduler.release

    async def slow_release(alias: str, request_id: str) -> None:
        await asyncio.sleep(0.3)
        await original_release(alias, request_id)

    monkeypatch.setattr(scheduler, "release", slow_release)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-race",
                "store": True,
                "input": "remember this",
            }
        )
        first = drain_turn(ws)
        prior_id = first[-1]["response"]["id"]
        # Deliver the disconnect while the committed turn is still finishing its
        # release + persist -- the race trigger. close() queues a
        # websocket.disconnect without tearing down the handler's task scope, so
        # the racer sees it exactly as a real server would.
        ws.close()
        # Poll for the persisted record: with the terminal_sent guard the racer
        # awaits the committed turn and persist lands; without it the turn is
        # cancelled and persist_turn_logged is skipped, so this never appears.
        deadline = _time.monotonic() + 5.0
        while _time.monotonic() < deadline and not _record_present(prior_id):
            _time.sleep(0.05)
        assert _record_present(prior_id), "store=true turn was not persisted"

    # A brand-new socket must resolve the store=true chain from Postgres.
    with client.websocket_connect("/v1/responses") as ws2:
        ws2.send_json(
            {
                "type": "response.create",
                "model": "ws-race",
                "store": True,
                "previous_response_id": prior_id,
                "input": "continue",
            }
        )
        frames2 = drain_turn(ws2)

    assert frames2[-1]["type"] == "response.completed", frames2[-1]
    assert len(calls) == 2
    assert calls[1]["input"][0] == {"role": "user", "content": "remember this"}
    assert calls[1]["input"][1] == prior_output[0]


# ---------------------------------------------------------------------------
# (6) call_id validation failure -> 400 + eviction -> retry -> not found
# ---------------------------------------------------------------------------
def test_ws_call_id_validation_evicts_previous(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-cid", machine_uid="ws-cid-m")
    fn_call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "f",
        "arguments": "{}",
    }
    events = [
        created_event(),
        completed_event(
            [fn_call], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        ),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, []))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-cid",
                "store": False,
                "input": "call the tool",
            }
        )
        first = drain_turn(ws)
        prior_id = first[-1]["response"]["id"]

        # A function_call_output whose call_id is not in the previous turn.
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-cid",
                "store": False,
                "previous_response_id": prior_id,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_bad",
                        "output": "x",
                    }
                ],
            }
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["status"] == 400
        assert err["error"]["code"] == "invalid_request"

        # The bad continuation evicted the referenced id -> retry misses.
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-cid",
                "store": False,
                "previous_response_id": prior_id,
                "input": "try again",
            }
        )
        err2 = ws.receive_json()
    assert err2["type"] == "error"
    assert err2["error"]["code"] == "previous_response_not_found"


def test_ws_valid_call_id_continuation_succeeds(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-cid2", machine_uid="ws-cid2-m")
    fn_call = {
        "type": "function_call",
        "call_id": "call_ok",
        "name": "f",
        "arguments": "{}",
    }
    events = [
        created_event(),
        completed_event(
            [fn_call], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        ),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, []))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-cid2",
                "store": False,
                "input": "call the tool",
            }
        )
        first = drain_turn(ws)
        prior_id = first[-1]["response"]["id"]
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-cid2",
                "store": False,
                "previous_response_id": prior_id,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_ok",
                        "output": "result",
                    }
                ],
            }
        )
        frames = drain_turn(ws)
    assert frames[-1]["type"] == "response.completed"


# ---------------------------------------------------------------------------
# (7) provider mid-turn failure -> failed + error dicts forwarded
# ---------------------------------------------------------------------------
def test_ws_provider_failure_forwards_failed_and_error(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-f", machine_uid="ws-f-m")

    async def failing_aresponses(*args, **kwargs):  # noqa: ARG001

        async def gen():
            yield created_event()
            raise litellm.exceptions.APIError(
                status_code=500, message="boom", llm_provider="openai", model="ws-f"
            )

        return gen()

    monkeypatch.setattr(litellm, "aresponses", failing_aresponses)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-f", "input": "x"})
        frames = drain_turn(ws, extra=1)

    types = [f["type"] for f in frames]
    assert types == ["response.created", "response.failed", "error"]
    assert frames[1]["response"]["error"]["code"] == "upstream_failed"
    assert _seqs(frames) == [0, 1, 2]


# ---------------------------------------------------------------------------
# (8) error-envelope taxonomy (socket stays open)
# ---------------------------------------------------------------------------
def test_ws_bad_json_envelope(client: TestClient, session: Session) -> None:
    seed_instance(session, alias="ws-bj", machine_uid="ws-bj-m")
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text("this is not json")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["status"] == 400
        assert err["error"]["code"] == "invalid_request"


def test_ws_stream_flag_rejected(client: TestClient, session: Session) -> None:
    seed_instance(session, alias="ws-sf", machine_uid="ws-sf-m")
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {"type": "response.create", "model": "ws-sf", "input": "x", "stream": True}
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["error"]["code"] == "invalid_request"
        assert err["error"]["param"] == "stream"


def test_ws_missing_model_envelope(client: TestClient, session: Session) -> None:
    seed_instance(session, alias="ws-mm", machine_uid="ws-mm-m")
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "input": "x"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["status"] == 400
        assert err["error"]["code"] == "invalid_request"


def test_ws_unknown_alias_envelope(client: TestClient) -> None:
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "nope", "input": "x"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["status"] == 404
        assert err["error"]["code"] == "model_not_found"


def test_ws_wrong_type_envelope(client: TestClient, session: Session) -> None:
    seed_instance(session, alias="ws-wt", machine_uid="ws-wt-m")
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.ping", "model": "ws-wt", "input": "x"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# (9) cold boot: synthesized created/in_progress + provider-duplicate drop
# ---------------------------------------------------------------------------
def test_ws_cold_boot_synthesizes_lifecycle_and_drops_duplicates(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-cb", machine_uid="ws-cb-m")
    monkeypatch.setattr(route_mod, "KEEPALIVE_INTERVAL_SECONDS", 0.05)
    scheduler = client.app.state.scheduler
    original_acquire = scheduler.acquire

    async def slow_acquire(alias, request_id):
        await asyncio.sleep(0.2)  # force several KEEPALIVE yields
        return await original_acquire(alias, request_id)

    monkeypatch.setattr(scheduler, "acquire", slow_acquire)

    # The provider emits its own created + in_progress, which must be dropped
    # after the synthesized pair.
    events = [
        created_event(),
        {
            "type": "response.in_progress",
            "response": {"id": WRAPPED_ID, "status": "in_progress", "output": []},
        },
        completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, []))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-cb", "input": "x"})
        frames = drain_turn(ws)

    types = [f["type"] for f in frames]
    # Exactly one created and one in_progress (synthesized), then completed.
    assert types == ["response.created", "response.in_progress", "response.completed"]
    # Sequence is monotonic (gaps from dropped provider frames are allowed).
    seqs = _seqs(frames)
    assert seqs == sorted(seqs)
    assert frames[0]["response"]["id"].startswith("resp_")


# ---------------------------------------------------------------------------
# (10) 60-minute connection limit
# ---------------------------------------------------------------------------
def test_ws_connection_limit(client: TestClient, session: Session, monkeypatch) -> None:
    seed_instance(session, alias="ws-lim", machine_uid="ws-lim-m")
    monkeypatch.setattr(route_mod_ws, "WS_CONNECTION_LIMIT_SECONDS", 0.0)
    with client.websocket_connect("/v1/responses") as ws:
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["status"] == 400
        assert err["error"]["code"] == "websocket_connection_limit_reached"


# ---------------------------------------------------------------------------
# (11) compaction item decoding in build_litellm_input (both transports)
# ---------------------------------------------------------------------------
def _encode(payload: dict) -> str:
    return base64.b64encode(json.dumps(payload).encode()).decode()


def test_build_litellm_input_decodes_compaction() -> None:
    compaction = {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": _encode({"summary": "user likes teal", "source_count": 3}),
    }
    body = {"input": [compaction, {"role": "user", "content": "next"}]}
    result = build_litellm_input(body, None)
    assert result[0] == {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "[Compacted context]\nuser likes teal"}
        ],
    }
    assert result[1] == {"role": "user", "content": "next"}


def test_build_litellm_input_drops_undecodable_compaction() -> None:
    bad = {
        "type": "compaction",
        "id": "cmp_bad",
        "encrypted_content": "!!!not-base64!!!",
    }
    good = {
        "type": "compaction",
        "id": "cmp_ok",
        "encrypted_content": _encode({"summary": "s", "source_count": 1}),
    }
    body = {"input": [bad, good, {"role": "user", "content": "tail"}]}
    result = build_litellm_input(body, None)
    # The undecodable compaction is dropped; the good one decoded; tail kept.
    assert len(result) == 2
    assert result[0]["content"][0]["text"] == "[Compacted context]\ns"
    assert result[1] == {"role": "user", "content": "tail"}


def test_build_litellm_input_compaction_on_continuation_prepends_chain() -> None:
    prior = ResponseRecord(
        response_id="resp_" + uuid.uuid4().hex,
        input_items=[{"role": "user", "content": "old"}],
        output_items=[],
        provider_definition_id=uuid.uuid4(),
        status="completed",
    )
    compaction = {
        "type": "compaction",
        "encrypted_content": _encode({"summary": "sum", "source_count": 1}),
    }
    result = build_litellm_input({"input": [compaction]}, prior)
    assert result[0] == {"role": "user", "content": "old"}
    assert result[1]["content"][0]["text"] == "[Compacted context]\nsum"


# ---------------------------------------------------------------------------
# (12) per-message 60-minute check (distinct from the idle-timer path)
# ---------------------------------------------------------------------------
def test_ws_connection_limit_per_message(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """The timer path is covered by the limit=0 test; this drives the
    per-message elapsed check specifically: the background limiter is parked on
    a long real sleep while a monkeypatched monotonic clock jumps past the cap
    between the connection opening and the first message."""
    seed_instance(session, alias="ws-lim2", machine_uid="ws-lim2-m")
    clock = {"t": 0.0}
    monkeypatch.setattr(route_mod_ws, "_now", lambda: clock["t"])
    monkeypatch.setattr(route_mod_ws, "WS_CONNECTION_LIMIT_SECONDS", 3600.0)

    with client.websocket_connect("/v1/responses") as ws:
        # start was captured as 0.0; jump past the cap before the message.
        clock["t"] = 4000.0
        ws.send_json({"type": "response.create", "model": "ws-lim2", "input": "x"})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["error"]["code"] == "websocket_connection_limit_reached"


# ---------------------------------------------------------------------------
# (13) mid-turn client disconnect releases the slot (decision 0)
# ---------------------------------------------------------------------------
def test_ws_mid_turn_disconnect_releases_slot(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-disc", machine_uid="ws-disc-m")
    scheduler = client.app.state.scheduler

    async def stalled_aresponses(*args, **kwargs):  # noqa: ARG001

        async def gen():
            yield created_event()
            await asyncio.sleep(1000)  # provider hangs after the first event

        return gen()

    monkeypatch.setattr(litellm, "aresponses", stalled_aresponses)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-disc", "input": "x"})
        first = ws.receive_json()
        assert first["type"] == "response.created"
        # Slot is held while the turn is in flight.
        assert scheduler.active_count("ws-disc") == 1
        # Exiting the context sends a websocket.disconnect mid-turn.

    # The disconnect cancelled the turn; the runner's shielded finally released.
    assert scheduler.active_count("ws-disc") == 0


# ---------------------------------------------------------------------------
# (14) a failed FRESH turn (no previous_response_id) is not cached
# ---------------------------------------------------------------------------
def test_ws_failed_fresh_turn_not_cached(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-ff", machine_uid="ws-ff-m")

    async def failing_aresponses(*args, **kwargs):  # noqa: ARG001

        async def gen():
            yield created_event()
            raise litellm.exceptions.APIError(
                status_code=500, message="boom", llm_provider="openai", model="ws-ff"
            )

        return gen()

    monkeypatch.setattr(litellm, "aresponses", failing_aresponses)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {"type": "response.create", "model": "ws-ff", "store": False, "input": "x"}
        )
        frames = drain_turn(ws, extra=1)
        assert frames[1]["type"] == "response.failed"
        failed_id = frames[1]["response"]["id"]

        # The failed turn produced no cache entry, so continuing from its id
        # (store=false, so also absent from the store=true DB fallback) misses.
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-ff",
                "store": False,
                "previous_response_id": failed_id,
                "input": "continue",
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["error"]["code"] == "previous_response_not_found"


# ---------------------------------------------------------------------------
# (15) zero-candidate pre-check -> 503 no_provider envelope (socket stays open)
# ---------------------------------------------------------------------------
def test_ws_no_provider_envelope(client: TestClient, session: Session) -> None:
    machine = get_or_create_machine(session, uid="ws-np-m", host="127.0.0.1")
    agent = make_agent(session, machine, connected=False)
    definition = make_definition(
        session, alias="ws-np", backend_config={"model": {"file": "m.gguf"}}
    )
    make_instance(session, agent, definition, backend_status="stopped")

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-np", "input": "x"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["status"] == 503
        assert err["error"]["code"] == "no_provider"


# ---------------------------------------------------------------------------
# (16) bounded LRU cache keeps only the most recent turns
# ---------------------------------------------------------------------------
def test_ws_cache_is_bounded_lru(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-lru", machine_uid="ws-lru-m")
    events = [
        created_event(),
        completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, []))

    ids: list[str] = []
    with client.websocket_connect("/v1/responses") as ws:
        for _ in range(route_mod_ws.WS_CACHE_MAX_TURNS + 3):
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "ws-lru",
                    "store": False,
                    "input": "x",
                }
            )
            frames = drain_turn(ws)
            ids.append(frames[-1]["response"]["id"])
        # The oldest turns were evicted (LRU); the most recent survive.
        oldest = ids[0]
        newest = ids[-1]
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-lru",
                "store": False,
                "previous_response_id": oldest,
                "input": "continue",
            }
        )
        evicted = ws.receive_json()
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-lru",
                "store": False,
                "previous_response_id": newest,
                "input": "continue",
            }
        )
        kept = drain_turn(ws)

    assert evicted["error"]["code"] == "previous_response_not_found"
    assert kept[-1]["type"] == "response.completed"


# ---------------------------------------------------------------------------
# (17) unexpected pre-turn exception -> generic 500 envelope, socket usable
# ---------------------------------------------------------------------------
def test_ws_internal_error_envelope_keeps_socket(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-500", machine_uid="ws-500-m")
    events = [
        created_event(),
        completed_event([], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    monkeypatch.setattr(litellm, "aresponses", make_recording_stream(events, []))

    # ensure_registered raises on the first call only (simulates an unexpected
    # pre-turn failure inside _run_turn), then delegates to a no-op.
    real = route_mod_ws.ensure_registered
    state = {"n": 0}

    def flaky(alias: str) -> None:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("kaboom")
        real(alias)

    monkeypatch.setattr(route_mod_ws, "ensure_registered", flaky)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-500", "input": "x"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["status"] == 500
        assert err["error"]["code"] == "server_error"
        # Generic message: the exception detail must not leak to the client.
        assert err["error"]["message"] == "internal server error"
        assert "kaboom" not in err["error"]["message"]

        # Socket stays usable: the next (non-raising) turn completes.
        ws.send_json({"type": "response.create", "model": "ws-500", "input": "y"})
        frames = drain_turn(ws)
    assert frames[-1]["type"] == "response.completed"


# ---------------------------------------------------------------------------
# (18) inbox overflow during a slow turn -> 400 "too many queued messages"
# ---------------------------------------------------------------------------
def test_ws_inbox_overflow_closes(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-of", machine_uid="ws-of-m")
    monkeypatch.setattr(route_mod_ws, "WS_INBOX_MAX", 1)

    async def slow_then_done(*args, **kwargs):  # noqa: ARG001

        async def gen():
            yield created_event()
            await asyncio.sleep(0.5)  # hold the turn open so strays queue up
            yield completed_event(
                [], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            )

        return gen()

    monkeypatch.setattr(litellm, "aresponses", slow_then_done)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-of", "input": "x"})
        first = ws.receive_json()
        assert first["type"] == "response.created"
        # Flood while the turn is stalled: the reader bounds the queue and
        # signals protocol abuse.
        ws.send_json({"type": "response.create", "model": "ws-of", "input": "a"})
        ws.send_json({"type": "response.create", "model": "ws-of", "input": "b"})
        # Drain until the abuse envelope arrives (after the turn settles).
        while True:
            msg = ws.receive_json()
            if msg.get("type") == "error":
                break
    assert msg["status"] == 400
    assert msg["error"]["code"] == "invalid_request"
    assert msg["error"]["message"] == "too many queued messages"


# ---------------------------------------------------------------------------
# (19) silence guard (Phase 21 addendum fix 2): a stalled provider stream
# fails the turn cleanly, releases the slot, and keeps the socket usable
# ---------------------------------------------------------------------------
def test_ws_silence_timeout_fails_turn_and_releases_slot(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-sil", machine_uid="ws-sil-m")
    scheduler = client.app.state.scheduler
    monkeypatch.setattr(route_mod_ws, "WS_TURN_SILENCE_SECONDS", 0.2)

    stall = asyncio.Event()  # never set: the first turn blocks forever
    calls: list[int] = []

    async def stalling_aresponses(*args, **kwargs):  # noqa: ARG001
        calls.append(1)
        stalled = len(calls) == 1

        async def gen():
            yield created_event()
            if stalled:
                await stall.wait()
            yield completed_event(
                [], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            )

        return gen()

    monkeypatch.setattr(litellm, "aresponses", stalling_aresponses)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-sil", "input": "x"})
        frames = drain_turn(ws, extra=1)
        assert [f["type"] for f in frames] == [
            "response.created",
            "response.failed",
            "error",
        ]
        # Same mapping shape as in-stream scheduler failures: server_error +
        # a specific code (here "timeout").
        err = frames[2]["error"]
        assert err["type"] == "server_error"
        assert err["code"] == "timeout"
        assert "silence" in err["message"]
        assert frames[1]["response"]["error"]["code"] == "timeout"
        assert _seqs(frames) == [0, 1, 2]
        # The stalled turn's slot was released by the runner's shielded
        # cleanup before the terminal frames reached the client.
        assert scheduler.active_count("ws-sil") == 0
        # Exactly ONE ResponseRecord for the timed-out id: the silence guard
        # persists once via the runner's shielded finally (no double-persist
        # from the handler side), and the persisted record carries the same
        # timeout error the client received (finding 3).
        timed_out_id = frames[1]["response"]["id"]
        from sqlmodel import select as _select

        from app.core.db import engine

        with Session(engine) as s:
            records = s.exec(
                _select(ResponseRecord).where(
                    ResponseRecord.response_id == timed_out_id
                )
            ).all()
        assert len(records) == 1, f"expected one record, got {len(records)}"
        assert records[0].status == "failed"
        assert records[0].error_code == "timeout"

        # Socket stays usable: the next turn completes.
        ws.send_json({"type": "response.create", "model": "ws-sil", "input": "y"})
        frames2 = drain_turn(ws)
    assert frames2[-1]["type"] == "response.completed"
    assert scheduler.active_count("ws-sil") == 0


# ---------------------------------------------------------------------------
# (19b) REGRESSION (finding 1): a provider that emits response.completed and
# then stalls the stream open must NOT trigger the silence guard into
# synthesizing a SECOND terminal (response.failed) to a client that already
# received its terminal.
# ---------------------------------------------------------------------------
def test_ws_silence_guard_does_not_double_terminal_after_completed(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-sil4", machine_uid="ws-sil4-m")
    scheduler = client.app.state.scheduler
    monkeypatch.setattr(route_mod_ws, "WS_TURN_SILENCE_SECONDS", 0.2)

    stall = asyncio.Event()  # never set: the provider hangs after its terminal

    async def stalling_after_terminal(*args, **kwargs):  # noqa: ARG001

        async def gen():
            yield created_event()
            yield completed_event(
                [], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            )
            await stall.wait()

        return gen()

    monkeypatch.setattr(litellm, "aresponses", stalling_after_terminal)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-sil4", "input": "x"})
        frames = drain_turn(ws)

    types = [f["type"] for f in frames]
    # Exactly one terminal (completed); no synthesized response.failed/error.
    assert types == ["response.created", "response.completed"]
    assert sum(1 for t in types if t in TERMINAL_TYPES) == 1
    # The stalled turn's slot was released via the runner's shielded aclose.
    assert scheduler.active_count("ws-sil4") == 0


# ---------------------------------------------------------------------------
# (20) silence guard does not fire on sub-timeout pauses
# ---------------------------------------------------------------------------
def test_ws_silence_guard_not_triggered_by_short_pause(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-sil3", machine_uid="ws-sil3-m")
    monkeypatch.setattr(route_mod_ws, "WS_TURN_SILENCE_SECONDS", 5.0)

    async def slow_aresponses(*args, **kwargs):  # noqa: ARG001

        async def gen():
            yield created_event()
            await asyncio.sleep(0.05)  # well under the guard
            yield completed_event(
                [], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            )

        return gen()

    monkeypatch.setattr(litellm, "aresponses", slow_aresponses)

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "ws-sil3", "input": "x"})
        frames = drain_turn(ws)
    assert frames[-1]["type"] == "response.completed"


# ---------------------------------------------------------------------------
# (21) a silence-timed-out CONTINUATION evicts the referenced previous id
# ---------------------------------------------------------------------------
def test_ws_silence_timeout_evicts_continuation(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ws-sil2", machine_uid="ws-sil2-m")
    monkeypatch.setattr(route_mod_ws, "WS_TURN_SILENCE_SECONDS", 0.2)

    stall = asyncio.Event()  # never set
    calls: list[int] = []

    async def stalling_aresponses(*args, **kwargs):  # noqa: ARG001
        calls.append(1)
        stalled = len(calls) == 2  # only the continuation turn stalls

        async def gen():
            yield created_event()
            if stalled:
                await stall.wait()
            yield completed_event(
                [], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            )

        return gen()

    monkeypatch.setattr(litellm, "aresponses", stalling_aresponses)

    with client.websocket_connect("/v1/responses") as ws:
        # Turn 1 (store=false): completes -> cached under its id.
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-sil2",
                "store": False,
                "input": "one",
            }
        )
        first = drain_turn(ws)
        prior_id = first[-1]["response"]["id"]

        # Turn 2: continuation from turn 1 stalls -> silence timeout.
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-sil2",
                "store": False,
                "previous_response_id": prior_id,
                "input": "two",
            }
        )
        frames = drain_turn(ws, extra=1)
        assert frames[1]["type"] == "response.failed"
        assert frames[2]["error"]["code"] == "timeout"

        # The failed continuation evicted turn 1's id (store=false, so the
        # Postgres fallback cannot resurrect it): turn 3 misses.
        ws.send_json(
            {
                "type": "response.create",
                "model": "ws-sil2",
                "store": False,
                "previous_response_id": prior_id,
                "input": "three",
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["status"] == 404
    assert err["error"]["code"] == "previous_response_not_found"
