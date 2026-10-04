"""Route integration for POST /v1/chat/completions with litellm monkeypatched.

Mirrors ``test_v1_responses.py``: a connected provider instance is seeded
directly in the UTF8 test DB so the real ``InferenceScheduler`` admits
against it, and ``litellm.acompletion`` is replaced with a fake emitting a
realistic chat.completion.chunk sequence. Asserts the admin-side chat
contract:

- admin-owned ``chatcmpl-<uuid>`` id on every chunk (never the upstream
  id), data-only SSE frames (no ``event:`` lines), ``[DONE]`` terminator;
- persistence with the ``api_format=chat_completions`` marker, aggregated
  assistant message as ``output_items``, token counts + TokenUsageSample;
- mid-stream exception -> chat-style error frame + [DONE] + failed record;
- client disconnect (generator aclose) -> cancellation-safe slot release
  + failed record;
- 404 unknown/disabled alias, 400 missing/empty messages, 503 no provider;
- scheduler acquire/release called exactly once per request.
"""

import asyncio
import json
import uuid
from typing import Any

import litellm
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.db import engine
from app.models import (
    Machine,
    ProviderDefinition,
    ProviderInstance,
    ResponseRecord,
    TokenUsageSample,
)
from app.services.alias_registry import ensure_registered
from app.services.scheduler import Admission

CLIENT_ID_PREFIX = "chatcmpl-"
UPSTREAM_ID = "chatcmpl-upstream000"


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
    enabled: bool = True,
    websocket_connected: bool = True,
) -> ProviderInstance:
    machine = session.query(Machine).filter(Machine.uid == machine_uid).first()
    if machine is None:
        machine = Machine(uid=machine_uid, name=f"name-{machine_uid}", host="127.0.0.1")
        session.add(machine)
    definition = ProviderDefinition(
        alias=alias,
        provider_type="mock",
        registration_token=f"tok-{alias}",
        enabled=enabled,
    )
    session.add(definition)
    session.commit()
    instance = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=definition.id,
        port=8081,
        backend_status="running",
        websocket_connected=websocket_connected,
    )
    session.add(instance)
    session.commit()
    session.refresh(instance)
    return instance


def chat_chunk(
    delta: dict, finish: str | None = None, usage: dict | None = None
) -> dict:
    c: dict[str, Any] = {
        "id": UPSTREAM_ID,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "ignored",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        c["usage"] = usage
    return c


def make_fake_stream(chunks: list[dict], captured: dict[str, Any]):
    """Build a fake ``litellm.acompletion``: awaitable returning an async
    iterator over ``chunks``, recording kwargs."""

    async def fake_acompletion(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)

        async def gen():
            for chunk in chunks:
                yield chunk

        return gen()

    return fake_acompletion


def parse_data_only_sse(text: str) -> tuple[list[dict], bool]:
    """Parse chat SSE: data-only frames. Returns (json payloads, saw_done).
    Fails the test if any ``event:`` line is present (chat spec is
    data-only)."""
    payloads: list[dict] = []
    saw_done = False
    for block in text.strip().split("\n\n"):
        for line in block.split("\n"):
            assert not line.startswith("event:"), (
                f"chat SSE must be data-only: {line!r}"
            )
            if line.startswith("data: "):
                data = line[len("data: ") :]
                if data == "[DONE]":
                    saw_done = True
                else:
                    payloads.append(json.loads(data))
    return payloads, saw_done


# ---------------------------------------------------------------------------
# Streaming happy path
# ---------------------------------------------------------------------------
def test_stream_happy_path(client: TestClient, session: Session, monkeypatch) -> None:
    instance = seed_instance(session, alias="ct-a", machine_uid="ct-m")
    usage = {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    chunks = [
        chat_chunk({"role": "assistant", "content": ""}),
        chat_chunk({"content": "hello "}),
        chat_chunk({"content": "there"}),
        chat_chunk({}, "stop", usage=usage),
    ]
    captured: dict[str, Any] = {}
    monkeypatch.setattr(litellm, "acompletion", make_fake_stream(chunks, captured))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ct-a",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    payloads, saw_done = parse_data_only_sse(resp.text)
    assert saw_done
    assert len(payloads) == 4

    # Admin-owned id on every chunk; upstream id never leaks.
    client_id = payloads[0]["id"]
    assert client_id.startswith(CLIENT_ID_PREFIX)
    for p in payloads:
        assert p["id"] == client_id
        assert p["id"] != UPSTREAM_ID

    # litellm called with our base_url + /v1, native chat, forced usage.
    assert captured["api_base"] == "http://127.0.0.1:8081/v1"
    assert captured["custom_llm_provider"] == "openai"
    assert captured["stream"] is True
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["stream_options"]["include_usage"] is True
    assert captured["_skip_responses_api_bridge"] is True

    # Persisted ResponseRecord with the chat marker.
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == client_id)
        .first()
    )
    assert record is not None
    assert record.status == "completed"
    assert record.parameters["api_format"] == "chat_completions"
    assert record.parameters["litellm_id"] == UPSTREAM_ID
    assert record.input_items == [{"role": "user", "content": "hi"}]
    assert record.output_items == [{"role": "assistant", "content": "hello there"}]
    assert record.provider_definition_id == instance.provider_definition_id
    assert record.provider_instance_id == instance.id
    assert record.input_tokens == 9
    assert record.output_tokens == 3
    assert record.total_tokens == 12

    sample = (
        session.query(TokenUsageSample)
        .filter(TokenUsageSample.provider_instance_id == instance.id)
        .first()
    )
    assert sample is not None
    assert sample.prompt_tokens == 9
    assert sample.cached_tokens == 2
    assert sample.completion_tokens == 3

    scheduler = client.app.state.scheduler
    assert scheduler.active_count("ct-a") == 0


def test_stream_merges_tool_call_deltas(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ct-tool", machine_uid="ct-tool-m")
    chunks = [
        chat_chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ],
            }
        ),
        chat_chunk(
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}
        ),
        chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"SF"}'}}]}),
        chat_chunk(
            {},
            "tool_calls",
            usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        ),
    ]
    monkeypatch.setattr(litellm, "acompletion", make_fake_stream(chunks, {}))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ct-tool",
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    payloads, _ = parse_data_only_sse(resp.text)
    client_id = payloads[0]["id"]
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == client_id)
        .first()
    )
    assert record is not None
    assert record.status == "completed"
    assert record.output_items == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Non-stream (explicit and default)
# ---------------------------------------------------------------------------
def _fake_json_completion(captured: dict[str, Any]):
    final = {
        "id": UPSTREAM_ID,
        "object": "chat.completion",
        "created": 1700000000,
        "model": "ignored",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 1},
        },
    }

    async def fake_acompletion(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return final

    return fake_acompletion


@pytest.mark.parametrize("stream_payload", [{"stream": False}, {}])
def test_non_stream_returns_json(
    client: TestClient, session: Session, monkeypatch, stream_payload: dict
) -> None:
    seed_instance(session, alias=f"ct-ns-{bool(stream_payload)}", machine_uid="ct-ns-m")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(litellm, "acompletion", _fake_json_completion(captured))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": f"ct-ns-{bool(stream_payload)}",
            "messages": [{"role": "user", "content": "hi"}],
            **stream_payload,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"].startswith(CLIENT_ID_PREFIX)
    assert body["id"] != UPSTREAM_ID
    assert body["choices"][0]["message"]["content"] == "hi back"
    assert captured["stream"] is False

    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == body["id"])
        .first()
    )
    assert record is not None
    assert record.status == "completed"
    assert record.parameters["api_format"] == "chat_completions"
    assert record.parameters["litellm_id"] == UPSTREAM_ID
    assert record.output_items == [{"role": "assistant", "content": "hi back"}]
    assert record.input_tokens == 5
    assert record.output_tokens == 2

    scheduler = client.app.state.scheduler
    assert scheduler.active_count(f"ct-ns-{bool(stream_payload)}") == 0


# ---------------------------------------------------------------------------
# Mid-stream failure -> chat-style error frame
# ---------------------------------------------------------------------------
def test_midstream_failure_error_frame_and_failed_record(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ct-fail", machine_uid="ct-fail-m")

    async def failing_acompletion(*args, **kwargs):  # noqa: ARG001
        async def gen():
            yield chat_chunk({"role": "assistant", "content": "partial"})
            raise litellm.exceptions.MidStreamFallbackError(
                "provider reported failure",
                "ct-fail",
                "openai",
                original_exception=litellm.exceptions.InternalServerError(
                    message="boom", model="ct-fail", llm_provider="openai"
                ),
            )

        return gen()

    monkeypatch.setattr(litellm, "acompletion", failing_acompletion)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ct-fail",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    blocks = [b for b in resp.text.strip().split("\n\n") if b]
    assert blocks[-1] == "data: [DONE]"
    error_payload = json.loads(blocks[-2][len("data: ") :])
    assert error_payload["error"]["code"] == "upstream_failed"

    first_payload = json.loads(blocks[0][len("data: ") :])
    client_id = first_payload["id"]
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == client_id)
        .first()
    )
    assert record is not None
    assert record.status == "failed"
    assert record.error_code == "upstream_failed"
    assert record.output_items == []

    scheduler = client.app.state.scheduler
    assert scheduler.active_count("ct-fail") == 0


# ---------------------------------------------------------------------------
# Client disconnect (generator aclose) releases slot + persists failed
# ---------------------------------------------------------------------------
async def test_stream_generator_aclose_releases_and_persists_failed(
    session, monkeypatch
) -> None:
    """Same cancellation-safe pattern as Phase 6: drive the route generator
    directly and aclose it mid-stream."""
    from app.api.v1 import chat_completions as route_mod
    from app.services.scheduler import InferenceScheduler

    instance = seed_instance(session, alias="ct-gen", machine_uid="ct-gen-m")
    definition_id = instance.provider_definition_id

    async def hang_acompletion(*args, **kwargs):  # noqa: ARG001
        async def gen():
            yield chat_chunk({"role": "assistant", "content": "first"})
            await asyncio.sleep(120)

        return gen()

    monkeypatch.setattr(litellm, "acompletion", hang_acompletion)

    import os

    import redis.asyncio as aioredis

    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    scheduler = InferenceScheduler(aredis)
    admission = await scheduler.acquire("ct-gen", "req-gen")

    agen = route_mod._stream_chat(
        scheduler=scheduler,
        alias="ct-gen",
        request_id="req-gen",
        client_completion_id="chatcmpl-gen000",
        input_items_for_record=[{"role": "user", "content": "x"}],
        definition_id=definition_id,
        admission=admission,
        messages=[{"role": "user", "content": "x"}],
        passthrough={},
        stream_options={"include_usage": True},
        base_params={
            "model": "ct-gen",
            "provider_type": "mock",
            "api_format": "chat_completions",
        },
    )
    first = await agen.__anext__()
    assert "chatcmpl-gen000" in first
    assert scheduler.active_count("ct-gen") == 1

    await agen.aclose()

    assert scheduler.active_count("ct-gen") == 0
    assert await aredis.smembers("im:sched:active:ct-gen") == set()
    with Session(engine) as check:
        rec = check.exec(
            select(ResponseRecord).where(
                ResponseRecord.response_id == "chatcmpl-gen000"
            )
        ).first()
        assert rec is not None
        assert rec.status == "failed"
        assert rec.error_code == "client_disconnected"
    await aredis.aclose()


# ---------------------------------------------------------------------------
# Validation / admission errors
# ---------------------------------------------------------------------------
def test_unknown_alias_404(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 404


def test_disabled_alias_404(client: TestClient, session: Session) -> None:
    seed_instance(session, alias="ct-off", machine_uid="ct-off-m", enabled=False)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "ct-off", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 404


def test_no_connected_provider_503(client: TestClient, session: Session) -> None:
    machine = Machine(uid="ct-np", name="ct-np", host="127.0.0.1")
    definition = ProviderDefinition(
        alias="ct-np", provider_type="mock", registration_token="tok-ct-np"
    )
    session.add_all([machine, definition])
    session.commit()
    instance = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=definition.id,
        websocket_connected=False,
        backend_status="stopped",
    )
    session.add(instance)
    session.commit()
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "ct-np", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "user", "content": "x"}]},  # no model
        {"model": "ct-v", "messages": []},  # empty messages
        {"model": "ct-v", "messages": "hi"},  # not a list
        {"model": "ct-v"},  # no messages
    ],
)
def test_bad_request_400(client: TestClient, body: dict) -> None:
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Scheduler interaction: acquire with alias, release exactly once
# ---------------------------------------------------------------------------
def test_scheduler_acquire_release_once(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="ct-cnt", machine_uid="ct-cnt-m")
    chunks = [
        chat_chunk(
            {"role": "assistant", "content": "x"}, "stop", usage={"total_tokens": 1}
        )
    ]
    monkeypatch.setattr(litellm, "acompletion", make_fake_stream(chunks, {}))

    scheduler = client.app.state.scheduler
    acquire_calls: list[tuple[str, str]] = []
    release_calls: list[tuple[str, str]] = []
    real_acquire, real_release = scheduler.acquire, scheduler.release

    async def spy_acquire(alias: str, request_id: str):
        acquire_calls.append((alias, request_id))
        return await real_acquire(alias, request_id)

    async def spy_release(alias: str, request_id: str):
        release_calls.append((alias, request_id))
        return await real_release(alias, request_id)

    monkeypatch.setattr(scheduler, "acquire", spy_acquire)
    monkeypatch.setattr(scheduler, "release", spy_release)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ct-cnt",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert [a for a, _ in acquire_calls] == ["ct-cnt"]
    assert [(a, r) for a, r in release_calls] == [(a, r) for a, r in acquire_calls]
    assert scheduler.active_count("ct-cnt") == 0


# ---------------------------------------------------------------------------
# Native-streaming registration probe
# ---------------------------------------------------------------------------
def test_ensure_registered_enables_native_chat_streaming() -> None:
    """What the chat route relies on in litellm's model table:

    - ``supports_native_streaming`` True -> acompletion/aresponses select
      real SSE instead of fake-streaming (``should_fake_stream`` -> False).
    - ``mode`` == "chat" (NOT "responses") -> ``acompletion`` stays on
      the native chat-completions path; a ``responses`` mode would bridge
      the call to /v1/responses (litellm/main.py responses_api_bridge_check).
    """
    alias = f"reg-probe-{uuid.uuid4().hex[:8]}"
    ensure_registered(alias)
    info = litellm.model_cost[alias]
    assert info["supports_native_streaming"] is True
    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "chat"
    from litellm.utils import supports_native_streaming

    assert supports_native_streaming(alias, "openai") is True
    # The responses-mode bridge must NOT fire for this alias.
    from litellm.main import responses_api_bridge_check

    bridged_info, _ = responses_api_bridge_check(
        model=alias, custom_llm_provider="openai", api_base="http://x/v1"
    )
    assert bridged_info.get("mode") != "responses"
    # ...and aresponses native streaming must still be selected (it keys
    # only on supports_native_streaming, never on mode). Guards the Phase 6
    # no-regression claim against the mode:chat union registration.
    from litellm.llms.openai.responses.transformation import (
        OpenAIResponsesAPIConfig,
    )

    assert (
        OpenAIResponsesAPIConfig().should_fake_stream(
            model=alias, stream=True, custom_llm_provider="openai"
        )
        is False
    )


async def test_non_stream_cancel_releases_slot(session, monkeypatch) -> None:
    """SF-1 regression: a client disconnect (CancelledError) during the
    non-stream `acompletion` await must still release the scheduler slot —
    `except Exception` does not catch cancellation; the finally does."""
    from app.api.v1 import chat_completions as route_mod
    from app.services.scheduler import InferenceScheduler

    instance = seed_instance(session, alias="ct-nsx", machine_uid="ct-nsx-m")

    async def cancel_acompletion(*args, **kwargs):  # noqa: ARG001
        raise asyncio.CancelledError

    monkeypatch.setattr(litellm, "acompletion", cancel_acompletion)

    import os

    import redis.asyncio as aioredis

    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("ct-nsx", "req-nsx")
    assert scheduler.active_count("ct-nsx") == 1

    with pytest.raises(asyncio.CancelledError):
        await route_mod._non_stream_chat(
            scheduler=scheduler,
            alias="ct-nsx",
            request_id="req-nsx",
            client_completion_id="chatcmpl-nsx000",
            input_items_for_record=[{"role": "user", "content": "x"}],
            definition_id=instance.provider_definition_id,
            admission=Admission(
                instance_id=str(instance.id),
                base_url="http://127.0.0.1:8081",
                machine_uid="ct-nsx-m",
            ),
            messages=[{"role": "user", "content": "x"}],
            passthrough={},
            base_params={"model": "ct-nsx", "api_format": "chat_completions"},
        )

    # The slot is released despite the cancellation.
    assert scheduler.active_count("ct-nsx") == 0
    assert await aredis.smembers("im:sched:active:ct-nsx") == set()
    await aredis.aclose()
