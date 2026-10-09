"""Route integration for POST /v1/responses/compact (Phase 20).

A connected provider instance is seeded directly in the UTF8 test DB so the
real InferenceScheduler admits against it. ``litellm.aresponses`` is replaced
with a fake returning a terminal (non-stream) response, letting us assert the
admin-side CompactResource contract: the admin-owned ``cmp_`` id,
``object: "response.compaction"``, one ``compaction`` output item whose
base64 ``encrypted_content`` carries the summary, concrete usage, persistence
with ``parameters.api_format == "compaction"``, slot release, the error
taxonomy, previous_response_id reconstruction, and instructions merging.
"""

import base64
import json
import uuid
from typing import Any

import litellm
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import ProviderDefinition, ProviderInstance, ResponseRecord
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

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
    enabled: bool = True,
) -> ProviderInstance:
    machine = get_or_create_machine(session, uid=machine_uid, host="127.0.0.1")
    agent = make_agent(session, machine, connected=True)
    definition = make_definition(
        session,
        alias=alias,
        capacity=capacity,
        enabled=enabled,
        backend_config={"model": {"file": "m.gguf"}},
    )
    return make_instance(session, agent, definition, backend_status=backend_status)


def make_fake_response(final: dict[str, Any], captured: dict[str, Any]):
    """Build a fake non-stream ``litellm.aresponses`` returning ``final`` and
    recording the kwargs it was called with."""

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return final

    return fake_aresponses


def completed_final(summary: str, usage: dict[str, Any] | None) -> dict[str, Any]:
    final: dict[str, Any] = {
        "id": WRAPPED_ID,
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": summary}],
            }
        ],
    }
    if usage is not None:
        final["usage"] = usage
    return final


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_compact_happy_path(client: TestClient, session: Session, monkeypatch) -> None:
    instance = seed_instance(session, alias="cp-a", machine_uid="cp-m")
    usage = {
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
        "input_tokens_details": {"cached_tokens": 2},
        "output_tokens_details": {"reasoning_tokens": 1},
    }
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        litellm,
        "aresponses",
        make_fake_response(completed_final("compressed notes here", usage), captured),
    )

    resp = client.post("/v1/responses/compact", json={"model": "cp-a", "input": "hi"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["object"] == "response.compaction"
    assert body["id"].startswith("cmp_")
    assert body["id"] != WRAPPED_ID
    assert isinstance(body["created_at"], int)

    # Concrete usage with both *_tokens_details carrying required keys.
    out_usage = body["usage"]
    assert out_usage["input_tokens"] == 20
    assert out_usage["output_tokens"] == 5
    assert out_usage["total_tokens"] == 25
    assert "cached_tokens" in out_usage["input_tokens_details"]
    assert "reasoning_tokens" in out_usage["output_tokens_details"]

    # One compaction item with an id and decodable encrypted_content.
    assert len(body["output"]) == 1
    item = body["output"][0]
    assert item["type"] == "compaction"
    assert item["id"].startswith("cmpitem_")
    decoded = json.loads(base64.b64decode(item["encrypted_content"]))
    assert decoded["summary"] == "compressed notes here"
    assert decoded["source_count"] == 1  # the single wrapped user message

    # litellm was called non-stream against the admitted base_url + /v1.
    assert captured["stream"] is False
    assert captured["api_base"] == "http://127.0.0.1:8081/v1"
    assert captured["custom_llm_provider"] == "openai"
    assert "previous_response_id" not in captured

    # Persisted ResponseRecord tagged as a compaction turn.
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == body["id"])
        .first()
    )
    assert record is not None
    assert record.status == "completed"
    assert record.parameters["api_format"] == "compaction"
    assert record.parameters["stream"] is False
    assert record.parameters["litellm_response_id"] == WRAPPED_ID
    assert record.output_items == [item]
    assert record.provider_instance_id == instance.id
    assert record.input_tokens == 20
    assert record.output_tokens == 5

    # Slot released.
    scheduler = client.app.state.scheduler
    assert scheduler.active_count("cp-a") == 0


def test_compact_synthesizes_zeroed_usage_when_absent(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="cp-nu", machine_uid="cp-nu-m")
    monkeypatch.setattr(
        litellm,
        "aresponses",
        make_fake_response(completed_final("summary", None), {}),
    )

    resp = client.post("/v1/responses/compact", json={"model": "cp-nu", "input": "hi"})
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


# ---------------------------------------------------------------------------
# previous_response_id reconstruction + instructions merge
# ---------------------------------------------------------------------------
def test_compact_previous_response_id_reconstructs_input(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="cp-cont", machine_uid="cp-cont-m")
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
        .filter(ProviderDefinition.alias == "cp-cont")
        .first()
        .id,
        status="completed",
    )
    session.add(prior)
    session.commit()

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        litellm,
        "aresponses",
        make_fake_response(completed_final("s", {"input_tokens": 1}), captured),
    )

    resp = client.post(
        "/v1/responses/compact",
        json={
            "model": "cp-cont",
            "input": "second question",
            "previous_response_id": prior.response_id,
        },
    )
    assert resp.status_code == 200
    handed = captured["input"]
    assert isinstance(handed, list)
    assert handed[0] == {"role": "user", "content": "first question"}
    assert handed[1] == prior_output[0]
    assert handed[-1] == {"role": "user", "content": "second question"}
    assert "previous_response_id" not in captured

    # The compaction record chains to the prior id and stores the full input.
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == resp.json()["id"])
        .first()
    )
    assert record.previous_response_id == prior.response_id
    assert record.input_items == handed


def test_compact_merges_request_instructions(
    client: TestClient, session: Session, monkeypatch
) -> None:
    from app.api.v1.responses_compact import COMPACT_INSTRUCTIONS

    seed_instance(session, alias="cp-ins", machine_uid="cp-ins-m")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        litellm,
        "aresponses",
        make_fake_response(completed_final("s", {"input_tokens": 1}), captured),
    )

    resp = client.post(
        "/v1/responses/compact",
        json={"model": "cp-ins", "input": "hi", "instructions": "Be terse."},
    )
    assert resp.status_code == 200
    assert captured["instructions"].startswith("Be terse.")
    assert COMPACT_INSTRUCTIONS in captured["instructions"]


def test_compact_passes_prompt_cache_key(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="cp-pck", machine_uid="cp-pck-m")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        litellm,
        "aresponses",
        make_fake_response(completed_final("s", {"input_tokens": 1}), captured),
    )

    resp = client.post(
        "/v1/responses/compact",
        json={"model": "cp-pck", "input": "hi", "prompt_cache_key": "abc"},
    )
    assert resp.status_code == 200
    assert captured["prompt_cache_key"] == "abc"


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------
def test_compact_missing_model_400(client: TestClient) -> None:
    resp = client.post("/v1/responses/compact", json={"input": "x"})
    assert resp.status_code == 400


def test_compact_missing_input_and_previous_400(client: TestClient) -> None:
    resp = client.post("/v1/responses/compact", json={"model": "x"})
    assert resp.status_code == 400


def test_compact_unknown_alias_404(client: TestClient) -> None:
    resp = client.post("/v1/responses/compact", json={"model": "nope", "input": "x"})
    assert resp.status_code == 404


def test_compact_previous_not_found_404(client: TestClient, session: Session) -> None:
    seed_instance(session, alias="cp-pnf", machine_uid="cp-pnf-m")
    resp = client.post(
        "/v1/responses/compact",
        json={"model": "cp-pnf", "input": "x", "previous_response_id": "resp_missing"},
    )
    assert resp.status_code == 404


def test_compact_no_connected_provider_503(
    client: TestClient, session: Session
) -> None:
    machine = get_or_create_machine(session, uid="cp-np", host="127.0.0.1")
    agent = make_agent(session, machine, connected=False)
    definition = make_definition(
        session, alias="cp-np", backend_config={"model": {"file": "m.gguf"}}
    )
    make_instance(session, agent, definition, backend_status="stopped")
    resp = client.post("/v1/responses/compact", json={"model": "cp-np", "input": "x"})
    assert resp.status_code == 503


def test_compact_upstream_failure_502_persists_failed(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_instance(session, alias="cp-fail", machine_uid="cp-fail-m")

    async def failing_aresponses(*args, **kwargs):  # noqa: ARG001
        raise litellm.exceptions.APIError(
            status_code=500,
            message="boom",
            llm_provider="openai",
            model="cp-fail",
        )

    monkeypatch.setattr(litellm, "aresponses", failing_aresponses)

    resp = client.post("/v1/responses/compact", json={"model": "cp-fail", "input": "x"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_failed"

    # A failed record is persisted (id derived from the request; query latest).
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.status == "failed")
        .order_by(ResponseRecord.id.desc())
        .first()
    )
    assert record is not None
    assert record.error_code == "upstream_failed"
    assert record.parameters["api_format"] == "compaction"

    # Slot released even on failure.
    scheduler = client.app.state.scheduler
    assert scheduler.active_count("cp-fail") == 0


def test_compact_queue_timeout_504(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """QueueTimeout on the (non-stream) compact path -> 504 JSON."""
    from app.services.scheduler import QueueTimeout

    seed_instance(session, alias="cp-qt", machine_uid="cp-qt-m")
    scheduler = client.app.state.scheduler

    async def timeout_acquire(alias, request_id):  # noqa: ARG001
        raise QueueTimeout("timed out waiting for a 'cp-qt' slot")

    monkeypatch.setattr(scheduler, "acquire", timeout_acquire)
    resp = client.post("/v1/responses/compact", json={"model": "cp-qt", "input": "x"})
    assert resp.status_code == 504


def test_compact_non_dict_body_400(client: TestClient) -> None:
    resp = client.post("/v1/responses/compact", json=["not", "a", "dict"])
    assert resp.status_code == 400


def test_compact_always_stores_record(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Compaction records are always stored — a body ``store: false`` is
    ignored (the spec compact body has no ``store`` field)."""
    seed_instance(session, alias="cp-store", machine_uid="cp-store-m")
    monkeypatch.setattr(
        litellm,
        "aresponses",
        make_fake_response(completed_final("s", {"input_tokens": 1}), {}),
    )

    resp = client.post(
        "/v1/responses/compact",
        json={"model": "cp-store", "input": "hi", "store": False},
    )
    assert resp.status_code == 200
    record = (
        session.query(ResponseRecord)
        .filter(ResponseRecord.response_id == resp.json()["id"])
        .first()
    )
    assert record is not None
    assert record.store is True


def test_compact_does_not_forward_unknown_params(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Only the compact body schema is forwarded to litellm — sampling/other
    params present in the request must NOT leak into the litellm call."""
    seed_instance(session, alias="cp-sec", machine_uid="cp-sec-m")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        litellm,
        "aresponses",
        make_fake_response(completed_final("s", {"input_tokens": 1}), captured),
    )

    resp = client.post(
        "/v1/responses/compact",
        json={
            "model": "cp-sec",
            "input": "hi",
            "temperature": 0,
            "top_p": 0.5,
            "max_output_tokens": 10,
            "tools": [],
        },
    )
    assert resp.status_code == 200
    for key in ("temperature", "top_p", "max_output_tokens", "tools"):
        assert key not in captured
