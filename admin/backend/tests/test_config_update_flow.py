"""Phase 9 config-update flow over a real provider agent WebSocket (Phase 16).

The test plays the provider side with ``client.websocket_connect``: after
registering + connecting, a definition PATCH must deliver a
``provider.config.update`` frame (addressed to the agent, carrying the target
``instance_id``) with the new backend_config + fingerprint; the ack sent back
drives the DB updates (fingerprint on ok, per-instance error on failure,
fingerprint untouched on failure). Also covers the stale-fingerprint reconnect
self-heal and the sweep heal helper.
"""

import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.admin.providers import compute_config_fingerprint
from app.core.config import settings
from app.core.db import engine
from app.models import Machine, ProviderAgent, ProviderDefinition, ProviderInstance
from app.services.wire import Frame


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition not met in time")


def _make_stack(session: Session, *, backend_config: dict) -> None:
    machine = Machine(
        uid="flow-mach",
        name="flow-machine",
        host="127.0.0.1",
        registration_secret="flow-secret",
    )
    definition = ProviderDefinition(
        alias="flow-model",
        provider_type="mock",
        backend_config=backend_config,
    )
    session.add_all([machine, definition])
    session.commit()


def _register(client: TestClient) -> dict:
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": "flow-mach",
            "machine_secret": "flow-secret",
            "agent_id": "flow-agent",
            "provider_type": "mock",
            "schema": {"type": "object"},
            "version": settings.VERSION,
            "base_port": 8081,
            "hardware": {"gpus": [], "total_vram_bytes": 1},
            "metrics_categories": [],
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _backend(reg: dict) -> dict:
    return reg["backends"][0]


def _patch_in_thread(client: TestClient, definition_id: str, body: dict):
    """Kick the PATCH off-thread so the WS frames can be exchanged while
    the admin awaits the provider ack."""
    result: dict = {}

    def run() -> None:
        result["resp"] = client.patch(
            f"/admin/api/definitions/{definition_id}", json=body
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, result


def _set_agent_connected(agent_id: str, connected: bool) -> None:
    with Session(engine) as s:
        agent = s.get(ProviderAgent, uuid.UUID(agent_id))
        agent.websocket_connected = connected
        s.add(agent)
        s.commit()


def test_patch_pushes_config_update_and_acks(
    client: TestClient, session: Session
) -> None:
    _make_stack(session, backend_config={"delta_count": 3})
    reg = _register(client)
    instance_id = _backend(reg)["instance_id"]
    definition_id = _backend(reg)["definition"]["id"]
    old_fp = _backend(reg)["definition"]["config_fingerprint"]

    with client.websocket_connect(
        f"/provider/ws?agent_id={reg['agent_id']}",
        headers={"Authorization": f"Bearer {reg['agent_secret']}"},
    ) as ws:
        hello = Frame.model_validate_json(ws.receive_text())
        assert hello.type == "provider.hello"

        t, res = _patch_in_thread(
            client, definition_id, {"backend_config": {"delta_count": 7}}
        )
        frame = Frame.model_validate_json(ws.receive_text())
        assert frame.type == "provider.config.update"
        new_fp = compute_config_fingerprint({"delta_count": 7})
        assert frame.payload["instance_id"] == instance_id
        assert frame.payload["backend_config"] == {"delta_count": 7}
        assert frame.payload["config_fingerprint"] == new_fp
        assert frame.payload["modality"] == "llm"  # Phase 18 slice 3
        assert frame.payload["capacity"] == 1
        assert frame.payload["idle_timeout_seconds"] == 300
        assert frame.epoch == hello.epoch

        ws.send_text(
            Frame(
                type="ack",
                id=str(uuid.uuid4()),
                reply_to=frame.id,
                epoch=hello.epoch,
                payload={
                    "ok": True,
                    "detail": {
                        "config_fingerprint": new_fp,
                        "capacity": 1,
                        "model_metadata": [{"id": "flow-model", "object": "model"}],
                    },
                },
            ).to_json()
        )
        t.join(timeout=10)

    assert res["resp"].status_code == 200
    results = res["resp"].json()["config_update_results"]
    assert len(results) == 1
    assert results[0]["instance_id"] == instance_id
    assert results[0]["ok"] is True
    assert results[0]["config_fingerprint"] == new_fp

    session.expunge_all()
    inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    assert inst.config_fingerprint == new_fp
    definition = session.get(ProviderDefinition, uuid.UUID(definition_id))
    assert definition.model_metadata == {
        "models": [{"id": "flow-model", "object": "model"}]
    }
    assert old_fp != new_fp


def test_patch_reports_provider_failure_without_touching_fingerprint(
    client: TestClient, session: Session
) -> None:
    _make_stack(session, backend_config={"delta_count": 3})
    reg = _register(client)
    instance_id = _backend(reg)["instance_id"]
    definition_id = _backend(reg)["definition"]["id"]
    old_fp = _backend(reg)["definition"]["config_fingerprint"]

    with client.websocket_connect(
        f"/provider/ws?agent_id={reg['agent_id']}",
        headers={"Authorization": f"Bearer {reg['agent_secret']}"},
    ) as ws:
        hello = Frame.model_validate_json(ws.receive_text())

        t, res = _patch_in_thread(
            client, definition_id, {"backend_config": {"delta_count": 99}}
        )
        frame = Frame.model_validate_json(ws.receive_text())
        assert frame.type == "provider.config.update"

        ws.send_text(
            Frame(
                type="ack",
                id=str(uuid.uuid4()),
                reply_to=frame.id,
                epoch=hello.epoch,
                payload={
                    "ok": False,
                    "error": "start failed: boom",
                    "detail": {"step": "start"},
                },
            ).to_json()
        )
        t.join(timeout=10)

    assert res["resp"].status_code == 200  # admin row saved regardless
    results = res["resp"].json()["config_update_results"]
    assert results[0]["ok"] is False
    assert results[0]["error"] == "start failed: boom"
    assert results[0]["step"] == "start"

    session.expunge_all()
    inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    assert inst.config_fingerprint == old_fp  # not updated on failure


def test_drain_refused_is_retried_then_reported(
    client: TestClient, session: Session, monkeypatch
) -> None:
    from app.services import config_update as cu

    monkeypatch.setattr(cu, "CONFIG_UPDATE_RETRIES", 2)
    monkeypatch.setattr(cu, "CONFIG_UPDATE_RETRY_DELAY", 0.01)

    _make_stack(session, backend_config={"delta_count": 3})
    reg = _register(client)
    instance_id = _backend(reg)["instance_id"]
    definition_id = _backend(reg)["definition"]["id"]

    with client.websocket_connect(
        f"/provider/ws?agent_id={reg['agent_id']}",
        headers={"Authorization": f"Bearer {reg['agent_secret']}"},
    ) as ws:
        hello = Frame.model_validate_json(ws.receive_text())

        t, res = _patch_in_thread(
            client, definition_id, {"backend_config": {"delta_count": 5}}
        )
        # First attempt: drain refused.
        frame1 = Frame.model_validate_json(ws.receive_text())
        assert frame1.type == "provider.config.update"
        ws.send_text(
            Frame(
                type="ack",
                reply_to=frame1.id,
                epoch=hello.epoch,
                payload={
                    "ok": False,
                    "error": "backend_in_use",
                    "detail": {"step": "drain", "retry_after": 10},
                },
            ).to_json()
        )
        # Second attempt (after the retry delay): accepted.
        frame2 = Frame.model_validate_json(ws.receive_text())
        assert frame2.type == "provider.config.update"
        new_fp = frame2.payload["config_fingerprint"]
        ws.send_text(
            Frame(
                type="ack",
                reply_to=frame2.id,
                epoch=hello.epoch,
                payload={"ok": True, "detail": {"config_fingerprint": new_fp}},
            ).to_json()
        )
        t.join(timeout=10)

    assert res["resp"].status_code == 200
    results = res["resp"].json()["config_update_results"]
    assert results[0]["ok"] is True
    session.expunge_all()
    inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    assert inst.config_fingerprint == new_fp


def test_reconnect_with_stale_fingerprint_self_heals(
    client: TestClient, session: Session
) -> None:
    _make_stack(session, backend_config={"delta_count": 3})
    reg = _register(client)
    instance_id = _backend(reg)["instance_id"]
    definition_id = _backend(reg)["definition"]["id"]
    stale_fp = compute_config_fingerprint({"delta_count": 3})

    # Admin PATCHes while the agent is NOT connected: no push happens.
    _set_agent_connected(reg["agent_id"], False)
    resp = client.patch(
        f"/admin/api/definitions/{definition_id}",
        json={"backend_config": {"delta_count": 4}},
    )
    assert resp.status_code == 200
    assert resp.json()["config_update_results"] == []
    new_fp = compute_config_fingerprint({"delta_count": 4})

    # The provider "reconnects" still carrying the stale fingerprint
    # (it missed the update while disconnected).
    with Session(engine) as s:
        inst = s.get(ProviderInstance, uuid.UUID(instance_id))
        inst.config_fingerprint = stale_fp
        s.add(inst)
        s.commit()

    with client.websocket_connect(
        f"/provider/ws?agent_id={reg['agent_id']}",
        headers={"Authorization": f"Bearer {reg['agent_secret']}"},
    ) as ws:
        hello = Frame.model_validate_json(ws.receive_text())
        # The connect-path background heal pushes the update unprompted.
        frame = Frame.model_validate_json(ws.receive_text())
        assert frame.type == "provider.config.update"
        assert frame.payload["config_fingerprint"] == new_fp
        ws.send_text(
            Frame(
                type="ack",
                reply_to=frame.id,
                epoch=hello.epoch,
                payload={"ok": True, "detail": {"config_fingerprint": new_fp}},
            ).to_json()
        )

        def healed() -> bool:
            session.expire_all()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return row.config_fingerprint == new_fp

        _wait_for(healed)


def _seed_agent_instance(session: Session, *, uid: str, config: dict, connected: bool):
    """Create machine + agent + definition + one backend; returns
    (agent_id, instance_id, definition_id)."""
    machine = Machine(uid=uid, name=f"name-{uid}", registration_secret="s")
    definition = ProviderDefinition(
        alias=f"model-{uid}", provider_type="mock", backend_config=config
    )
    session.add_all([machine, definition])
    session.commit()
    agent = ProviderAgent(
        machine_id=machine.id,
        provider_type="mock",
        agent_id=f"agent-{uid}",
        websocket_connected=connected,
    )
    session.add(agent)
    session.commit()
    inst = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=definition.id,
        config_fingerprint=compute_config_fingerprint(config),
    )
    session.add(inst)
    session.commit()
    return str(agent.id), str(inst.id), definition.id


def test_heal_helper_pushes_only_when_stale(monkeypatch) -> None:
    """Unit-level: the sweep/connect heal helper is cheap on a match and
    pushes on a mismatch (no live socket needed; send_command stubbed)."""
    import asyncio

    from app.services import config_update as cu

    with Session(engine) as s:
        _agent_id, instance_id, _def_id = _seed_agent_instance(
            s, uid="heal-m", config={"delta_count": 1}, connected=True
        )

    calls: list[dict] = []

    async def fake_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append(
            {
                "instance": payload.get("instance_id"),
                "type": type_,
                "payload": dict(payload),
            }
        )
        return Frame(
            type="ack",
            payload={
                "ok": True,
                "detail": {"config_fingerprint": payload["config_fingerprint"]},
            },
        )

    monkeypatch.setattr(cu.manager, "send_command", fake_send_command)

    async def run() -> None:
        # Fresh fingerprint -> heal is a no-op (no push).
        assert await cu.heal_stale_fingerprint(instance_id) is False
        assert calls == []

        # Stale fingerprint -> heal pushes once and updates the row.
        with Session(engine) as s:
            row = s.get(ProviderInstance, uuid.UUID(instance_id))
            row.config_fingerprint = "stale"
            s.add(row)
            s.commit()
        assert await cu.heal_stale_fingerprint(instance_id) is True
        assert len(calls) == 1
        assert calls[0]["type"] == "provider.config.update"
        with Session(engine) as s:
            row = s.get(ProviderInstance, uuid.UUID(instance_id))
            assert row.config_fingerprint != "stale"

    asyncio.run(run())


def test_heal_targets_stale_instance_only(monkeypatch) -> None:
    """SF-3: a heal pushes to the single stale backend, never the whole
    definition fan-out (a current sibling must not be churned)."""
    import asyncio

    from app.services import config_update as cu

    with Session(engine) as s:
        _a1, stale_id, _d1 = _seed_agent_instance(
            s, uid="fan-m", config={"delta_count": 2}, connected=True
        )
    with Session(engine) as s:
        # a second agent/backend on the SAME definition, current fingerprint
        definition = (
            s.query(ProviderDefinition)
            .filter(ProviderDefinition.alias == "model-fan-m")
            .first()
        )
        machine = s.query(Machine).filter(Machine.uid == "fan-m").first()
        agent2 = ProviderAgent(
            machine_id=machine.id,
            provider_type="mock",
            agent_id="agent-fan-m-2",
            websocket_connected=True,
        )
        s.add(agent2)
        s.commit()
        fresh = ProviderInstance(
            agent_id=agent2.id,
            provider_definition_id=definition.id,
            config_fingerprint=compute_config_fingerprint({"delta_count": 2}),
        )
        s.add(fresh)
        s.commit()
        fresh_id = str(fresh.id)
        # make the first backend stale
        row = s.get(ProviderInstance, uuid.UUID(stale_id))
        row.config_fingerprint = "stale-fp"
        s.add(row)
        s.commit()

    calls: list[str] = []

    async def fake_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append(payload.get("instance_id"))
        return Frame(
            type="ack",
            payload={
                "ok": True,
                "detail": {"config_fingerprint": payload["config_fingerprint"]},
            },
        )

    monkeypatch.setattr(cu.manager, "send_command", fake_send_command)

    async def run() -> None:
        assert await cu.heal_stale_fingerprint(stale_id) is True
        # Only the stale backend got the push; the current sibling did not.
        assert calls == [stale_id]
        # The now-current sibling heals to a no-op.
        assert await cu.heal_stale_fingerprint(fresh_id) is False
        assert calls == [stale_id]

    asyncio.run(run())


def test_heal_in_flight_guard_skips_duplicates(monkeypatch) -> None:
    """SF-3: a second heal while the first push is in flight is skipped
    (no duplicate command queued); the guard clears on completion."""
    import asyncio

    from app.services import config_update as cu

    with Session(engine) as s:
        _agent_id, instance_id, _def_id = _seed_agent_instance(
            s, uid="g-m", config={"delta_count": 2}, connected=True
        )
        row = s.get(ProviderInstance, uuid.UUID(instance_id))
        row.config_fingerprint = "stale-fp"
        s.add(row)
        s.commit()

    calls: list[str] = []
    release = asyncio.Event()
    first_started = asyncio.Event()

    async def slow_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append(payload.get("instance_id"))
        first_started.set()
        await release.wait()
        return Frame(
            type="ack",
            payload={
                "ok": True,
                "detail": {"config_fingerprint": payload["config_fingerprint"]},
            },
        )

    monkeypatch.setattr(cu.manager, "send_command", slow_send_command)

    async def run() -> None:
        first = asyncio.create_task(cu.heal_stale_fingerprint(instance_id))
        await first_started.wait()
        # Concurrent heal for the same backend: skipped by the guard.
        assert await cu.heal_stale_fingerprint(instance_id) is False
        assert calls == [instance_id]  # exactly one push delivered
        release.set()
        assert await first is True
        assert instance_id not in cu._push_in_flight  # guard cleared
        # After completion a new heal works again (row is fresh now, so
        # force it stale once more).
        with Session(engine) as s:
            row = s.get(ProviderInstance, uuid.UUID(instance_id))
            row.config_fingerprint = "stale-again"
            s.add(row)
            s.commit()
        assert await cu.heal_stale_fingerprint(instance_id) is True
        assert calls == [instance_id, instance_id]

    asyncio.run(run())


def test_heal_guard_cleared_on_exception(monkeypatch) -> None:
    """SF-3: the in-flight guard is released even when the push raises."""
    import asyncio

    from app.services import config_update as cu

    with Session(engine) as s:
        _agent_id, instance_id, _def_id = _seed_agent_instance(
            s, uid="x-m", config={"delta_count": 2}, connected=True
        )
        row = s.get(ProviderInstance, uuid.UUID(instance_id))
        row.config_fingerprint = "stale-fp"
        s.add(row)
        s.commit()

    async def boom_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        raise RuntimeError("socket died mid-push")

    monkeypatch.setattr(cu.manager, "send_command", boom_send_command)

    async def run() -> None:
        # Push attempted (reported failed inside the result), guard released.
        assert await cu.heal_stale_fingerprint(instance_id) is True
        assert instance_id not in cu._push_in_flight

    asyncio.run(run())


def test_patch_push_slower_than_idle_tx_timeout_still_200(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """N-8 (prod incident): a config.push slower than Postgres's
    idle_in_transaction_session_timeout (15s) used to 500 the PATCH —
    the route session sat idle-in-transaction across the awaited push and
    Postgres killed the connection. The route now ends its transaction
    before awaiting (push_config_update_by_id) and re-reads after, so a
    slow push yields a correct 200 even when the underlying connection
    dies mid-push."""
    import asyncio

    from sqlmodel import select

    from app.api.admin.providers import compute_config_fingerprint
    from app.core.db import get_session
    from app.main import app
    from app.services import config_update as cu
    from app.services.wire import Frame

    _make_stack(session, backend_config={"delta_count": 3})
    reg = _register(client)
    definition = session.exec(
        select(ProviderDefinition).where(
            ProviderDefinition.id == uuid.UUID(_backend(reg)["definition"]["id"])
        )
    ).one()
    # The registered agent is connected (WS connect happened in _register?
    # No — registration marks registering; connect sets it). Mark the agent
    # connected so the fan-out has a target.
    agent = session.exec(
        select(ProviderAgent).where(ProviderAgent.id == uuid.UUID(reg["agent_id"]))
    ).one()
    agent.websocket_connected = True
    session.add(agent)
    session.commit()

    slow_gate = asyncio.Event()

    async def slow_send(agent_id, kind, payload, timeout=None):  # noqa: ARG001
        # Simulate a >15s provider apply: hold the push while the
        # route's open transaction would idle out.
        await asyncio.wait_for(slow_gate.wait(), timeout=5)
        return Frame(
            type="ack",
            payload={
                "ok": True,
                "detail": {"config_fingerprint": payload["config_fingerprint"]},
            },
        )

    monkeypatch.setattr(cu.manager, "send_command", slow_send)

    real_get_session = get_session

    def killing_get_session():
        """Yield a session, then KILL its pooled connections on close —
        exactly the server-side effect of
        idle_in_transaction_session_timeout while the push runs."""
        gen = real_get_session()
        s = next(gen)
        try:
            yield s
        finally:
            engine_obj = s.get_bind()
            s.close()
            engine_obj.dispose()

    app.dependency_overrides[get_session] = killing_get_session
    try:
        import threading

        results_box: dict = {}

        def do_patch():
            results_box["resp"] = client.patch(
                f"/admin/api/definitions/{definition.id}",
                json={"backend_config": {"delta_count": 7}},
            )

        t = threading.Thread(target=do_patch)
        t.start()
        in_flight = False
        for _ in range(100):
            if cu._push_in_flight:
                in_flight = True
                break
            time.sleep(0.05)
        assert in_flight, "push never started"
        slow_gate.set()
        t.join(timeout=30)
        resp = results_box.get("resp")
        if resp is not None and resp.status_code != 200:
            print("PATCH-DEBUG", resp.status_code, resp.text[:400])  # noqa: T201

    finally:
        app.dependency_overrides.pop(get_session, None)

    assert resp is not None
    assert resp.status_code == 200, resp.text
    body = resp.json()
    expected = compute_config_fingerprint({"delta_count": 7})
    assert body["config_fingerprint"] == expected
    assert body["config_update_results"][0]["ok"] is True
    session.expunge_all()
    inst = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.provider_definition_id == definition.id
        )
    ).first()
    assert inst is not None and inst.config_fingerprint == expected


def test_build_payload_carries_modality() -> None:
    """Phase 18 slice 3: the provider.config.update payload mirrors the
    definition's modality so the provider boots the right endpoint kind."""
    from app.services.config_update import _build_payload

    emb = ProviderDefinition(
        alias="emb-cu",
        provider_type="llama-cpp",
        backend_config={"artifacts": {}},
        modality="embedding",
    )
    payload = _build_payload(emb, "fp-x")
    assert payload["modality"] == "embedding"
    assert payload["config_fingerprint"] == "fp-x"

    llm = ProviderDefinition(alias="llm-cu", provider_type="mock", backend_config={})
    assert _build_payload(llm, "fp-y")["modality"] == "llm"
