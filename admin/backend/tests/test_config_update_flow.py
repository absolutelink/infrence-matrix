"""Phase 9 config-update flow over a real provider WebSocket.

The test plays the provider side with ``client.websocket_connect``: after
registering + connecting, a definition PATCH must deliver a
``provider.config.update`` frame with the new backend_config +
fingerprint; the ack sent back drives the DB updates (fingerprint on
ok, per-instance error on failure, fingerprint untouched on failure).
Also covers the stale-fingerprint reconnect self-heal and the sweep
heal helper.
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
from app.models import Machine, ProviderDefinition, ProviderInstance
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
    machine = Machine(uid="flow-mach", name="flow-machine", host="127.0.0.1")
    definition = ProviderDefinition(
        alias="flow-model",
        provider_type="mock",
        registration_token="flow-token",
        backend_config=backend_config,
    )
    session.add_all([machine, definition])
    session.commit()


def _register(client: TestClient) -> dict:
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": "flow-mach",
            "registration_token": "flow-token",
            "provider_type": "mock",
            "schema": {"type": "object"},
            "version": settings.VERSION,
            "port": 8081,
            "hardware": {"gpus": [], "total_vram_bytes": 1},
            "metrics_categories": [],
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


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


def _set_instance_state(instance_id: str, **changes) -> None:
    with Session(engine) as s:
        inst = s.get(ProviderInstance, uuid.UUID(instance_id))
        for key, value in changes.items():
            setattr(inst, key, value)
        s.add(inst)
        s.commit()


def test_patch_pushes_config_update_and_acks(
    client: TestClient, session: Session
) -> None:
    _make_stack(session, backend_config={"delta_count": 3})
    reg = _register(client)
    instance_id = reg["instance_id"]
    definition_id = reg["provider_definition"]["id"]
    old_fp = reg["provider_definition"]["config_fingerprint"]

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers={"Authorization": f"Bearer {reg['instance_secret']}"},
    ) as ws:
        hello = Frame.model_validate_json(ws.receive_text())
        assert hello.type == "provider.hello"

        t, res = _patch_in_thread(
            client, definition_id, {"backend_config": {"delta_count": 7}}
        )
        frame = Frame.model_validate_json(ws.receive_text())
        assert frame.type == "provider.config.update"
        new_fp = compute_config_fingerprint({"delta_count": 7})
        assert frame.payload["backend_config"] == {"delta_count": 7}
        assert frame.payload["config_fingerprint"] == new_fp
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
    instance_id = reg["instance_id"]
    definition_id = reg["provider_definition"]["id"]
    old_fp = reg["provider_definition"]["config_fingerprint"]

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers={"Authorization": f"Bearer {reg['instance_secret']}"},
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
    instance_id = reg["instance_id"]
    definition_id = reg["provider_definition"]["id"]

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers={"Authorization": f"Bearer {reg['instance_secret']}"},
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
    instance_id = reg["instance_id"]
    definition_id = reg["provider_definition"]["id"]
    stale_fp = compute_config_fingerprint({"delta_count": 3})

    # Admin PATCHes while the instance is NOT connected: no push happens.
    resp = client.patch(
        f"/admin/api/definitions/{definition_id}",
        json={"backend_config": {"delta_count": 4}},
    )
    assert resp.status_code == 200
    assert resp.json()["config_update_results"] == []
    new_fp = compute_config_fingerprint({"delta_count": 4})

    # The provider "reconnects" still carrying the stale fingerprint
    # (it missed the update while disconnected).
    _set_instance_state(
        instance_id, config_fingerprint=stale_fp, websocket_connected=False
    )

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers={"Authorization": f"Bearer {reg['instance_secret']}"},
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


def test_heal_helper_pushes_only_when_stale(monkeypatch) -> None:
    """Unit-level: the sweep/connect heal helper is cheap on a match and
    pushes on a mismatch (no live socket needed; send_command stubbed)."""
    import asyncio

    from app.services import config_update as cu

    machine = Machine(uid="heal-m", name="heal-machine")
    definition = ProviderDefinition(
        alias="heal-model",
        provider_type="mock",
        registration_token="heal-token",
        backend_config={"delta_count": 1},
    )
    with Session(engine) as s:
        s.add_all([machine, definition])
        s.commit()
        definition_id = definition.id
        inst = ProviderInstance(
            machine_id=machine.id,
            provider_definition_id=definition_id,
            websocket_connected=True,
            config_fingerprint=compute_config_fingerprint({"delta_count": 1}),
        )
        s.add(inst)
        s.commit()
        instance_id = str(inst.id)

    calls: list[dict] = []

    async def fake_send_command(inst_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append({"instance": inst_id, "type": type_, "payload": dict(payload)})
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

        # Disconnected instance -> never pushed.
        with Session(engine) as s:
            row = s.get(ProviderInstance, uuid.UUID(instance_id))
            row.config_fingerprint = "stale"
            row.websocket_connected = False
            s.add(row)
            s.commit()
        assert await cu.heal_stale_fingerprint(instance_id) is False
        assert len(calls) == 1

    asyncio.run(run())


def test_heal_targets_stale_instance_only(monkeypatch) -> None:
    """SF-3: a heal pushes to the single stale instance, never the whole
    definition fan-out (a current sibling must not be churned)."""
    import asyncio

    from app.services import config_update as cu

    machine = Machine(uid="fan-m", name="fan-machine")
    other = Machine(uid="fan-m2", name="fan-machine-2")
    definition = ProviderDefinition(
        alias="fan-model",
        provider_type="mock",
        registration_token="fan-token",
        backend_config={"delta_count": 2},
    )
    with Session(engine) as s:
        s.add_all([machine, other, definition])
        s.commit()
        current_fp = compute_config_fingerprint({"delta_count": 2})
        stale = ProviderInstance(
            machine_id=machine.id,
            provider_definition_id=definition.id,
            websocket_connected=True,
            config_fingerprint="stale-fp",
        )
        fresh = ProviderInstance(
            machine_id=other.id,
            provider_definition_id=definition.id,
            websocket_connected=True,
            config_fingerprint=current_fp,
        )
        s.add_all([stale, fresh])
        s.commit()
        stale_id, fresh_id = str(stale.id), str(fresh.id)

    calls: list[str] = []

    async def fake_send_command(inst_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append(inst_id)
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
        # Only the stale instance got the push; the current sibling did not.
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

    machine = Machine(uid="g-m", name="guard-machine")
    definition = ProviderDefinition(
        alias="guard-model",
        provider_type="mock",
        registration_token="guard-token",
        backend_config={"delta_count": 2},
    )
    with Session(engine) as s:
        s.add_all([machine, definition])
        s.commit()
        inst = ProviderInstance(
            machine_id=machine.id,
            provider_definition_id=definition.id,
            websocket_connected=True,
            config_fingerprint="stale-fp",
        )
        s.add(inst)
        s.commit()
        instance_id = str(inst.id)

    calls: list[str] = []
    release = asyncio.Event()
    first_started = asyncio.Event()

    async def slow_send_command(inst_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append(inst_id)
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
        # Concurrent heal for the same instance: skipped by the guard.
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

    machine = Machine(uid="x-m", name="x-machine")
    definition = ProviderDefinition(
        alias="x-model",
        provider_type="mock",
        registration_token="x-token",
        backend_config={"delta_count": 2},
    )
    with Session(engine) as s:
        s.add_all([machine, definition])
        s.commit()
        inst = ProviderInstance(
            machine_id=machine.id,
            provider_definition_id=definition.id,
            websocket_connected=True,
            config_fingerprint="stale-fp",
        )
        s.add(inst)
        s.commit()
        instance_id = str(inst.id)

    async def boom_send_command(inst_id, type_, payload, timeout=30.0):  # noqa: ARG001
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
    dies mid-push.

    Simulated by making the push abort this test process's route-side
    pooled connections (connection-killing monkeypatch, mirroring the
    server-side termination) while the provider acks ok."""
    import asyncio

    from sqlmodel import select

    from app.api.admin.providers import compute_config_fingerprint
    from app.core.db import get_session
    from app.main import app
    from app.services import config_update as cu
    from app.services.wire import Frame

    # _make_stack seeds the machine + definition (token flow-token);
    # _register then bootstraps the mock ProviderType (permissive schema)
    # and leaves a CONNECTED instance — exactly what the push fans out to.
    _make_stack(session, backend_config={"delta_count": 3})
    reg = _register(client)
    from sqlmodel import select as _select

    definition = session.exec(
        _select(ProviderDefinition).where(
            ProviderDefinition.id == uuid.UUID(reg["provider_definition"]["id"])
        )
    ).one()
    # No real WS is opened here (send_command is monkeypatched below);
    # mark the registered instance connected so the fan-out has a target.
    instance = session.exec(
        _select(ProviderInstance).where(
            ProviderInstance.provider_definition_id == definition.id
        )
    ).one()
    instance.websocket_connected = True
    session.add(instance)
    session.commit()

    slow_gate = asyncio.Event()

    async def slow_send(instance_id, kind, payload, timeout=None):  # noqa: ARG001
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
        # Fire the PATCH in a thread; release the gate shortly after the
        # push begins.
        import threading

        results_box: dict = {}

        def do_patch():
            results_box["resp"] = client.patch(
                f"/admin/api/definitions/{definition.id}",
                json={"backend_config": {"delta_count": 7}},
            )

        t = threading.Thread(target=do_patch)
        t.start()
        # Wait until the slow push is in flight, then let it proceed.
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
            # Debug visibility: why did the PATCH fail?
            print("PATCH-DEBUG", resp.status_code, resp.text[:400])

    finally:
        app.dependency_overrides.pop(get_session, None)

    assert resp is not None
    assert resp.status_code == 200, resp.text
    body = resp.json()
    expected = compute_config_fingerprint({"delta_count": 7})
    assert body["config_fingerprint"] == expected
    assert body["config_update_results"][0]["ok"] is True
    # The ack-echoed fingerprint was persisted despite the killed route
    # connection.
    session.expunge_all()
    inst = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.provider_definition_id == definition.id
        )
    ).first()
    assert inst is not None and inst.config_fingerprint == expected
