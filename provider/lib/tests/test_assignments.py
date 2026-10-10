"""provider_lib slice 5: ``agent.assignments.update`` registry reconcile.

The provider reconciles its ``BackendRegistry`` to the pushed assignment set:
add new handles (via a package ``make_handle`` factory), retire dropped ones
busy-safe (stop if idle, refuse if a slot is held), and ack
``{ok, added, removed, refused}``. A single-backend package (no factory) must
refuse an add it cannot host rather than crash.
"""

from typing import Any

from config_fakes import RecordingClient, TrackDriver

from provider_lib.assignments import install_assignment_handler
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState
from provider_lib.models import ModelSpec
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.wire import BackendStatusValue


class _ModelTrackingDriver(TrackDriver):
    """TrackDriver that records ``set_models`` calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_models_calls: list[list[ModelSpec]] = []

    def set_models(self, models: list[ModelSpec]) -> None:
        self.set_models_calls.append(list(models))


def _entry(iid: str, **over: Any) -> dict[str, Any]:
    entry = {
        "instance_id": iid,
        "provider_definition_id": f"def-{iid}",
        "alias": f"alias-{iid}",
        "backend_config": {"a": 1},
        "config_fingerprint": f"fp-{iid}",
        "port": 8081,
        "capacity": 1,
        "idle_timeout_seconds": 300,
        "vram_required_bytes": 0,
    }
    entry.update(over)
    return entry


def _registry_client() -> tuple[RecordingClient, BackendRegistry, list[BackendHandle]]:
    client = RecordingClient(ProviderSettings())
    registry = BackendRegistry()
    created: list[BackendHandle] = []

    def make_handle(entry: dict[str, Any]) -> BackendHandle:
        lifecycle = BackendLifecycle(
            TrackDriver(),
            capacity=entry.get("capacity", 1),
            instance_id=entry["instance_id"],
        )
        handle = BackendHandle(str(entry["instance_id"]), lifecycle, ConfigState())
        created.append(handle)
        return handle

    install_assignment_handler(client, registry, make_handle=make_handle)
    return client, registry, created


async def test_add_new_handles() -> None:
    client, registry, created = _registry_client()
    ack = await client.dispatch(
        "agent.assignments.update",
        {"assignments": [_entry("i1"), _entry("i2")], "max_running_backends": 0},
    )
    assert ack["ok"] is True
    assert sorted(ack["added"]) == ["i1", "i2"]
    assert ack["removed"] == []
    assert ack["refused"] == []
    assert "i1" in registry and "i2" in registry
    assert len(created) == 2


async def test_add_is_idempotent_for_hosted_ids() -> None:
    client, registry, _created = _registry_client()
    await client.dispatch("agent.assignments.update", {"assignments": [_entry("i1")]})
    ack = await client.dispatch(
        "agent.assignments.update", {"assignments": [_entry("i1")]}
    )
    assert ack["added"] == []  # already hosted -> not re-added
    assert len(registry) == 1


async def test_remove_idle_handle_stops_and_drops() -> None:
    client, registry, _created = _registry_client()
    await client.dispatch("agent.assignments.update", {"assignments": [_entry("i1")]})
    # Now drop it (empty assignment set) -> the idle backend is stopped+removed.
    ack = await client.dispatch("agent.assignments.update", {"assignments": []})
    assert ack["removed"] == ["i1"]
    assert "i1" not in registry


async def test_remove_busy_handle_is_refused_and_kept() -> None:
    client, registry, created = _registry_client()
    await client.dispatch("agent.assignments.update", {"assignments": [_entry("i1")]})
    handle = registry.get("i1")
    assert handle is not None
    await handle.lifecycle.start()
    await handle.lifecycle.acquire_slot()  # RUNNING -> IN_USE (busy)

    ack = await client.dispatch("agent.assignments.update", {"assignments": []})
    assert ack["removed"] == []
    assert {"instance_id": "i1", "reason": "backend_in_use"} in ack["refused"]
    assert "i1" in registry  # kept until it frees
    assert handle.lifecycle.backend_status == BackendStatusValue.IN_USE
    await handle.lifecycle.release_slot()
    del created


async def test_mixed_add_and_remove() -> None:
    client, registry, _created = _registry_client()
    await client.dispatch("agent.assignments.update", {"assignments": [_entry("old")]})
    ack = await client.dispatch(
        "agent.assignments.update", {"assignments": [_entry("new")]}
    )
    assert ack["added"] == ["new"]
    assert ack["removed"] == ["old"]
    assert "old" not in registry and "new" in registry


async def test_single_backend_refuses_unknown_add() -> None:
    """A package with no make_handle factory refuses an add it cannot host
    (never crashes) and still acks ok."""
    client = RecordingClient(ProviderSettings())
    registry = BackendRegistry()
    lifecycle = BackendLifecycle(TrackDriver(), instance_id="mine")
    registry.add(BackendHandle("mine", lifecycle, ConfigState()))
    install_assignment_handler(client, registry)  # no make_handle

    ack = await client.dispatch(
        "agent.assignments.update",
        {"assignments": [_entry("mine"), _entry("extra")]},
    )
    assert ack["ok"] is True
    assert ack["added"] == []  # 'mine' already hosted; 'extra' refused
    assert any(r["instance_id"] == "extra" for r in ack["refused"])
    assert "extra" not in registry


async def test_single_backend_placeholder_key_is_kept() -> None:
    """B1: a real single-backend agent builds its registry BEFORE
    apply_registration stamps lifecycle.instance_id, so the handle is keyed by
    the "" placeholder while its live lifecycle.instance_id is the real UUID.
    A same-type placement push naming the REAL id must NOT tear down the agent's
    own backend — the remove loop resolves identity via lifecycle.instance_id
    (exactly like resolve_target), not the registry key."""
    client = RecordingClient(ProviderSettings())
    registry = BackendRegistry()
    lifecycle = BackendLifecycle(TrackDriver(), instance_id="real-id")
    registry.add(
        BackendHandle("", lifecycle, ConfigState())
    )  # key predates registration
    install_assignment_handler(client, registry)  # no make_handle (single backend)
    await lifecycle.start()  # RUNNING + idle: the buggy path would stop_if_idle it

    ack = await client.dispatch(
        "agent.assignments.update",
        {"assignments": [_entry("real-id")]},
    )
    assert ack["ok"] is True
    assert ack["removed"] == []  # its own backend NOT torn down
    assert ack["added"] == []  # already hosted (resolve_target matches live id)
    assert registry.get("") is not None  # handle still present under its key
    assert lifecycle.backend_status == BackendStatusValue.RUNNING  # never stopped


async def test_invalid_payload_naks() -> None:
    client, _registry, _created = _registry_client()
    ack = await client.dispatch("agent.assignments.update", {"assignments": "nope"})
    assert ack["ok"] is False
    assert ack["detail"]["step"] == "validate"


async def test_add_sets_driver_modality() -> None:
    """Phase 18 slice 3: an assignment entry's modality reaches the created
    handle's driver AND its config_state; an absent modality defaults to "llm"."""
    client, registry, _created = _registry_client()
    await client.dispatch(
        "agent.assignments.update",
        {
            "assignments": [
                _entry("i1", modality="embedding"),
                _entry("i2"),  # no modality -> llm
            ]
        },
    )
    h1 = registry.get("i1")
    h2 = registry.get("i2")
    assert h1 is not None and h1.lifecycle.driver.modality == "embedding"
    assert h1.config_state is not None and h1.config_state.modality == "embedding"
    assert h2 is not None and h2.lifecycle.driver.modality == "llm"
    assert h2.config_state is not None and h2.config_state.modality == "llm"


async def test_max_running_callback_invoked() -> None:
    client = RecordingClient(ProviderSettings())
    registry = BackendRegistry()
    seen: list[int] = []
    install_assignment_handler(
        client, registry, on_max_running=lambda n: seen.append(n)
    )
    await client.dispatch(
        "agent.assignments.update", {"assignments": [], "max_running_backends": 1}
    )
    assert seen == [1]


# ---------------------------------------------------------------------------
# Phase 25 S1: served_models threading through the reconcile.
# ---------------------------------------------------------------------------


def _model_registry_client() -> tuple[
    RecordingClient, BackendRegistry, list[BackendHandle]
]:
    client = RecordingClient(ProviderSettings())
    registry = BackendRegistry()
    created: list[BackendHandle] = []

    def make_handle(entry: dict[str, Any]) -> BackendHandle:
        driver = _ModelTrackingDriver()
        lifecycle = BackendLifecycle(
            driver,
            capacity=entry.get("capacity", 1),
            instance_id=entry["instance_id"],
        )
        handle = BackendHandle(str(entry["instance_id"]), lifecycle, ConfigState())
        created.append(handle)
        return handle

    install_assignment_handler(client, registry, make_handle=make_handle)
    return client, registry, created


async def test_add_with_served_models_reaches_handle_and_driver() -> None:
    client, registry, _created = _model_registry_client()
    entry = _entry(
        "i1",
        served_models=[
            {"name": "a", "modality": "tts"},
            {"name": "b", "modality": "asr"},
        ],
    )
    ack = await client.dispatch("agent.assignments.update", {"assignments": [entry]})
    assert ack["added"] == ["i1"]
    h = registry.get("i1")
    assert h is not None
    # The handle carries both served models (not just the legacy alias).
    assert [m.name for m in h.models] == ["a", "b"]
    # Both names route to the handle; the legacy alias does not (it was replaced).
    assert registry.resolve_by_model("a") is h
    assert registry.resolve_by_model("b") is h
    # The driver hook fired with the parsed list.
    assert h.lifecycle.driver.set_models_calls == [
        [ModelSpec(name="a", modality="tts"), ModelSpec(name="b", modality="asr")]
    ]


async def test_live_add_of_served_name_refreshes_existing_backend() -> None:
    """A changed served_models list on an ALREADY-HOSTED backend re-applies in
    place (no restart): the new name reaches the handle + driver while the
    backend keeps serving."""
    client, registry, _created = _model_registry_client()
    await client.dispatch(
        "agent.assignments.update",
        {"assignments": [_entry("i1", served_models=[{"name": "a"}])]},
    )
    h = registry.get("i1")
    assert h is not None
    await h.lifecycle.start()  # bring it up; a refresh must NOT stop it
    assert h.lifecycle.in_flight == 0

    # Push the same backend with an extra served name "b".
    ack = await client.dispatch(
        "agent.assignments.update",
        {"assignments": [_entry("i1", served_models=[{"name": "a"}, {"name": "b"}])]},
    )
    assert ack["added"] == []  # already hosted, not re-added
    assert ack["removed"] == []
    assert h.lifecycle.backend_status == BackendStatusValue.RUNNING  # never stopped
    assert [m.name for m in h.models] == ["a", "b"]
    assert registry.resolve_by_model("b") is h
    # The driver got a second set_models call with the expanded list.
    assert len(h.lifecycle.driver.set_models_calls) == 2
    assert [m.name for m in h.lifecycle.driver.set_models_calls[-1]] == ["a", "b"]


async def test_unchanged_served_models_does_not_reapply() -> None:
    client, registry, _created = _model_registry_client()
    entry = _entry("i1", served_models=[{"name": "a"}])
    await client.dispatch("agent.assignments.update", {"assignments": [entry]})
    h = registry.get("i1")
    assert h is not None
    before = len(h.lifecycle.driver.set_models_calls)
    # Re-push the identical entry: no change -> no extra set_models call.
    await client.dispatch("agent.assignments.update", {"assignments": [entry]})
    assert len(h.lifecycle.driver.set_models_calls) == before


async def test_live_disable_of_served_name_is_unroutable() -> None:
    """F1 via the reconcile: toggling the sole served name off must make it
    unroutable (no alias resurrection), not leave it served."""
    client, registry, _created = _model_registry_client()
    await client.dispatch(
        "agent.assignments.update",
        {"assignments": [_entry("i1", served_models=[{"name": "x"}])]},
    )
    h = registry.get("i1")
    assert h is not None
    assert registry.resolve_by_model("x") is h

    ack = await client.dispatch(
        "agent.assignments.update",
        {
            "assignments": [
                _entry("i1", served_models=[{"name": "x", "enabled": False}])
            ]
        },
    )
    assert ack["added"] == [] and ack["removed"] == []  # refreshed in place
    assert registry.resolve_by_model("x") is None
    assert h.served_names == []
