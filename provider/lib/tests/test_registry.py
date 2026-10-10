"""Port model overhaul: ``BackendRegistry`` alias index + ``resolve_by_model``.

The agent's single ``/v1`` surface routes an inbound request to the backend
whose ``alias`` equals the request's ``model`` field. The registry maintains an
``alias -> handle`` index kept in sync by ``add``/``remove``.
"""

from __future__ import annotations

import logging

from config_fakes import TrackDriver

from provider_lib.backend import BackendLifecycle
from provider_lib.models import ModelSpec
from provider_lib.registry import BackendHandle, BackendRegistry


def _handle(iid: str, alias: str | None) -> BackendHandle:
    return BackendHandle(
        iid,
        BackendLifecycle(TrackDriver(), instance_id=iid),
        config_state=None,
        alias=alias,
    )


def _multi_handle(iid: str, *specs: ModelSpec) -> BackendHandle:
    return BackendHandle(
        iid,
        BackendLifecycle(TrackDriver(), instance_id=iid),
        config_state=None,
        alias=specs[0].name if specs else None,
        models=list(specs),
    )


def test_resolve_by_model_hit_miss_none() -> None:
    reg = BackendRegistry()
    reg.add(_handle("i1", "alpha"))
    reg.add(_handle("i2", "beta"))

    assert reg.resolve_by_model("alpha") is reg.get("i1")
    assert reg.resolve_by_model("beta") is reg.get("i2")
    # Unknown alias -> None.
    assert reg.resolve_by_model("gamma") is None
    # None / empty model -> None (never a fallback to the sole handle).
    assert reg.resolve_by_model(None) is None
    assert reg.resolve_by_model("") is None
    # L1: a non-string model (list/int) -> None, not a TypeError from dict.get.
    assert reg.resolve_by_model(["x"]) is None  # type: ignore[arg-type]
    assert reg.resolve_by_model(123) is None  # type: ignore[arg-type]


def test_alias_index_maintained_on_add_remove() -> None:
    reg = BackendRegistry()
    h1 = _handle("i1", "alpha")
    reg.add(h1)
    assert reg.resolve_by_model("alpha") is h1

    # Removing the handle drops its alias from the index.
    reg.remove("i1")
    assert reg.resolve_by_model("alpha") is None


def test_handle_without_alias_is_not_indexed() -> None:
    reg = BackendRegistry()
    reg.add(_handle("i1", None))
    # A single unnamed handle is still resolvable by instance id, but there is
    # no model to route to.
    assert reg.resolve_by_model("anything") is None
    assert reg.resolve("i1") is not None


def test_replaced_alias_points_at_newest_handle() -> None:
    reg = BackendRegistry()
    old = _handle("i1", "alpha")
    reg.add(old)
    new = _handle("i2", "alpha")
    reg.add(new)
    assert reg.resolve_by_model("alpha") is new
    # Removing the OLD handle must not clear the alias the NEW one owns.
    reg.remove("i1")
    assert reg.resolve_by_model("alpha") is new


# ---------------------------------------------------------------------------
# Phase 25 S1: multi-model handles.
# ---------------------------------------------------------------------------


def test_handle_synthesizes_single_model_from_alias() -> None:
    # A legacy construction site (alias only, no models) carries one spec.
    h = _handle("i1", "alpha")
    assert h.models == [ModelSpec(name="alpha")]
    assert h.served_names == ["alpha"]


def test_nameless_placeholder_has_no_models() -> None:
    h = _handle("i1", None)
    assert h.models == []
    assert h.served_names == []


def test_resolve_by_model_matches_any_enabled_name() -> None:
    reg = BackendRegistry()
    h = _multi_handle(
        "i1",
        ModelSpec(name="a", modality="tts"),
        ModelSpec(name="b", modality="asr"),
        ModelSpec(name="c", enabled=False),  # disabled -> unroutable
    )
    reg.add(h)
    assert reg.resolve_by_model("a") is h
    assert reg.resolve_by_model("b") is h
    # A disabled served name resolves to None (caller 404s).
    assert reg.resolve_by_model("c") is None
    # An unknown name resolves to None.
    assert reg.resolve_by_model("z") is None


def test_set_models_refreshes_index_and_calls_driver() -> None:
    reg = BackendRegistry()
    h = _handle("i1", "alpha")  # single legacy model
    reg.add(h)
    assert reg.resolve_by_model("alpha") is h

    calls: list[list[ModelSpec]] = []
    h.lifecycle.driver.set_models = lambda models: calls.append(list(models))  # type: ignore[method-assign]

    new_models = [ModelSpec(name="x"), ModelSpec(name="y")]
    reg.set_models(h, new_models)

    # The handle now carries the new list and re-derives alias (first enabled).
    assert h.models == new_models
    assert h.alias == "x"
    # Old name dropped, both new names route to the handle.
    assert reg.resolve_by_model("alpha") is None
    assert reg.resolve_by_model("x") is h
    assert reg.resolve_by_model("y") is h
    # The driver hook fired exactly once with the new list.
    assert calls == [new_models]


def test_set_models_disabled_name_not_indexed() -> None:
    reg = BackendRegistry()
    h = _handle("i1", "alpha")
    reg.add(h)
    reg.set_models(h, [ModelSpec(name="on"), ModelSpec(name="off", enabled=False)])
    assert reg.resolve_by_model("on") is h
    assert reg.resolve_by_model("off") is None
    assert reg.resolve_by_model("alpha") is None


def test_remove_drops_all_enabled_names() -> None:
    reg = BackendRegistry()
    h = _multi_handle("i1", ModelSpec(name="a"), ModelSpec(name="b"))
    reg.add(h)
    assert reg.resolve_by_model("a") is h and reg.resolve_by_model("b") is h
    reg.remove("i1")
    assert reg.resolve_by_model("a") is None
    assert reg.resolve_by_model("b") is None


# --- F1: all-disabled toggle must NOT resurrect the alias ---------------------


def test_set_models_all_disabled_not_routable() -> None:
    """Toggling the sole enabled name off via the live set_models path must make
    it unroutable — the stale alias must NOT be re-indexed (spec §3/§5: disabled
    -> 404)."""
    reg = BackendRegistry()
    h = _multi_handle("i1", ModelSpec(name="x"))
    reg.add(h)
    assert reg.resolve_by_model("x") is h

    reg.set_models(h, [ModelSpec(name="x", enabled=False)])

    assert reg.resolve_by_model("x") is None
    # The alias is left as a display value but is NOT routable via the index.
    assert h.alias == "x"
    assert reg.resolve_by_model(h.alias) is None
    # And the handle reports no served names at all.
    assert h.served_names == []


def test_all_disabled_survives_construction_and_add() -> None:
    """A handle built with an all-disabled served list is unroutable from the
    start (no alias resurrection)."""
    reg = BackendRegistry()
    h = BackendHandle(
        "i1",
        BackendLifecycle(TrackDriver(), instance_id="i1"),
        None,
        alias="x",
        models=[ModelSpec(name="x", enabled=False)],
    )
    reg.add(h)
    assert reg.resolve_by_model("x") is None


# --- F2: cross-handle collision warns, last-write-wins -----------------------


def test_add_collision_warns_and_last_write_wins(caplog) -> None:
    reg = BackendRegistry()
    a = _handle("i1", "dup")
    reg.add(a)
    b = _handle("i2", "dup")
    with caplog.at_level(logging.WARNING, logger="provider.registry"):
        reg.add(b)
    assert reg.resolve_by_model("dup") is b  # newest handle wins
    assert any("dup" in rec.message for rec in caplog.records)


def test_set_models_collision_warns(caplog) -> None:
    reg = BackendRegistry()
    a = _handle("i1", "dup")
    b = _handle("i2", "other")
    reg.add(a)
    reg.add(b)
    with caplog.at_level(logging.WARNING, logger="provider.registry"):
        reg.set_models(b, [ModelSpec(name="dup")])
    assert reg.resolve_by_model("dup") is b
    assert any("dup" in rec.message for rec in caplog.records)


def test_same_handle_reindex_does_not_warn() -> None:
    """Re-applying the same handle's own name is not a collision (no warning)."""
    reg = BackendRegistry()
    h = _handle("i1", "x")
    reg.add(h)
    # No caplog assertion needed beyond: re-adding the same handle's name must
    # not emit a warning.
    reg.set_models(h, [ModelSpec(name="x"), ModelSpec(name="y")])
    assert reg.resolve_by_model("x") is h


# --- (c) name MOVE: remove guard protects the other handle -------------------


def test_set_models_move_preserves_other_handles_won_name() -> None:
    """B owns 'x', then A wins 'x' via a collision. When B moves to 'y', the
    points-at-handle guard in set_models must NOT evict A's 'x'."""
    reg = BackendRegistry()
    b = _handle("i2", "x")  # B added first -> index x->B
    reg.add(b)
    a = _handle("i1", "x")  # A added second -> collision, index x->A
    reg.add(a)
    assert reg.resolve_by_model("x") is a

    reg.set_models(b, [ModelSpec(name="y")])  # B moves off 'x'

    assert reg.resolve_by_model("x") is a  # A's 'x' survives (guard)
    assert reg.resolve_by_model("y") is b


def test_set_models_empty_list_does_not_steal_won_name() -> None:
    """set_models([]) on a handle that lost a collision must not resurrect its
    stale alias and steal the name back from the winning handle."""
    reg = BackendRegistry()
    b = _handle("i2", "x")
    reg.add(b)
    a = _handle("i1", "x")  # A wins 'x'
    reg.add(a)
    assert reg.resolve_by_model("x") is a

    reg.set_models(b, [])  # B clears its list

    assert reg.resolve_by_model("x") is a  # A keeps 'x'
    assert b.served_names == []
