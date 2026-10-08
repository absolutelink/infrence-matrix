"""Port model overhaul: ``BackendRegistry`` alias index + ``resolve_by_model``.

The agent's single ``/v1`` surface routes an inbound request to the backend
whose ``alias`` equals the request's ``model`` field. The registry maintains an
``alias -> handle`` index kept in sync by ``add``/``remove``.
"""

from __future__ import annotations

from config_fakes import TrackDriver

from provider_lib.backend import BackendLifecycle
from provider_lib.registry import BackendHandle, BackendRegistry


def _handle(iid: str, alias: str | None) -> BackendHandle:
    return BackendHandle(
        iid,
        BackendLifecycle(TrackDriver(), instance_id=iid),
        config_state=None,
        alias=alias,
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
