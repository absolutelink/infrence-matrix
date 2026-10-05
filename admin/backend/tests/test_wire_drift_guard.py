"""Drift guard for the duplicated wire envelope.

The admin ships its own copy of the frame envelope
(``app.services.wire``) because the production admin image is built with
``uv sync --no-install-workspace --package matrix-admin`` and does not
include provider packages. The canonical definition lives in
``provider_lib.wire``.

This test runs in the dev/CI environment where BOTH packages are installed
(CI ``test-admin`` uses a full ``uv sync --frozen``), and asserts the two
copies expose an identical public surface. If they diverge, this fails and
reminds the author to sync them. In an isolated env where provider_lib is
absent, the test skips rather than false-failing.
"""

import importlib.util

import pytest

# NOTE: query the top-level "provider_lib" name, not "provider_lib.wire" —
# find_spec() on a dotted name imports the parent package and raises
# ModuleNotFoundError (not returns None) when it is absent, which would
# crash collection in an isolated admin-only environment.
try:
    _provider_lib_spec = importlib.util.find_spec("provider_lib")
except ModuleNotFoundError:
    _provider_lib_spec = None

if _provider_lib_spec is None:
    pytest.skip(
        "provider_lib not installed (isolated admin env); drift guard N/A",
        allow_module_level=True,
    )

import provider_lib.wire as canonical  # noqa: E402

from app.services import wire as mirror  # noqa: E402


def _public_constants(cls: type) -> dict[str, object]:
    return {
        k: v
        for k, v in vars(cls).items()
        if not k.startswith("_") and isinstance(v, (str, int))
    }


def test_protocol_version_matches() -> None:
    assert mirror.PROTOCOL_VERSION == canonical.PROTOCOL_VERSION


def test_frame_fields_match() -> None:
    assert set(mirror.Frame.model_fields) == set(canonical.Frame.model_fields)
    assert mirror.Frame.model_fields.keys() == canonical.Frame.model_fields.keys()


def test_ack_fields_match() -> None:
    assert set(mirror.Ack.model_fields) == set(canonical.Ack.model_fields)


def test_frame_kind_constants_match() -> None:
    assert _public_constants(mirror.FrameKind) == _public_constants(canonical.FrameKind)


def test_backend_status_values_match() -> None:
    assert _public_constants(mirror.BackendStatusValue) == _public_constants(
        canonical.BackendStatusValue
    )


def test_instance_status_values_match() -> None:
    assert _public_constants(mirror.InstanceStatusValue) == _public_constants(
        canonical.InstanceStatusValue
    )
