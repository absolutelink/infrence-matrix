"""Phase 12 provider_lib.schema tests: bundled-schema loading, the
canonical fingerprint (must equal the admin's canonical_json_sha256),
and provider-side backend_config validation."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from provider_lib.schema import (
    load_schema,
    schema_fingerprint,
    validate_backend_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Frozen vector: sha256 of canonical JSON (sort_keys + tight separators)
# for the tiny schema below. If this ever changes, the canonicalization
# broke and every registered fingerprint in the fleet is invalidated.
FROZEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "integer", "default": 1}},
}
FROZEN_FP = "21ade9d25d78ee396a8042bf5851885d7e60a8c879aed6220ebda66152113ae3"


def test_frozen_vector_is_self_consistent() -> None:
    canonical = json.dumps(FROZEN_SCHEMA, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == FROZEN_FP


def test_schema_fingerprint_matches_frozen_vector() -> None:
    assert schema_fingerprint(FROZEN_SCHEMA) == FROZEN_FP


def test_schema_fingerprint_key_order_independent() -> None:
    reordered = {
        "properties": {"a": {"default": 1, "type": "integer"}},
        "type": "object",
    }
    assert schema_fingerprint(reordered) == FROZEN_FP


def test_fingerprint_equals_admin_canonical_json_sha256() -> None:
    """Cross-check against the admin helper loaded BY FILE PATH.

    The runtime import boundary between admin and provider packages is
    forbidden (AGENTS.md); loading the admin's hashing module directly
    from its source file in the test pins the two canonicalizations
    together without violating it.
    """
    admin_hashing_path = (
        REPO_ROOT / "admin" / "backend" / "app" / "services" / "hashing.py"
    )
    assert admin_hashing_path.is_file(), (
        f"admin hashing module missing: {admin_hashing_path}"
    )
    spec = importlib.util.spec_from_file_location(
        "_admin_hashing_test_load", admin_hashing_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for value in (
        FROZEN_SCHEMA,
        {},
        {"b": [3, {"a": 1}], "a": None},
        {"unicode": "ünïcødé ✓"},
    ):
        assert schema_fingerprint(value) == module.canonical_json_sha256(value)


def test_load_schema_from_package_resource(tmp_path: Any, monkeypatch: Any) -> None:
    """A bundled schema.json next to the package's __init__.py loads via
    importlib.resources (works from wheels and the source tree)."""
    pkg = tmp_path / "fake_provider_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "schema.json").write_text(json.dumps(FROZEN_SCHEMA))
    monkeypatch.syspath_prepend(str(tmp_path))

    schema = load_schema("fake_provider_pkg")
    assert schema == FROZEN_SCHEMA
    assert schema_fingerprint(schema) == FROZEN_FP


def test_load_schema_missing_file_raises(tmp_path: Any, monkeypatch: Any) -> None:
    pkg = tmp_path / "empty_provider_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_schema("empty_provider_pkg")


def test_load_schema_non_object_raises(tmp_path: Any, monkeypatch: Any) -> None:
    pkg = tmp_path / "bad_provider_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "schema.json").write_text("[1, 2, 3]")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError):
        load_schema("bad_provider_pkg")


def test_validate_backend_config_catches_bad_config() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "context": {
                "type": "object",
                "properties": {"ctx": {"type": "integer", "minimum": 1}},
            }
        },
    }
    assert validate_backend_config(schema, {"context": {"ctx": 4096}}) == []
    errors = validate_backend_config(schema, {"context": {"ctx": 0}, "bogus": True})
    assert len(errors) == 2
    # Path-prefixed, human-readable, sorted by path (the root-level
    # additionalProperties error has path "$" and sorts first).
    assert errors[0].startswith("$:")
    assert "bogus" in errors[0]
    assert errors[1].startswith("$.context")
