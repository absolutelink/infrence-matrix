"""Provider-side JSON Schema (2020-12) helpers for backend_config (Phase 12).

Each provider package ships ``provider_<type>/schema.json`` describing the
sectioned shape of its ``backend_config``. These helpers locate and load
the bundled file via ``importlib.resources`` (works from installed wheels
and from the source tree), compute the registration fingerprint, and run
provider-side pre-validation.

The fingerprint canonicalization MUST stay byte-identical to the admin's
``app.services.hashing.canonical_json_sha256`` (docs/ws-protocol.md §2).
Provider code never imports admin packages at runtime; the equality is
pinned by the provider test suite (frozen known vector + a file-path
module load of the admin helper).
"""

import hashlib
import json
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator


def load_schema(package: str, filename: str = "schema.json") -> dict[str, Any]:
    """Load a provider package's bundled JSON Schema.

    ``package`` is the importable package name (e.g. ``"provider_mock"``);
    ``filename`` defaults to ``"schema.json"`` next to the package's
    ``__init__.py``. Raises ``FileNotFoundError`` when the resource is
    missing and ``ValueError`` when it is not a JSON object.
    """
    ref = resources.files(package).joinpath(filename)
    if not ref.is_file():
        raise FileNotFoundError(f"no {filename} bundled in package {package!r}")
    schema = json.loads(ref.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError(f"{package}:{filename} must contain a JSON object")
    return schema


def schema_fingerprint(schema: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of ``schema``.

    Identical canonicalization to the admin's ``canonical_json_sha256``:
    ``sort_keys=True`` + tight separators, UTF-8 encoded.
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_backend_config(
    schema: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    """Provider-side pre-check of a ``backend_config`` against a schema.

    Returns a list of human-readable ``"$.<path>: <message>"`` errors
    (empty when valid). Drivers may call this in ``apply_config`` for
    fail-fast logs; the authoritative gates are the admin's registration
    schema check and definition CRUD validation.
    """
    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda e: [str(p) for p in e.absolute_path],
    )
    out: list[str] = []
    for err in errors:
        path = "$" + "".join(
            f"[{part!r}]" if isinstance(part, int) else f".{part}"
            for part in err.absolute_path
        )
        out.append(f"{path}: {err.message}")
    return out
