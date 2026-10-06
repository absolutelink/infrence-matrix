"""Canonical JSON fingerprinting shared by config and schema hashes.

Both ``config_fingerprint`` (backend_config) and ``schema_fingerprint``
(ProviderType.schema) must use the *identical* canonicalization so
fingerprints are comparable across the fleet (docs/ws-protocol.md §2).
"""

import hashlib
import json
from typing import Any


def canonical_json_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
