"""Phase 12 mock schema tests: the shipped schema.json is valid 2020-12,
a canonical sectioned backend_config validates against it, and the mock
driver reads the new nested shape."""

import json
from typing import Any

from jsonschema import Draft202012Validator
from provider_lib.schema import load_schema, schema_fingerprint, validate_backend_config

from provider_mock.backend import SCHEMA, MockBackend
from provider_mock.main import PROVIDER_TYPE

# Canonical example backend_config for the mock (also mirrored in
# scripts/dev.sh's seed + development.md).
CANONICAL_CONFIG: dict[str, Any] = {
    "artifacts": {
        "model": {"source": "hf", "repo": "mock-org/models", "file": "mock.gguf"}
    },
    "context": {"ctx": 4096, "predict": -1},
    "sampling": {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "seed": -1},
    "stream": {"delta_count": 5, "delta_delay": 0.01},
    "server": {"backend_port": 8082},
}


def test_shipped_schema_loads_and_is_valid_2020_12() -> None:
    schema = load_schema("provider_mock")
    assert schema == SCHEMA
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


# Frozen fingerprint of the shipped provider_mock/schema.json — the
# fleet consensus contract (docs/ws-protocol.md §2). If you change
# schema.json, every known mock instance must re-register with the new
# file (or the operator force-commits); update this pin deliberately.
SHIPPED_SCHEMA_FP = "5dabfdb6e16d023faef81de7e1256399cf1ffe7b90983aa0144a5b9cff9af564"


def test_shipped_schema_declares_serves_modalities() -> None:
    # Phase 18 slice 6 added embedding; Phase 24 (audio) added tts + asr so
    # the mock can host speech/transcription definitions for the data plane.
    assert SCHEMA.get("x-serves-modalities") == ["llm", "embedding", "tts", "asr"]


def test_schema_fingerprint_is_stable() -> None:
    fp = schema_fingerprint(SCHEMA)
    assert fp == SHIPPED_SCHEMA_FP
    # Round-trip determinism (serialization-independent digest).
    assert fp == schema_fingerprint(json.loads(json.dumps(SCHEMA)))
    assert len(fp) == 64


def test_canonical_config_validates() -> None:
    assert validate_backend_config(SCHEMA, CANONICAL_CONFIG) == []


def test_local_path_artifact_descriptor_validates() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["artifacts"]["model"] = {"path": "/models/mock.gguf"}
    assert validate_backend_config(SCHEMA, cfg) == []


def test_empty_and_minimal_configs_validate() -> None:
    assert validate_backend_config(SCHEMA, {}) == []
    assert validate_backend_config(SCHEMA, {"context": {"ctx": 128}}) == []


def test_schema_rejects_unknown_top_level_section() -> None:
    errors = validate_backend_config(SCHEMA, {"args": {"ctx": 4096}})
    assert errors
    assert any("args" in e for e in errors)


def test_schema_rejects_out_of_range_values() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["context"]["ctx"] = 0
    cfg["sampling"]["temperature"] = 3.0
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("ctx" in e for e in errors)
    assert any("temperature" in e for e in errors)


def test_driver_reads_nested_sections() -> None:
    driver = MockBackend()
    driver.apply_config(CANONICAL_CONFIG)
    assert driver.context == {"ctx": 4096, "predict": -1}
    assert driver.sampling == {
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "seed": -1,
    }
    assert driver.artifacts == CANONICAL_CONFIG["artifacts"]
    assert driver.server == {"backend_port": 8082}
    # The stream section drives the mock's real behavioral knobs.
    assert driver.delta_count == 5
    assert driver.delta_delay == 0.01
    assert driver.applied_configs == [CANONICAL_CONFIG]


def test_apply_config_resets_resolved_artifacts() -> None:
    driver = MockBackend()
    driver.resolved_artifacts = ["/models/old.gguf"]
    driver.apply_config(CANONICAL_CONFIG)
    assert driver.resolved_artifacts == []


def test_apply_config_partial_sections_keep_defaults_elsewhere() -> None:
    driver = MockBackend(model="m", delta_count=3)
    driver.apply_config({"sampling": {"temperature": 1.5}})
    assert driver.sampling == {"temperature": 1.5}
    # Untouched sections stay empty dicts; stream knobs stay as-is.
    assert driver.context == {}
    assert driver.delta_count == 3


def test_apply_config_empty_is_safe() -> None:
    driver = MockBackend()
    driver.apply_config({})
    driver.apply_config(None)  # type: ignore[arg-type]
    assert driver.artifacts == {}
    assert len(driver.applied_configs) == 2


def test_provider_type_constant_matches_schema_context() -> None:
    assert PROVIDER_TYPE == "mock"
