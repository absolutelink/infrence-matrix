"""Phase 24 talkies schema tests: the shipped schema.json is valid 2020-12,
declares the audio modality pair (and deliberately NO backend cap), a
canonical sectioned backend_config validates against it, the frozen
fingerprint pins the consensus contract, and the field-metadata rules
(title/description/default on every field, x-secret on the token) hold."""

import json
from typing import Any

from jsonschema import Draft202012Validator
from provider_lib.schema import load_schema, schema_fingerprint, validate_backend_config

from provider_talkies.driver import SCHEMA
from provider_talkies.main import PROVIDER_TYPE

# Canonical example backend_config for talkies (sectioned, Phase 24).
CANONICAL_CONFIG: dict[str, Any] = {
    "model": {"slug": "qwen3-tts-1.7b-custom", "revision": None},
    "engine": {"device": "cuda"},
    "defaults": {
        "voice": "Vivian",
        "language": "en",
        "response_format": "pcm",
        "speed": 1.0,
        "instructions": "warm narrator",
    },
    "limits": {
        "model_concurrency": 1,
        "max_upload_bytes": 104857600,
        "model_ttl": 0,
    },
    "security": {"auth_token": "s3cret"},
}

# Frozen fingerprint of the shipped provider_talkies/schema.json — the
# fleet consensus contract (docs/ws-protocol.md §2). If you change
# schema.json, every known talkies instance must re-register with the new
# file (or the operator force-commits); update this pin deliberately.
SHIPPED_SCHEMA_FP = "8299a8e312cb293acffaad7a8c539b9094fa8a7fa18bc881bd9d84a22d0e87a1"


def test_shipped_schema_loads_and_is_valid_2020_12() -> None:
    schema = load_schema("provider_talkies")
    assert schema == SCHEMA
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


def test_schema_declares_audio_modalities_and_no_cap() -> None:
    # Phase 24: speech ships as the two concrete modalities tts + asr.
    assert SCHEMA["x-serves-modalities"] == ["tts", "asr"]
    # VRAM admission governs: multiple talkies processes per agent are
    # allowed, so there is deliberately no x-max-running-backends key.
    assert "x-max-running-backends" not in SCHEMA


def test_schema_has_the_planned_sections() -> None:
    sections = list(SCHEMA["properties"])
    assert sections == ["model", "engine", "defaults", "limits", "security"]
    for name, section in SCHEMA["properties"].items():
        assert section["title"], name
        assert section["additionalProperties"] is False, name
        assert isinstance(section.get("x-order"), int), name
    orders = [SCHEMA["properties"][k]["x-order"] for k in sections]
    assert orders == sorted(orders) == list(range(1, len(sections) + 1))


def test_schema_fingerprint_is_stable() -> None:
    fp = schema_fingerprint(SCHEMA)
    assert fp == SHIPPED_SCHEMA_FP
    assert fp == schema_fingerprint(json.loads(json.dumps(SCHEMA)))
    assert len(fp) == 64


def test_canonical_config_validates() -> None:
    assert validate_backend_config(SCHEMA, CANONICAL_CONFIG) == []


def test_minimal_config_validates() -> None:
    # Only the slug is required; every other field has a default.
    assert validate_backend_config(SCHEMA, {"model": {"slug": "kokoro-82m"}}) == []


def test_slug_required() -> None:
    assert any("model" in e for e in validate_backend_config(SCHEMA, {}))
    errors = validate_backend_config(SCHEMA, {"model": {}})
    assert any("slug" in e for e in errors)


def test_schema_rejects_unknown_keys_everywhere() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["nonsense"] = 1
    cfg["model"]["extra"] = 1
    cfg["limits"]["top_m"] = 5
    errors = validate_backend_config(SCHEMA, cfg)
    joined = " ".join(errors)
    for token in ("nonsense", "extra", "top_m"):
        assert token in joined, token


def test_schema_rejects_bad_types_and_ranges() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["limits"]["model_concurrency"] = 0
    cfg["limits"]["model_ttl"] = -1
    cfg["defaults"]["response_format"] = "ogg"
    cfg["defaults"]["speed"] = 10.0
    cfg["model"]["slug"] = 5
    errors = validate_backend_config(SCHEMA, cfg)
    joined = " ".join(errors)
    for token in ("model_concurrency", "model_ttl", "response_format", "speed", "slug"):
        assert token in joined, token


def test_auth_token_is_marked_secret() -> None:
    field = SCHEMA["properties"]["security"]["properties"]["auth_token"]
    assert field.get("x-secret") is True
    assert field["default"] == ""


def test_model_ttl_default_pins() -> None:
    field = SCHEMA["properties"]["limits"]["properties"]["model_ttl"]
    assert field["default"] == 0
    assert "VRAM" in field["description"]


def test_defaults_response_format_is_speech_only() -> None:
    # L3: the recorded default must be documented as speech-only; a
    # transcription response_format always comes from the client request
    # (the driver likewise never injects it — see test_transcribe_does_not_leak_speech_defaults).
    desc = SCHEMA["properties"]["defaults"]["properties"]["response_format"][
        "description"
    ]
    assert "speech" in desc.lower()
    assert "never defaulted" in desc.lower() or "client request" in desc.lower()


def test_every_field_has_title_description_and_default() -> None:
    for section_name, section in SCHEMA["properties"].items():
        for key, field in section["properties"].items():
            label = f"{section_name}.{key}"
            assert field.get("title"), label
            assert field.get("description"), label
            # The required slug carries no default (it has no sane one);
            # every optional field must.
            required = key in section.get("required", [])
            assert "default" in field or required, label


def test_provider_type_constant() -> None:
    assert PROVIDER_TYPE == "talkies"
