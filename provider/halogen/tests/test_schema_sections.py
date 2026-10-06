"""Phase 12 halogen schema tests: the shipped schema.json is valid
2020-12, a canonical sectioned backend_config validates against it, the
frozen fingerprint pins the consensus contract, the sections→options
adapter + env build honor every wired field, port resolution is atomic,
and the old flat shape (top-level model/tokenizer/ports + options blob)
is rejected at the top level."""

import json
from typing import Any

from jsonschema import Draft202012Validator
from provider_lib.schema import load_schema, schema_fingerprint, validate_backend_config

from provider_halogen.driver import SCHEMA
from provider_halogen.env import (
    CONFIG_SECTIONS,
    ENV_MAP,
    build_env,
    flatten_options,
    resolve_ports,
)
from provider_halogen.main import PROVIDER_TYPE

# Canonical example backend_config for halogen (sectioned, Phase 12).
# Exercises every section with realistic values.
CANONICAL_CONFIG: dict[str, Any] = {
    "artifacts": {
        "model": {
            "source": "hf",
            "repo": "peonist-ai/halogen-qwen3.8-27b",
            "file": "qwen3.8-27b-p1w4d-d2.hgn",
        },
        "tokenizer": {"path": "/models/tokenizer"},
    },
    "cache": {"cache_mb": 2048, "cache_reserve_mb": 256, "cache_align": 16},
    "concurrency": {"kv_slots": 4, "slot_ctx": 4096, "queue_timeout": 300},
    "quantization": {"w4a4": 1, "w4a4_excl": "attn"},
    "speculative": {"drafter": "mtp"},
    "server": {"max_tokens_cap": 8192, "keepalive_timeout": 60, "sse_keepalive_s": 15},
    "networking": {"api_port": 8082, "engine_port": 8083},
}


def test_shipped_schema_loads_and_is_valid_2020_12() -> None:
    schema = load_schema("provider_halogen")
    assert schema == SCHEMA
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


def test_schema_has_the_planned_sections() -> None:
    sections = list(SCHEMA["properties"])
    assert sections == [
        "artifacts",
        "cache",
        "concurrency",
        "quantization",
        "speculative",
        "server",
        "networking",
    ]
    for name, section in SCHEMA["properties"].items():
        assert section["title"], name
        assert isinstance(section.get("x-order"), int), name
    orders = [SCHEMA["properties"][k]["x-order"] for k in sections]
    assert orders == sorted(orders) == list(range(1, len(sections) + 1))
    # The adapter flattens exactly the non-artifacts sections.
    assert CONFIG_SECTIONS == tuple(s for s in sections if s != "artifacts")


# Frozen fingerprint of the shipped provider_halogen/schema.json — the
# fleet consensus contract (docs/ws-protocol.md §2). If you change
# schema.json, every known halogen instance must re-register with the
# new file (or the operator force-commits); update this pin deliberately.
SHIPPED_SCHEMA_FP = "07e4a3b40e6bbb5470f87b2e80f68a6263ceeb3333cfed42a180336cc9d24717"


def test_schema_fingerprint_is_stable() -> None:
    fp = schema_fingerprint(SCHEMA)
    assert fp == SHIPPED_SCHEMA_FP
    # Round-trip determinism (serialization-independent digest).
    assert fp == schema_fingerprint(json.loads(json.dumps(SCHEMA)))
    assert len(fp) == 64


def test_canonical_config_validates() -> None:
    assert validate_backend_config(SCHEMA, CANONICAL_CONFIG) == []


def test_artifacts_required_model() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    del cfg["artifacts"]["model"]
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("model" in e for e in errors)


def test_empty_config_validates() -> None:
    # Every section is optional at the top level; only artifacts.model is
    # required when the artifacts section is present.
    assert validate_backend_config(SCHEMA, {}) == []
    assert validate_backend_config(SCHEMA, {"concurrency": {"kv_slots": 2}}) == []


def test_schema_rejects_old_flat_shape() -> None:
    # The pre-Phase-12 flat shape (top-level model/tokenizer/ports +
    # options blob) must be rejected by additionalProperties: false.
    old = {
        "model": {"source": "hf", "repo": "r", "file": "f.hgn"},
        "tokenizer": {"path": "/tok"},
        "api_port": 8082,
        "engine_port": 8083,
        "options": {"kv_slots": 4},
    }
    errors = validate_backend_config(SCHEMA, old)
    joined = " ".join(errors)
    for key in ("model", "tokenizer", "api_port", "engine_port", "options"):
        assert key in joined, key


def test_schema_rejects_unknown_keys_in_sections() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["concurrency"]["nonsense"] = 1
    cfg["cache"]["cache_disk_gib"] = 64
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("nonsense" in e for e in errors)
    assert any("cache_disk_gib" in e for e in errors)


def test_schema_rejects_out_of_range_and_bad_enum() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["concurrency"]["kv_slots"] = 0
    cfg["quantization"]["w4a4"] = 2
    cfg["networking"]["api_port"] = 70000
    cfg["cache"]["cache_mb"] = -1
    errors = validate_backend_config(SCHEMA, cfg)
    joined = " ".join(errors)
    for token in ("kv_slots", "w4a4", "api_port", "cache_mb"):
        assert token in joined, token


def test_null_semantics_for_optional_fields() -> None:
    # None-skip contract: explicit nulls are schema-legal (never emitted;
    # engine keeps its default).
    cfg = {
        "artifacts": {"model": {"path": "/c.hgn"}},
        "cache": {"cache_mb": None, "cache_align": None},
        "concurrency": {"slot_ctx": None, "queue_timeout": None},
        "quantization": {"w4a4": None, "w4a4_excl": None},
        "speculative": {"drafter": None},
        "server": {"max_tokens_cap": None, "keepalive_timeout": None},
    }
    assert validate_backend_config(SCHEMA, cfg) == []
    env = build_env(
        flatten_options(cfg),
        api_port=1,
        engine_port=2,
        checkpoint_path="/c.hgn",
        tokenizer_path="/t",
        base_env={},
    )
    for var in ENV_MAP.values():
        assert var not in env


# Fixed build_env wiring: these HALOGEN_* vars are always exported by
# build_env (ports, bind, engine address, artifacts), never through the
# ENV_MAP options loop.
FIXED_WIRING = {
    "HALOGEN_API_PORT",
    "HALOGEN_PORT",
    "HALOGEN_BIND",
    "HALOGEN_ENGINE",
    "HALOGEN_CHECKPOINT",
    "HALOGEN_TOKENIZER",
}


def _wired_and_unwired_flags() -> tuple[set[str], set[str]]:
    wired: set[str] = set()
    unwired: set[str] = set()
    for section in SCHEMA["properties"].values():
        for field in section["properties"].values():
            flag = field.get("x-flag")
            if not flag:
                continue
            (unwired if field.get("x-supported") is False else wired).add(flag)
    return wired, unwired


def test_every_field_has_description_and_flag() -> None:
    for section_name, section in SCHEMA["properties"].items():
        for key, field in section["properties"].items():
            label = f"{section_name}.{key}"
            assert field.get("description"), label
            assert field.get("title"), label
            # Artifact fields are $ref-based (descriptor objects have no
            # scalar default); networking ports have no schema default
            # either (the pair derives from PROVIDER_PORT when unpinned).
            no_scalar_default = "$ref" in field or (
                section_name == "networking" and key in ("api_port", "engine_port")
            )
            assert "default" in field or no_scalar_default, label
            # Every field names its exact HALOGEN_* env var.
            assert field.get("x-flag"), label
            assert str(field["x-flag"]).startswith("HALOGEN_"), label
            # A wired field must be reachable through ENV_MAP or the fixed
            # build_env/driver wiring; an unwired one is x-supported: false.
            if field.get("x-supported") is not False:
                reachable = (
                    field["x-flag"] in ENV_MAP.values()
                    or field["x-flag"] in FIXED_WIRING
                )
                assert reachable, label


def test_all_env_map_keys_are_wired_schema_fields() -> None:
    wired, unwired = _wired_and_unwired_flags()
    # Every one of the 12 ENV_MAP keys is present in a section and wired.
    assert set(ENV_MAP.values()) <= wired
    assert unwired == set()
    # All 12 semantic options + 2 ports + 2 artifacts = 16 fields.
    total_fields = sum(len(sec["properties"]) for sec in SCHEMA["properties"].values())
    assert (
        total_fields == len(ENV_MAP) + len(FIXED_WIRING) - 2
    )  # BIND/ENGINE not fields
    assert len(wired) == total_fields


# --- env-build spot-checks per section -----------------------------------


def _build(cfg: dict[str, Any], **kw: Any) -> dict[str, str]:
    defaults: dict[str, Any] = {
        "api_port": 8082,
        "engine_port": 8083,
        "checkpoint_path": "/c.hgn",
        "tokenizer_path": "/tok",
        "base_env": {},
    }
    defaults.update(kw)
    return build_env(flatten_options(cfg), **defaults)  # type: ignore[arg-type]


def test_env_spotcheck_concurrency_section() -> None:
    env = _build(
        {"concurrency": {"kv_slots": 12, "slot_ctx": 8192, "queue_timeout": 45}}
    )
    assert env["HALOGEN_KV_SLOTS"] == "12"
    assert env["HALOGEN_SLOT_CTX"] == "8192"
    assert env["HALOGEN_QUEUE_TIMEOUT"] == "45"


def test_env_spotcheck_cache_and_quantization_sections() -> None:
    env = _build(
        {
            "cache": {"cache_mb": 4096, "cache_reserve_mb": 512, "cache_align": 32},
            "quantization": {"w4a4": 1, "w4a4_excl": "attn"},
        }
    )
    assert env["HALOGEN_CACHE_MB"] == "4096"
    assert env["HALOGEN_CACHE_RESERVE_MB"] == "512"
    assert env["HALOGEN_CACHE_ALIGN"] == "32"
    assert env["HALOGEN_W4A4"] == "1"
    assert env["HALOGEN_W4A4_EXCL"] == "attn"


def test_env_spotcheck_speculative_drafter() -> None:
    env = _build({"speculative": {"drafter": "mtp"}})
    assert env["HALOGEN_DRAFTER"] == "mtp"


def test_env_spotcheck_server_section() -> None:
    env = _build(
        {
            "server": {
                "max_tokens_cap": 2048,
                "keepalive_timeout": 90,
                "sse_keepalive_s": 10,
            }
        }
    )
    assert env["HALOGEN_MAX_TOKENS_CAP"] == "2048"
    assert env["HALOGEN_KEEPALIVE_TIMEOUT"] == "90"
    assert env["HALOGEN_SSE_KEEPALIVE_S"] == "10"


def test_env_spotcheck_networking_ports_are_fixed_wiring_not_options() -> None:
    # Ports flow through resolve_ports + build_env kwargs, never ENV_MAP.
    cfg = {"networking": {"api_port": 9401, "engine_port": 9402}}
    api, engine = resolve_ports(cfg, 8081)
    assert (api, engine) == (9401, 9402)
    env = _build(cfg, api_port=api, engine_port=engine)
    assert env["HALOGEN_API_PORT"] == "9401"
    assert env["HALOGEN_PORT"] == "9402"
    assert "api_port" not in ENV_MAP and "engine_port" not in ENV_MAP


def test_resolve_ports_is_atomic() -> None:
    # D3 M1 lesson: a half-pinned pair never mixes sources.
    # Full networking pair wins over everything.
    assert resolve_ports(
        {"networking": {"api_port": 1111, "engine_port": 2222}, "engine_port": 9999},
        8081,
    ) == (1111, 2222)
    # networking with only ONE port → ignored, default pair used.
    assert resolve_ports({"networking": {"api_port": 1111}}, 8081) == (8082, 8083)
    assert resolve_ports({"networking": {"engine_port": 2222}}, 8081) == (8082, 8083)
    # Legacy top-level pair (api_port or backend_port alias) honored only
    # when BOTH present.
    assert resolve_ports({"api_port": 3333, "engine_port": 4444}, 8081) == (
        3333,
        4444,
    )
    assert resolve_ports({"backend_port": 3333, "engine_port": 4444}, 8081) == (
        3333,
        4444,
    )
    assert resolve_ports({"api_port": 3333}, 8081) == (8082, 8083)
    assert resolve_ports({"engine_port": 4444}, 8081) == (8082, 8083)
    # Nothing pinned → PROVIDER_PORT + 1 / + 2.
    assert resolve_ports({}, 8081) == (8082, 8083)


def test_canonical_config_flattens_into_env_completely() -> None:
    """Every wired (x-supported != false) non-artifacts schema field with
    a value in the canonical config emits its exact env var."""
    env = _build(CANONICAL_CONFIG)
    for section_name, section in SCHEMA["properties"].items():
        if section_name == "artifacts":
            continue
        cfg_section = CANONICAL_CONFIG.get(section_name) or {}
        for key, field in section["properties"].items():
            if field.get("x-supported") is False or key not in cfg_section:
                continue
            if field["x-flag"] not in ENV_MAP.values():
                continue
            assert env.get(field["x-flag"]) == str(cfg_section[key]), (
                f"{section_name}.{key} -> {field['x-flag']}"
            )


def test_flatten_options_merges_sections_and_prefers_them() -> None:
    flat = flatten_options(
        {
            "options": {"kv_slots": 1, "drafter": "legacy"},
            "concurrency": {"kv_slots": 8},
            "speculative": {"drafter": "mtp"},
        }
    )
    assert flat["kv_slots"] == 8
    assert flat["drafter"] == "mtp"
    # artifacts never flatten into semantic options.
    assert "artifacts" not in flatten_options(CANONICAL_CONFIG)


def test_numeric_effect_annotations_present() -> None:
    numeric = {
        f"{s}.{k}"
        for s, sec in SCHEMA["properties"].items()
        for k, f in sec["properties"].items()
        if f.get("x-numeric-effect") == "numeric"
    }
    bitwise = {
        f"{s}.{k}"
        for s, sec in SCHEMA["properties"].items()
        for k, f in sec["properties"].items()
        if f.get("x-numeric-effect") == "bitwise"
    }
    # w4a4 lowers matmul precision → outputs change; drafter only
    # proposes (verification keeps outputs bitwise-identical).
    assert {"quantization.w4a4"} <= numeric
    assert {"speculative.drafter", "concurrency.kv_slots"} <= bitwise
    for s, sec in SCHEMA["properties"].items():
        for k, f in sec["properties"].items():
            assert f.get("x-numeric-effect") in (None, "bitwise", "numeric"), f"{s}.{k}"


# --- driver: artifacts section + revision passthrough + ports ----------


def test_driver_reads_nested_sections(tmp_path) -> None:
    from conftest import make_settings

    from provider_halogen.driver import HalogenBackend

    settings = make_settings(tmp_path, "halogen-entrypoint")
    driver = HalogenBackend(settings, {})
    driver.apply_config(CANONICAL_CONFIG)
    assert driver._config == CANONICAL_CONFIG
    assert driver.api_port == 8082
    assert driver.engine_port == 8083
    assert driver.effective_capacity == 4
    # Unpinned → default pair from PROVIDER_PORT.
    driver.apply_config({"concurrency": {"kv_slots": 2}})
    assert driver.api_port == settings.PROVIDER_PORT + 1
    assert driver.engine_port == settings.PROVIDER_PORT + 2
    # Legacy top-level pair still honored.
    driver.apply_config({"api_port": 7777, "engine_port": 7778})
    assert (driver.api_port, driver.engine_port) == (7777, 7778)
    # Legacy half-pair is ignored (atomic), not mixed with the section.
    driver.apply_config({"networking": {"api_port": 6001}, "engine_port": 6002})
    assert (driver.api_port, driver.engine_port) == (
        settings.PROVIDER_PORT + 1,
        settings.PROVIDER_PORT + 2,
    )


def test_apply_config_resets_resolved_artifacts(tmp_path) -> None:
    from conftest import make_settings

    from provider_halogen.driver import HalogenBackend

    driver = HalogenBackend(make_settings(tmp_path, "bin"), {})
    driver.resolved_artifacts = ["/models/old.hgn"]
    driver.apply_config(CANONICAL_CONFIG)
    assert driver.resolved_artifacts == []


async def test_revision_descriptor_reaches_downloader(tmp_path, monkeypatch) -> None:
    """The schema's optional `revision` on hfFile descriptors is passed
    through to provider_lib.downloader.ensure_artifact for both the
    checkpoint and the tokenizer."""
    from conftest import make_settings

    from provider_halogen import driver as driver_mod

    calls: list[dict[str, Any]] = []

    async def fake_ensure_artifact(descriptor, **kwargs):
        calls.append({"descriptor": descriptor, **kwargs})
        return f"/models/{descriptor.get('file', 'x')}"

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)

    driver = driver_mod.HalogenBackend(make_settings(tmp_path, "bin"), {})
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["artifacts"] = {
        "model": {
            "source": "hf",
            "repo": "peonist-ai/halogen-qwen3.8-27b",
            "file": "m.hgn",
            "revision": "v3",
        },
        "tokenizer": {
            "source": "hf",
            "repo": "peonist-ai/halogen-qwen3.8-27b",
            "file": "tokenizer.bin",
            "revision": "  ",
        },
    }
    driver.apply_config(cfg)
    checkpoint, tokenizer = await driver._resolve_artifacts()
    assert (checkpoint, tokenizer) == ("/models/m.hgn", "/models/tokenizer.bin")
    by_file = {c["descriptor"]["file"]: c for c in calls}
    assert by_file["m.hgn"]["revision"] == "v3"
    # Blank revision normalizes to None (downloader default), never "  ".
    assert by_file["tokenizer.bin"]["revision"] is None


async def test_legacy_top_level_artifact_fallback(tmp_path, monkeypatch) -> None:
    """Hand-rolled configs with top-level model/tokenizer (no `artifacts`
    section) still resolve via the driver's fallback."""
    from conftest import make_settings

    from provider_halogen import driver as driver_mod

    seen: list[Any] = []

    async def fake_ensure_artifact(descriptor, **_kwargs):
        seen.append(descriptor)
        return f"/models/{descriptor.get('file', 'x')}"

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)
    legacy = {
        "model": {"source": "hf", "repo": "r", "file": "main.hgn"},
        "tokenizer": {"source": "hf", "repo": "r", "file": "tokenizer.bin"},
    }
    driver = driver_mod.HalogenBackend(make_settings(tmp_path, "bin"), legacy)
    checkpoint, tokenizer = await driver._resolve_artifacts()
    assert (checkpoint, tokenizer) == ("/models/main.hgn", "/models/tokenizer.bin")
    assert seen == [legacy["model"], legacy["tokenizer"]]
    # Local `{"path": ...}` artifacts resolve without the downloader.
    ckpt = tmp_path / "m" / "main.hgn"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"hgn")
    tok = tmp_path / "m" / "tokenizer"
    tok.mkdir(exist_ok=True)
    driver.apply_config({"model": {"path": str(ckpt)}, "tokenizer": {"path": str(tok)}})
    assert await driver._resolve_artifacts() == (str(ckpt), str(tok))


def test_provider_type_constant_matches_schema_context() -> None:
    assert PROVIDER_TYPE == "halogen"
