"""Phase 12 halogen-flash schema tests: the shipped schema.json is valid
2020-12, a canonical sectioned backend_config validates against it, the
frozen fingerprint pins the consensus contract, the sections→options
adapter + env build honor every wired field, and the old flat
`options`-blob shape is rejected at the top level."""

import json
from typing import Any

from jsonschema import Draft202012Validator
from provider_lib.schema import load_schema, schema_fingerprint, validate_backend_config

from provider_halogen_flash.driver import SCHEMA
from provider_halogen_flash.env import (
    CONFIG_SECTIONS,
    ENV_MAP,
    build_env,
    flatten_options,
    resolve_ports,
)
from provider_halogen_flash.main import PROVIDER_TYPE

# Canonical example backend_config for halogen-flash (sectioned,
# Phase 12). Exercises every section with realistic values.
CANONICAL_CONFIG: dict[str, Any] = {
    "artifacts": {
        "model": {
            "source": "hf",
            "repo": "peonist-ai/halogen-qwen3.8-flash-next",
            "file": "qwen38-flash-next-w4b.hgn",
        },
        "tokenizer": {
            "source": "hf",
            "repo": "peonist-ai/halogen-qwen3.8-flash-next",
            "file": "tokenizer",
        },
        "mtp_head": {"path": "/models/mtp-head.hgn"},
        "vision_tower": True,
    },
    "context": {
        "ctx": 131072,
        "max_tok": 16384,
        "rope_yarn": 4,
        "admit_chunk": 4096,
        "indexer_budget": 64,
        "kv_slots": 8,
        "kv_pool_positions": 524288,
        "kv_pool_fit": 1,
        "host_reserve_gib": 20,
    },
    "disk_cache": {
        "cache_dir_enabled": True,
        "cache_disk_gib": 64,
        "cache_prune_old": 1,
        "prompt_cache": 2,
        "cache_inplace": 1,
        "cache_entries": 2048,
        "cache_branches": 4,
        "cache_snap3": 1,
        "cache_full": 0,
        "prefill_chunk": 8192,
    },
    "speculative": {
        "mtp_depth": 2,
        "drafter_default": "mtp",
        "pld": "64,8",
        "spec_adapt": "4,0.5,2",
        "grammar": "json",
    },
    "sampling": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens_cap": 8192,
        "max_tokens_default": 1024,
    },
    "reasoning": {
        "reasoning_effort": "high",
        "enable_thinking": 1,
        "max_thinking_tokens": 16384,
        "thinking_answer_room": 4096,
    },
    "composable": {
        "composable_context": 1,
        "composable_context_floor": 4096,
        "composable_context_bytes": 268435456,
    },
    "vision": {"vision_max_pixels": 1048576},
    "npu": {
        "npu_models": ["qwen3-embedding-0.6b", "qwen3.5-2b"],
    },
    "networking": {},
    "troubleshooting": {},
}


def test_shipped_schema_loads_and_is_valid_2020_12() -> None:
    schema = load_schema("provider_halogen_flash")
    assert schema == SCHEMA
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


def test_schema_has_the_planned_sections() -> None:
    sections = list(SCHEMA["properties"])
    assert sections == [
        "artifacts",
        "context",
        "disk_cache",
        "speculative",
        "sampling",
        "reasoning",
        "composable",
        "vision",
        "npu",
        "networking",
        "troubleshooting",
    ]
    for name, section in SCHEMA["properties"].items():
        assert section["title"], name
        assert isinstance(section.get("x-order"), int), name
    orders = [SCHEMA["properties"][k]["x-order"] for k in sections]
    assert orders == sorted(orders) == list(range(1, len(sections) + 1))
    # The adapter flattens exactly the non-artifacts sections.
    assert CONFIG_SECTIONS == tuple(s for s in sections if s != "artifacts")


# Frozen fingerprint of the shipped provider_halogen_flash/schema.json —
# the fleet consensus contract (docs/ws-protocol.md §2). If you change
# schema.json, every known halogen-flash instance must re-register with
# the new file (or the operator force-commits); update this pin
# deliberately.
# Port model overhaul: the operator-facing `networking.{api_port,engine_port}`
# fields were removed (the engine always uses the static MACHINE_UID-derived
# pair), bumping the fp.
# cache_entries default corrected 1024 -> 24 to match the real engine default,
# bumping the fp.
# vision_tower oneOf branches titled (boolean + local-path string) so the
# admin form dropdown reads meaningfully instead of "option 2/3", bumping the fp.
SHIPPED_SCHEMA_FP = "5d29689168e8eb8a7f8872a9fc692bea3853321cc101b577017f1fe8948d9919"


def test_schema_fingerprint_is_stable() -> None:
    fp = schema_fingerprint(SCHEMA)
    assert fp == SHIPPED_SCHEMA_FP
    # Round-trip determinism (serialization-independent digest).
    assert fp == schema_fingerprint(json.loads(json.dumps(SCHEMA)))
    assert len(fp) == 64


def test_schema_declares_single_running_backend() -> None:
    """Phase 16 slice 6: halogen-flash drives one NPU, so its shipped schema
    declares ``x-max-running-backends: 1`` at the top level. The admin reads
    this into ``ProviderType.max_running_backends`` at every commit point, so
    the scheduler hot-swaps (boots one halogen-flash backend per agent at a
    time)."""
    assert SCHEMA.get("x-max-running-backends") == 1


def test_canonical_config_validates() -> None:
    assert validate_backend_config(SCHEMA, CANONICAL_CONFIG) == []


def test_artifacts_are_all_optional() -> None:
    """Blank artifacts mean "use the image/engine default": the driver omits
    HALOGEN_CHECKPOINT / HALOGEN_TOKENIZER (see env.build_env), so no artifact
    key is required — neither inside the section nor at the top level."""
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    del cfg["artifacts"]["model"]
    assert validate_backend_config(SCHEMA, cfg) == []
    assert validate_backend_config(SCHEMA, {"artifacts": {}}) == []
    assert (
        validate_backend_config(SCHEMA, {"artifacts": {"tokenizer": {"path": "/tok"}}})
        == []
    )
    # The section still closes itself: unknown keys are rejected.
    assert (
        validate_backend_config(SCHEMA, {"artifacts": {"checkpoint": {"path": "/c"}}})
        != []
    )


def test_empty_config_validates() -> None:
    # Every section is optional at the top level, and so is every artifact
    # inside `artifacts` (a fully empty config boots the image defaults).
    assert validate_backend_config(SCHEMA, {}) == []
    assert validate_backend_config(SCHEMA, {"context": {"ctx": 128}}) == []


def test_schema_rejects_old_flat_shape() -> None:
    # The pre-Phase-12 flat shape (top-level model/tokenizer/ports +
    # options blob) must be rejected by additionalProperties: false.
    old = {
        "model": {"source": "hf", "repo": "r", "file": "f.hgn"},
        "tokenizer": {"path": "/tok"},
        "api_port": 8200,
        "engine_port": 8201,
        "options": {"kv_slots": 8},
    }
    errors = validate_backend_config(SCHEMA, old)
    joined = " ".join(errors)
    for key in ("model", "tokenizer", "api_port", "engine_port", "options"):
        assert key in joined, key


def test_schema_rejects_networking_ports() -> None:
    """Port model overhaul: `networking.{api_port,engine_port}` are no longer
    accepted schema properties — the engine always uses the static
    MACHINE_UID-derived pair (env.derive_static_ports). The `networking`
    section's additionalProperties:false rejects them."""
    for key in ("api_port", "engine_port"):
        cfg = json.loads(json.dumps(CANONICAL_CONFIG))
        cfg["networking"][key] = 8200
        errors = validate_backend_config(SCHEMA, cfg)
        assert any(key in e for e in errors), key
    props = SCHEMA["properties"]["networking"]["properties"]
    assert "api_port" not in props and "engine_port" not in props


def test_schema_rejects_unknown_keys_in_sections() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["context"]["nonsense"] = 1
    cfg["disk_cache"]["prompt_cache_mode"] = 3
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("nonsense" in e for e in errors)
    assert any("prompt_cache_mode" in e for e in errors)


def test_schema_rejects_out_of_range_and_bad_enum() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["context"]["kv_slots"] = 65  # max 64
    cfg["context"]["ctx"] = 0
    cfg["sampling"]["temperature"] = 3.0
    cfg["sampling"]["presence_penalty"] = -3.0
    cfg["disk_cache"]["prompt_cache"] = 5
    cfg["reasoning"]["reasoning_effort"] = "ultra"
    cfg["speculative"]["pld"] = "not-a-pair"
    errors = validate_backend_config(SCHEMA, cfg)
    joined = " ".join(errors)
    for token in (
        "kv_slots",
        "ctx",
        "temperature",
        "presence_penalty",
        "prompt_cache",
        "reasoning_effort",
        "pld",
    ):
        assert token in joined, token


def test_pld_and_spec_adapt_patterns() -> None:
    for good in ("64,8", "1,1"):
        cfg = json.loads(json.dumps(CANONICAL_CONFIG))
        cfg["speculative"]["pld"] = good
        assert validate_backend_config(SCHEMA, cfg) == [], good
    for bad in ("64", "8,64,", "a,b", "64, 8"):
        cfg = json.loads(json.dumps(CANONICAL_CONFIG))
        cfg["speculative"]["pld"] = bad
        assert validate_backend_config(SCHEMA, cfg), bad
    for good in ("4,0.5,2", "4,1,2"):
        cfg = json.loads(json.dumps(CANONICAL_CONFIG))
        cfg["speculative"]["spec_adapt"] = good
        assert validate_backend_config(SCHEMA, cfg) == [], good
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["speculative"]["spec_adapt"] = "4,0.5"
    assert validate_backend_config(SCHEMA, cfg)


def test_vision_tower_accepts_descriptor_bool_and_path() -> None:
    for value in (True, False, "/models/vt.hgn", {"path": "/vt.hgn"}):
        cfg = json.loads(json.dumps(CANONICAL_CONFIG))
        cfg["artifacts"]["vision_tower"] = value
        assert validate_backend_config(SCHEMA, cfg) == [], value


def test_npu_models_is_string_array() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["npu"]["npu_models"] = [{"source": "hf", "repo": "r", "file": "f"}]
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("npu_models" in e for e in errors)


# Fixed build_env wiring: these HALOGEN_* vars are always (or
# conditionally) exported by build_env / the driver, never through the
# ENV_MAP options loop.
FIXED_WIRING = {
    "HALOGEN_API_PORT",
    "HALOGEN_PORT",
    "HALOGEN_CHECKPOINT",
    "HALOGEN_TOKENIZER",
    "HALOGEN_CACHE_DIR",
    "HALOGEN_NPU_MODELS",
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


def test_obsolete_or_never_invented_env_flags_absent() -> None:
    # Wired schema flags must be reachable through ENV_MAP or the fixed
    # build_env/driver wiring; unwired flags must NOT appear in ENV_MAP.
    # Ad-hoc inventions (HALOGEN_MODEL, the obsolete llama.cpp-style
    # PROMPT_CACHE, legacy plain-halogen-only vars) never appear at all.
    wired, unwired = _wired_and_unwired_flags()
    allowed = set(ENV_MAP.values()) | FIXED_WIRING
    assert wired <= allowed
    assert unwired & allowed == set()
    # Every ENV_MAP entry is a wired schema flag (no orphan mapping).
    assert set(ENV_MAP.values()) <= wired
    all_flags = wired | unwired
    assert all(str(flag).startswith("HALOGEN_") for flag in all_flags)
    for forbidden in (
        "HALOGEN_MODEL",
        "HALOGEN_PROMPT",
        "PROMPT_CACHE",
        "HALOGEN_CACHE_MB",
        "HALOGEN_CACHE_RESERVE_MB",
        "HALOGEN_SLOT_CTX",
        "HALOGEN_DRAFTER",
        "HALOGEN_W4A4",
    ):
        assert forbidden not in all_flags, forbidden


def test_every_field_has_description_and_flag_or_supported() -> None:
    for section_name, section in SCHEMA["properties"].items():
        for key, field in section["properties"].items():
            label = f"{section_name}.{key}"
            assert field.get("description"), label
            assert field.get("title"), label
            # Artifact fields are $ref-based (descriptor objects have no
            # scalar default); the port fields were removed in the port model
            # overhaul (the engine always uses the static MACHINE_UID pair).
            no_scalar_default = (
                "$ref" in field
                or "oneOf" in field  # vision_tower: descriptor | bool | path
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


def test_wired_set_matches_schema_unsupported_false_complement() -> None:
    wired, unwired = _wired_and_unwired_flags()
    # The known unwired upstream flags (documented, disabled in UI).
    assert "HALOGEN_ADMIT_TICKS" in unwired
    assert "HALOGEN_REPETITION_PENALTY" in unwired
    assert "HALOGEN_CK_OVERLAY" in unwired
    # HALOGEN_DOWNLOAD is wired by this provider (driver-resolved): the
    # engine's own start-time download is what a blank artifact selection
    # relies on, so it must NOT sit in the documented-but-unwired set.
    assert "HALOGEN_DOWNLOAD" in wired
    assert "HALOGEN_DOWNLOAD" not in unwired
    # Wired and unwired sets are disjoint and cover the whole flag list.
    assert wired & unwired == set()
    total_fields = sum(len(sec["properties"]) for sec in SCHEMA["properties"].values())
    assert len(wired) + len(unwired) == total_fields


def test_numeric_effect_annotations_present() -> None:
    numeric = {
        f"{s}.{k}"
        for s, sec in SCHEMA["properties"].items()
        for k, f in sec["properties"].items()
        if f.get("x-numeric-effect") == "numeric"
    }
    # Per FLAGS.md NUMERIC markings (the output-changing subset).
    assert {
        "context.indexer_budget",
        "disk_cache.prompt_cache",
        "composable.composable_context",
        "sampling.temperature",
        "sampling.top_p",
        "sampling.top_k",
        "sampling.min_p",
        "sampling.presence_penalty",
        "sampling.frequency_penalty",
        "sampling.repetition_penalty",
        "reasoning.reasoning_effort",
        "troubleshooting.ck_overlay",
    } <= numeric
    for s, sec in SCHEMA["properties"].items():
        for k, f in sec["properties"].items():
            assert f.get("x-numeric-effect") in (None, "bitwise", "numeric"), f"{s}.{k}"


# --- env-build spot-checks per section -----------------------------------


def _build(cfg: dict[str, Any], **kw: Any) -> dict[str, str]:
    defaults: dict[str, Any] = {
        "api_port": 8200,
        "engine_port": 8201,
        "checkpoint_path": "/c.hgn",
        "tokenizer_path": "/tok",
        "base_env": {},
    }
    defaults.update(kw)
    return build_env(flatten_options(cfg), **defaults)  # type: ignore[arg-type]


def test_env_spotcheck_kv_slots_and_context_section() -> None:
    env = _build({"context": {"kv_slots": 12, "ctx": 65536, "admit_chunk": 2048}})
    assert env["HALOGEN_KV_SLOTS"] == "12"
    assert env["HALOGEN_CTX"] == "65536"
    assert env["HALOGEN_ADMIT_CHUNK"] == "2048"
    # Unwired fields never emit.
    assert "HALOGEN_ADMIT_TICKS" not in env


def test_env_spotcheck_prompt_cache_enum_and_pld_passthrough() -> None:
    env = _build(
        {
            "disk_cache": {"prompt_cache": 1},
            "speculative": {"pld": "64,8", "spec_adapt": "4,0.5,2"},
        }
    )
    assert env["HALOGEN_PROMPT_CACHE"] == "1"
    assert env["HALOGEN_PLD"] == "64,8"
    assert env["HALOGEN_SPEC_ADAPT"] == "4,0.5,2"


def test_env_spotcheck_composable_and_sampling_sections() -> None:
    env = _build(
        {
            "composable": {"composable_context": 1, "composable_context_bytes": 4096},
            "sampling": {"temperature": 0.3, "max_tokens_cap": 512},
            "reasoning": {"reasoning_effort": "low", "enable_thinking": 0},
        }
    )
    assert env["HALOGEN_COMPOSABLE_CONTEXT"] == "1"
    assert env["HALOGEN_COMPOSABLE_CONTEXT_BYTES"] == "4096"
    assert env["HALOGEN_TEMPERATURE"] == "0.3"
    assert env["HALOGEN_MAX_TOKENS_CAP"] == "512"
    assert env["HALOGEN_REASONING_EFFORT"] == "low"
    assert env["HALOGEN_ENABLE_THINKING"] == "0"


def test_env_spotcheck_npu_section_joins() -> None:
    env = _build(
        {"npu": {"npu_models": ["qwen3-embedding-0.6b", "qwen3.5-2b"]}},
        npu_models=["qwen3-embedding-0.6b", "qwen3.5-2b"],
    )
    assert env["HALOGEN_NPU_MODELS"] == "qwen3-embedding-0.6b,qwen3.5-2b"
    # The raw list never stringifies into the env via options (join only).
    assert "['" not in " ".join(env.values())


def test_env_spotcheck_cache_dir_enabled_gating() -> None:
    env = _build({"disk_cache": {"cache_dir_enabled": True}}, cache_dir=None)
    assert "HALOGEN_CACHE_DIR" not in env  # no dir provided → nothing to export
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "fc"
        env = _build({"disk_cache": {"cache_dir_enabled": True}}, cache_dir=cache)
        assert env["HALOGEN_CACHE_DIR"] == str(cache)
        assert cache.is_dir()
        env = _build({"disk_cache": {"cache_dir_enabled": False}}, cache_dir=cache)
        assert "HALOGEN_CACHE_DIR" not in env


def test_env_spotcheck_networking_ports_are_fixed_wiring_not_options() -> None:
    # Ports flow through resolve_ports + build_env kwargs, never ENV_MAP.
    cfg = {"networking": {"api_port": 8250, "engine_port": 8251}}
    api, engine = resolve_ports(cfg, "uid")
    assert (api, engine) == (8250, 8251)
    env = _build(cfg, api_port=api, engine_port=engine)
    assert env["HALOGEN_API_PORT"] == "8250"
    assert env["HALOGEN_PORT"] == "8251"
    # The raw port values never leak through the options loop as other vars.
    assert "api_port" not in ENV_MAP and "engine_port" not in ENV_MAP


def test_canonical_config_flattens_into_env_completely() -> None:
    """Every wired (x-supported != false) non-artifacts schema field with a
    value in the canonical config emits its exact env var (ENV_MAP path;
    fixed-wiring vars like CACHE_DIR/ports are covered by their own
    spot-checks)."""
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
    # Conditional wiring from the canonical config (npu list joined by
    # the driver; emulate what start() passes).
    env = _build(CANONICAL_CONFIG, npu_models=CANONICAL_CONFIG["npu"]["npu_models"])
    assert env["HALOGEN_NPU_MODELS"] == "qwen3-embedding-0.6b,qwen3.5-2b"


def test_provider_type_constant_matches_schema_context() -> None:
    assert PROVIDER_TYPE == "halogen-flash"


# --- driver: artifacts section + revision passthrough + resolved
# vision_tower / mtp_head env wiring -------------------------------------


async def test_driver_artifacts_revision_reaches_downloader(
    tmp_path, monkeypatch
) -> None:
    """H2 mirror: the schema's optional `revision` on hfFile descriptors
    is passed through to provider_lib.downloader.ensure_artifact."""
    from conftest import make_settings

    from provider_halogen_flash import driver as driver_mod

    calls: list[dict[str, Any]] = []

    async def fake_ensure_artifact(descriptor, **kwargs):
        calls.append({"descriptor": descriptor, **kwargs})
        return f"/models/{descriptor.get('file', 'x')}"

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)
    driver = driver_mod.HalogenFlashBackend(make_settings(tmp_path, "bin"), {})
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["artifacts"] = {
        "model": {
            "source": "hf",
            "repo": "peonist-ai/halogen-qwen3.8-flash-next",
            "file": "m.hgn",
            "revision": "v9",
        },
        "tokenizer": {
            "source": "hf",
            "repo": "peonist-ai/halogen-qwen3.8-flash-next",
            "file": "tokenizer",
        },
        "mtp_head": {
            "source": "hf",
            "repo": "peonist-ai/halogen-qwen3.8-flash-next",
            "file": "head.hgn",
            "revision": "  ",
        },
    }
    driver.apply_config(cfg)
    checkpoint = await driver._resolve_artifact(
        driver._artifact_descriptor("model"), "model"
    )
    tokenizer = await driver._resolve_artifact(
        driver._artifact_descriptor("tokenizer"), "tokenizer"
    )
    mtp = await driver._resolve_artifact(
        driver._artifact_descriptor("mtp_head"), "mtp_head"
    )
    assert (checkpoint, tokenizer, mtp) == (
        "/models/m.hgn",
        "/models/tokenizer",
        "/models/head.hgn",
    )
    by_file = {c["descriptor"]["file"]: c for c in calls}
    assert by_file["m.hgn"]["revision"] == "v9"
    # Absent / blank revision → None (downloader default), never "" or "  ".
    assert by_file["tokenizer"]["revision"] is None
    assert by_file["head.hgn"]["revision"] is None


async def test_start_resolves_sectioned_artifacts_and_emits_head_and_tower(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    """start() reads artifacts.{model,tokenizer,mtp_head,vision_tower}:
    resolved paths flow to HALOGEN_MTP_HEAD / HALOGEN_VISION_TOWER and
    `true` becomes 1 (look beside the checkpoint)."""
    from conftest import NPU_UNAVAILABLE, make_settings

    head = tmp_path / "models" / "mtp-head.hgn"
    head.write_bytes(b"hgn")
    settings = make_settings(tmp_path, fake_flash_binary)
    driver = driver_backend(
        settings,
        {
            "artifacts": {
                "model": {"path": local_artifacts["checkpoint"]},
                "tokenizer": {"path": local_artifacts["tokenizer"]},
                "mtp_head": {"path": str(head)},
                "vision_tower": True,
            }
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert recorded["HALOGEN_MTP_HEAD"] == str(head)
        assert recorded["HALOGEN_VISION_TOWER"] == "1"
        assert str(head) in driver.resolved_artifacts
    finally:
        await driver.aclose()


async def test_vision_tower_false_emits_explicit_zero(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    """H1: artifacts.vision_tower false is an explicit off-switch —
    HALOGEN_VISION_TOWER=0 is emitted (not omitted)."""
    from conftest import NPU_UNAVAILABLE, make_settings

    settings = make_settings(tmp_path, fake_flash_binary)
    driver = driver_backend(
        settings,
        {
            "artifacts": {
                "model": {"path": local_artifacts["checkpoint"]},
                "tokenizer": {"path": local_artifacts["tokenizer"]},
                "vision_tower": False,
            }
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert recorded["HALOGEN_VISION_TOWER"] == "0"
    finally:
        await driver.aclose()


async def test_vision_tower_absent_emits_nothing(
    tmp_path, fake_flash_binary, local_artifacts, state_dir
) -> None:
    """Absent vision_tower (and mtp_head) never emits the vars: the
    engine keeps its own default."""
    from conftest import NPU_UNAVAILABLE, make_settings

    settings = make_settings(tmp_path, fake_flash_binary)
    driver = driver_backend(
        settings,
        {
            "artifacts": {
                "model": {"path": local_artifacts["checkpoint"]},
                "tokenizer": {"path": local_artifacts["tokenizer"]},
            }
        },
        npu_probe=lambda: NPU_UNAVAILABLE,
    )
    try:
        await driver.start()
        recorded = json.loads((state_dir / "env.json").read_text())
        assert "HALOGEN_VISION_TOWER" not in recorded
        assert "HALOGEN_MTP_HEAD" not in recorded
    finally:
        await driver.aclose()


def driver_backend(settings: Any, cfg: dict[str, Any], **kw: Any) -> Any:
    from provider_halogen_flash.driver import HalogenFlashBackend

    return HalogenFlashBackend(settings, cfg, **kw)
