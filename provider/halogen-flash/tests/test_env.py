"""Unit tests for the halogen-flash HALOGEN_* env map + static ports.

Phase 12: configs arrive sectioned; `flatten_options()` assembles the
flat semantic dict that ENV_MAP / build_env consume (legacy flat
`cfg["options"]` remains accepted, sections winning on conflicts).
"""

import pytest
from provider_lib.schema import validate_backend_config

from provider_halogen_flash.driver import SCHEMA
from provider_halogen_flash.env import (
    CONFIG_SECTIONS,
    DEFAULT_DOWNLOAD_REPO,
    ENV_MAP,
    FLASH_PORT_END,
    FLASH_PORT_START,
    build_argv,
    build_env,
    derive_static_ports,
    effective_capacity,
    flatten_options,
    resolve_ports,
)


def _env(**opts):
    return build_env(
        opts,
        api_port=8200,
        engine_port=8201,
        checkpoint_path="/models/flash.hgn",
        tokenizer_path="/models/tokenizer",
        base_env={},
    )


def test_env_map_covers_all_ported_options() -> None:
    # 45 semantic HALOGEN_* vars wired by this provider (43 legacy-ported
    # options + the Phase 12 artifacts-resolved mtp_head + the driver-
    # resolved engine auto-download repo).
    assert len(ENV_MAP) == 45
    assert ENV_MAP["kv_slots"] == "HALOGEN_KV_SLOTS"
    assert ENV_MAP["composable_context_bytes"] == "HALOGEN_COMPOSABLE_CONTEXT_BYTES"
    assert ENV_MAP["vision_max_pixels"] == "HALOGEN_VISION_MAX_PIXELS"
    assert ENV_MAP["mtp_head"] == "HALOGEN_MTP_HEAD"


def test_every_mapped_option_emits_its_var() -> None:
    opts = {key: f"v{i}" for i, key in enumerate(ENV_MAP)}
    env = _env(**opts)
    for key, var in ENV_MAP.items():
        assert env[var] == str(opts[key])


def test_none_values_never_emitted() -> None:
    env = _env(kv_slots=None, temperature=None, grammar=None)
    for var in ("HALOGEN_KV_SLOTS", "HALOGEN_TEMPERATURE", "HALOGEN_GRAMMAR"):
        assert var not in env
    assert "None" not in " ".join(env.values())


def test_fixed_wiring_always_set() -> None:
    env = _env()
    assert env["HALOGEN_API_PORT"] == "8200"
    assert env["HALOGEN_PORT"] == "8201"
    assert env["HALOGEN_BIND"] == "127.0.0.1"
    assert env["HALOGEN_ENGINE"] == "127.0.0.1:8201"
    assert env["HALOGEN_CHECKPOINT"] == "/models/flash.hgn"
    assert env["HALOGEN_TOKENIZER"] == "/models/tokenizer"


def test_blank_artifacts_omit_the_engine_paths() -> None:
    """`None` = the operator left artifacts.model / .tokenizer blank: the
    variable is not exported AT ALL so the image's own resolution wins
    (`resolve_checkpoint` for the head, `need_tokenizer` for the sidecar
    tokenizer). An empty string would defeat both fallbacks."""
    env = build_env(
        {},
        api_port=8200,
        engine_port=8201,
        checkpoint_path=None,
        tokenizer_path=None,
        base_env={},
    )
    assert "HALOGEN_CHECKPOINT" not in env
    assert "HALOGEN_TOKENIZER" not in env
    # The rest of the fixed wiring is unaffected.
    assert env["HALOGEN_API_PORT"] == "8200"
    assert env["HALOGEN_BIND"] == "127.0.0.1"

    half = build_env(
        {},
        api_port=1,
        engine_port=2,
        checkpoint_path="/models/flash.hgn",
        tokenizer_path=None,
        base_env={},
    )
    assert half["HALOGEN_CHECKPOINT"] == "/models/flash.hgn"
    assert "HALOGEN_TOKENIZER" not in half


def test_download_repo_is_the_engine_fetch_switch() -> None:
    """`download` → HALOGEN_DOWNLOAD is an ordinary ENV_MAP option, but its
    value is DRIVER-resolved. It is what lets the engine pull the files it is
    missing into MODELS_DIR at start; None (operator opted out with an empty
    `troubleshooting.download`) is never emitted, so the image stays offline.
    """
    env = _env(download="peonist-ai/halogen-qwen3.8-flash-next")
    assert env["HALOGEN_DOWNLOAD"] == "peonist-ai/halogen-qwen3.8-flash-next"
    assert "HALOGEN_DOWNLOAD" not in _env()
    assert "HALOGEN_DOWNLOAD" not in _env(download=None)


def test_default_download_repo_is_the_pinned_weights_repo() -> None:
    assert DEFAULT_DOWNLOAD_REPO == "peonist-ai/halogen-qwen3.8-flash-next"


def test_cache_dir_only_when_enabled(tmp_path) -> None:
    cache = tmp_path / "fcache"
    env = build_env(
        {"cache_dir_enabled": True},
        api_port=1,
        engine_port=2,
        checkpoint_path="/c",
        tokenizer_path="/t",
        cache_dir=cache,
        base_env={},
    )
    assert env["HALOGEN_CACHE_DIR"] == str(cache)
    assert cache.is_dir()

    env = build_env(
        {"cache_dir_enabled": False},
        api_port=1,
        engine_port=2,
        checkpoint_path="/c",
        tokenizer_path="/t",
        cache_dir=cache,
        base_env={},
    )
    assert "HALOGEN_CACHE_DIR" not in env


def test_cache_dir_emitted_when_key_pruned_away(tmp_path) -> None:
    """UI prunes untouched defaults: an ABSENT cache_dir_enabled follows
    the schema default (true) and must export the dir. Production symptom
    this guards: backend_config = {"disk_cache": {"cache_disk_gib": 512}}
    started the engine with HALOGEN_CACHE_DISK_GIB but no HALOGEN_CACHE_DIR
    (disk cache enabled but nowhere to persist)."""
    cache = tmp_path / "fcache-default"
    env = build_env(
        {"cache_disk_gib": 512},
        api_port=1,
        engine_port=2,
        checkpoint_path="/c",
        tokenizer_path="/t",
        cache_dir=cache,
        base_env={},
    )
    assert env["HALOGEN_CACHE_DIR"] == str(cache)
    assert env["HALOGEN_CACHE_DISK_GIB"] == "512"
    assert cache.is_dir()


def test_npu_models_joined_as_bare_comma_list() -> None:
    env = build_env(
        {},
        api_port=1,
        engine_port=2,
        checkpoint_path="/c",
        tokenizer_path="/t",
        npu_models=["qwen3-embedding-0.6b", "qwen3.5-2b"],
        base_env={},
    )
    assert env["HALOGEN_NPU_MODELS"] == "qwen3-embedding-0.6b,qwen3.5-2b"
    assert "[" not in env["HALOGEN_NPU_MODELS"]  # never a Python repr


def test_no_npu_models_key_when_empty() -> None:
    env = build_env(
        {},
        api_port=1,
        engine_port=2,
        checkpoint_path="/c",
        tokenizer_path="/t",
        npu_models=None,
        base_env={},
    )
    assert "HALOGEN_NPU_MODELS" not in env


# --- Static ports / disk-cache fingerprint -------------------------------


def test_derive_static_ports_deterministic() -> None:
    a1, e1 = derive_static_ports("machine-x")
    a2, e2 = derive_static_ports("machine-x")
    assert (a1, e1) == (a2, e2)


def test_derive_static_ports_in_range_and_paired() -> None:
    for uid in ("a", "b", "machine-1", "core-10-100-2-111"):
        api, engine = derive_static_ports(uid)
        assert FLASH_PORT_START <= api < FLASH_PORT_END
        assert engine == api + 1
        assert FLASH_PORT_START <= engine < FLASH_PORT_END


def test_derive_static_ports_differ_across_machines() -> None:
    assert derive_static_ports("machine-1") != derive_static_ports("machine-2")


def test_resolve_ports_prefers_explicit_config() -> None:
    api, engine = resolve_ports({"api_port": 9100, "engine_port": 9101}, "uid")
    assert (api, engine) == (9100, 9101)


def test_resolve_ports_reads_networking_section() -> None:
    api, engine = resolve_ports(
        {"networking": {"api_port": 9200, "engine_port": 9201}}, "uid"
    )
    assert (api, engine) == (9200, 9201)
    # Section wins over the legacy top-level pair.
    api, engine = resolve_ports(
        {
            "api_port": 9100,
            "engine_port": 9101,
            "networking": {"api_port": 9300, "engine_port": 9301},
        },
        "uid",
    )
    assert (api, engine) == (9300, 9301)
    # Half a pair is not a pair: falls back to static derivation.
    api, engine = resolve_ports({"networking": {"api_port": 9400}}, "uid-half")
    assert (api, engine) == derive_static_ports("uid-half")


def test_resolve_ports_sources_are_atomic_no_cross_mixing() -> None:
    """M1: a `networking` section pinning only api_port must NOT pair
    with a legacy top-level engine_port (and vice versa) — a mixed pair
    would silently re-point the disk-cache fingerprint. Each source is
    honored only as a complete pair; otherwise derive."""
    # networking.api_port only + top-level.engine_port only -> derived.
    api, engine = resolve_ports(
        {"engine_port": 9101, "networking": {"api_port": 9400}}, "uid-mix"
    )
    assert (api, engine) == derive_static_ports("uid-mix")
    # Reverse mix: networking.engine_port only + top-level.api_port only.
    api, engine = resolve_ports(
        {"api_port": 9100, "networking": {"engine_port": 9401}}, "uid-mix2"
    )
    assert (api, engine) == derive_static_ports("uid-mix2")
    # A complete networking pair still wins over a complete legacy pair.
    api, engine = resolve_ports(
        {
            "api_port": 9100,
            "engine_port": 9101,
            "networking": {"api_port": 9300, "engine_port": 9301},
        },
        "uid",
    )
    assert (api, engine) == (9300, 9301)
    # No networking section: the complete legacy top-level pair is used.
    api, engine = resolve_ports({"api_port": 9100, "engine_port": 9101}, "uid")
    assert (api, engine) == (9100, 9101)


def test_resolve_ports_falls_back_to_static_derivation() -> None:
    api, engine = resolve_ports({}, "uid-static")
    assert (api, engine) == derive_static_ports("uid-static")
    # The provider's own port never participates in the fingerprint pair.
    assert api != 8081


def test_argv_is_line_buffered_entrypoint_all() -> None:
    assert build_argv("/usr/local/bin/entrypoint.sh") == [
        "stdbuf",
        "-oL",
        "-eL",
        "/usr/local/bin/entrypoint.sh",
        "all",
    ]


def test_effective_capacity_from_kv_slots() -> None:
    assert effective_capacity({"kv_slots": 16}) == 16
    assert effective_capacity({}) == 1
    assert effective_capacity({"kv_slots": "bogus"}) == 1


@pytest.mark.parametrize("key", sorted(ENV_MAP))
def test_map_targets_are_halogen_env_vars(key: str) -> None:
    assert ENV_MAP[key].startswith("HALOGEN_")


# --- Phase 12 sections -> flat options adapter --------------------------


def test_flatten_options_merges_sections_in_order() -> None:
    cfg = {
        "options": {"kv_slots": 1, "ctx": 1024},
        "context": {"ctx": 8192, "kv_slots": 4},
        "sampling": {"temperature": 0.5},
        "networking": {"api_port": 8250, "engine_port": 8251},
    }
    flat = flatten_options(cfg)
    # Section values win over the legacy flat options blob.
    assert flat["ctx"] == 8192
    assert flat["kv_slots"] == 4
    assert flat["temperature"] == 0.5
    # Ports ride into the flat dict but are never ENV_MAP-mapped (the
    # driver reads ports via resolve_ports, which fixes them in build_env).
    assert flat["api_port"] == 8250
    assert "api_port" not in ENV_MAP and "engine_port" not in ENV_MAP


def test_flatten_options_ignores_artifacts_section() -> None:
    flat = flatten_options(
        {"artifacts": {"model": {"path": "/m.hgn"}}, "context": {"ctx": 1}}
    )
    assert "model" not in flat
    assert flat == {"ctx": 1}
    # Every flattened section is declared; artifacts is never among them.
    assert "artifacts" not in CONFIG_SECTIONS


def test_flatten_options_empty_config() -> None:
    assert flatten_options({}) == {}


def test_effective_capacity_from_sectioned_config() -> None:
    assert effective_capacity(flatten_options({"context": {"kv_slots": 16}})) == 16
    assert effective_capacity(flatten_options({"options": {"kv_slots": 8}})) == 8
    assert effective_capacity(flatten_options({})) == 1


def test_build_env_from_flattened_sections() -> None:
    cfg = {
        "context": {"kv_slots": 6, "ctx": 32768},
        "disk_cache": {"prompt_cache": 2, "cache_dir_enabled": True},
        "sampling": {"temperature": 0.6},
        "composable": {"composable_context": 1},
    }
    env = build_env(
        flatten_options(cfg),
        api_port=8200,
        engine_port=8201,
        checkpoint_path="/c.hgn",
        tokenizer_path="/tok",
        cache_dir=None,
        base_env={},
    )
    assert env["HALOGEN_KV_SLOTS"] == "6"
    assert env["HALOGEN_CTX"] == "32768"
    assert env["HALOGEN_PROMPT_CACHE"] == "2"
    assert env["HALOGEN_TEMPERATURE"] == "0.6"
    assert env["HALOGEN_COMPOSABLE_CONTEXT"] == "1"
    # cache_dir_enabled without a cache_dir still emits nothing.
    assert "HALOGEN_CACHE_DIR" not in env


def test_every_wired_schema_field_is_in_env_map() -> None:
    # Cross-check schema x-flag <-> ENV_MAP for every x-supported field
    # outside artifacts (ports/cache dir/npu list are wired by fixed
    # build_env logic, not ENV_MAP).
    wired_env = {
        f["x-flag"]
        for section in SCHEMA["properties"].values()
        for f in section["properties"].values()
        if f.get("x-flag") and f.get("x-supported") is not False
    }
    fixed = {
        "HALOGEN_API_PORT",
        "HALOGEN_PORT",
        "HALOGEN_CACHE_DIR",
        "HALOGEN_NPU_MODELS",
        "HALOGEN_CHECKPOINT",
        "HALOGEN_TOKENIZER",
    }
    assert wired_env - fixed == set(ENV_MAP.values())
    # And nothing in ENV_MAP is marked x-supported: false in the schema.
    for section_name, section in SCHEMA["properties"].items():
        for key, field in section["properties"].items():
            flag = field.get("x-flag")
            if flag in ENV_MAP and section_name != "artifacts":
                assert field.get("x-supported") is not False, f"{section_name}.{key}"
    # validate_backend_config sanity: the shipped schema is usable.
    assert validate_backend_config(SCHEMA, {"context": {"ctx": 128}}) == []
