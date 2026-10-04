"""Unit tests for the halogen-flash HALOGEN_* env map + static ports."""

import pytest

from provider_halogen_flash.env import (
    ENV_MAP,
    FLASH_PORT_END,
    FLASH_PORT_START,
    build_argv,
    build_env,
    derive_static_ports,
    effective_capacity,
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
    # 43 semantic HALOGEN_* vars ported from the legacy manager.
    assert len(ENV_MAP) == 43
    assert ENV_MAP["kv_slots"] == "HALOGEN_KV_SLOTS"
    assert ENV_MAP["composable_context_bytes"] == "HALOGEN_COMPOSABLE_CONTEXT_BYTES"
    assert ENV_MAP["vision_max_pixels"] == "HALOGEN_VISION_MAX_PIXELS"


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
