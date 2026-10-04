"""Unit tests for the HALOGEN_* env map (ported from legacy)."""

import pytest

from provider_halogen.env import ENV_MAP, build_argv, build_env, effective_capacity


def _env(**opts):
    return build_env(
        opts,
        api_port=8191,
        engine_port=8192,
        checkpoint_path="/models/ckpt.hgn",
        tokenizer_path="/models/tokenizer",
        base_env={},
    )


def test_fixed_wiring_always_set() -> None:
    env = _env()
    assert env["HALOGEN_API_PORT"] == "8191"
    assert env["HALOGEN_PORT"] == "8192"
    assert env["HALOGEN_BIND"] == "127.0.0.1"
    assert env["HALOGEN_ENGINE"] == "127.0.0.1:8192"
    assert env["HALOGEN_CHECKPOINT"] == "/models/ckpt.hgn"
    assert env["HALOGEN_TOKENIZER"] == "/models/tokenizer"


def test_every_option_key_maps_to_its_env_var() -> None:
    opts = {key: f"v{i}" for i, key in enumerate(ENV_MAP)}
    env = _env(**opts)
    for key, var in ENV_MAP.items():
        assert env[var] == str(opts[key])


def test_none_values_never_emitted() -> None:
    env = _env(kv_slots=None, drafter=None)
    assert "HALOGEN_KV_SLOTS" not in env
    assert "HALOGEN_DRAFTER" not in env
    assert "None" not in " ".join(env.values())


def test_absent_options_omitted() -> None:
    env = _env()
    for var in ENV_MAP.values():
        assert var not in env


def test_values_stringified() -> None:
    env = _env(kv_slots=4, cache_mb=1024, w4a4=True, sse_keepalive_s=15.5)
    assert env["HALOGEN_KV_SLOTS"] == "4"
    assert env["HALOGEN_CACHE_MB"] == "1024"
    assert env["HALOGEN_W4A4"] == "True"
    assert env["HALOGEN_SSE_KEEPALIVE_S"] == "15.5"


def test_base_env_is_copied_not_mutated() -> None:
    base = {"PATH": "/usr/bin"}
    env = build_env(
        {"kv_slots": 2},
        api_port=1,
        engine_port=2,
        checkpoint_path="/c",
        tokenizer_path="/t",
        base_env=base,
    )
    assert base == {"PATH": "/usr/bin"}
    assert env["PATH"] == "/usr/bin"
    assert env["HALOGEN_KV_SLOTS"] == "2"


def test_argv_is_line_buffered_entrypoint_all() -> None:
    assert build_argv("/usr/local/bin/entrypoint.sh") == [
        "stdbuf",
        "-oL",
        "-eL",
        "/usr/local/bin/entrypoint.sh",
        "all",
    ]


def test_effective_capacity_from_kv_slots() -> None:
    assert effective_capacity({"kv_slots": 8}) == 8
    assert effective_capacity({}) == 1
    assert effective_capacity({"kv_slots": "3"}) == 3
    assert effective_capacity({"kv_slots": 0}) == 1
    assert effective_capacity({"kv_slots": "junk"}) == 1


def test_unknown_option_keys_ignored() -> None:
    env = _env(**{"not_a_halogen_option": 1})
    assert "HALOGEN_NOT_A_HALOGEN_OPTION" not in env


@pytest.mark.parametrize("key", sorted(ENV_MAP))
def test_map_has_no_none_targets(key: str) -> None:
    assert ENV_MAP[key].startswith("HALOGEN_")
