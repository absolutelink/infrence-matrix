"""Unit tests for the gufo argv mapping (ported from legacy)."""

from pathlib import Path

from provider_gufo.command import (
    VALUE_FLAGS,
    build_command,
    effective_capacity,
)


def test_base_command_always_includes_serve_llm_host_port_model() -> None:
    cmd = build_command({}, model_path="/models/m.gguf", port=8123)
    assert cmd == [
        "gufo",
        "serve",
        "llm",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--model",
        "/models/m.gguf",
    ]


def test_value_flags_are_mapped() -> None:
    options = {key: f"v{i}" for i, key in enumerate(VALUE_FLAGS)}
    cmd = build_command(options, model_path="/m.gguf", port=1)
    joined = " ".join(cmd)
    for key, flag in VALUE_FLAGS.items():
        assert f"{flag} {options[key]}" in joined


def test_bool_flags_only_emitted_when_true() -> None:
    cmd = build_command(
        {"verbose": True, "log_progress": True}, model_path="/m.gguf", port=1
    )
    assert "--verbose" in cmd
    assert "--log-progress" in cmd
    # False booleans are omitted so gufo keeps its own default.
    cmd = build_command({"verbose": False}, model_path="/m.gguf", port=1)
    assert "--verbose" not in cmd
    cmd = build_command({"log_progress": None}, model_path="/m.gguf", port=1)
    assert "--log-progress" not in cmd


def test_none_values_never_emitted() -> None:
    cmd = build_command(
        {"context": None, "temperature": None}, model_path="/m.gguf", port=1
    )
    joined = " ".join(cmd)
    assert "--context" not in joined
    assert "--temperature" not in joined
    assert "None" not in joined


def test_binary_override() -> None:
    cmd = build_command({}, model_path="/m.gguf", port=1, binary="/opt/gufo")
    assert cmd[0] == "/opt/gufo"


def test_disk_cache_enabled_creates_per_instance_dir(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cmd = build_command(
        {"cache_disk": True, "cache_disk_bytes": 1024},
        model_path="/m.gguf",
        port=1,
        cache_dir=cache,
        instance_id="srv-42",
    )
    joined = " ".join(cmd)
    assert f"--cache-disk {cache / 'srv-42'}" in joined
    assert "--cache-disk-bytes 1024" in joined
    assert (cache / "srv-42").is_dir()


def test_disk_cache_disabled_omits_flag(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for value in (False, None, "default"):
        cmd = build_command(
            {"cache_disk": value},
            model_path="/m.gguf",
            port=1,
            cache_dir=cache,
            instance_id="srv-off",
        )
        assert "--cache-disk" not in cmd
    assert not (cache / "srv-off").exists()


def test_served_model_name_mapping() -> None:
    cmd = build_command({"served_model_name": "my-alias"}, model_path="/m.gguf", port=1)
    assert "--served-model-name my-alias" in " ".join(cmd)


def test_aux_spec_paths_mapped() -> None:
    cmd = build_command(
        {
            "mmproj": "/models/vision.gguf",
            "dflash_model": "/models/dflash.gguf",
            "dspark_model": "/models/dspark.gguf",
            "mtp_model": "/models/mtp.gguf",
        },
        model_path="/m.gguf",
        port=1,
    )
    joined = " ".join(cmd)
    assert "--mmproj /models/vision.gguf" in joined
    assert "--dflash-model /models/dflash.gguf" in joined
    assert "--dspark-model /models/dspark.gguf" in joined
    assert "--mtp-model /models/mtp.gguf" in joined


def test_effective_capacity_uses_sessions() -> None:
    assert effective_capacity({"sessions": 3}) == 3
    assert effective_capacity({}) == 1
    assert effective_capacity({"sessions": "4"}) == 4
    assert effective_capacity({"sessions": 0}) == 1
    assert effective_capacity({"sessions": "bogus"}) == 1
