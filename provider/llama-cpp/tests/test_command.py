"""Flag-mapping unit tests for build_llama_command (Phase 12 sectioned input).

Inputs use the sectioned backend_config shape (schema.json); the legacy
flat `{"args": {...}}` form is also accepted by flatten_args and covered
for backward compatibility.
"""

from typing import Any

from provider_llama_cpp.command import build_llama_command, flatten_args


def _flags(cmd: list[str]) -> set[str]:
    return {a for a in cmd if a.startswith("--")}


def test_base_flags() -> None:
    cmd = build_llama_command(
        {}, model_path="/m/x.gguf", port=9999, binary="llama-server"
    )
    assert cmd[:3] == ["llama-server", "--model", "/m/x.gguf"]
    assert cmd[3:5] == ["--port", "9999"]
    assert "--n-gpu-layers" in cmd
    assert "--ctx-size" in cmd
    assert "--batch-size" in cmd
    assert "--metrics" in cmd
    # Defaults: gpu_layers is now upstream 'auto' (was legacy 35);
    # batch_size fallback aligns with the schema default (2048).
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "auto"
    assert cmd[cmd.index("--ctx-size") + 1] == "4096"
    assert cmd[cmd.index("--batch-size") + 1] == "2048"


def test_ctx_section_and_context_size_synonym() -> None:
    for key in ("ctx", "context_size"):
        cmd = build_llama_command(
            {"context": {key: 8192}}, model_path="/m/x.gguf", port=1
        )
        assert cmd[cmd.index("--ctx-size") + 1] == "8192", key
        assert "--ctx" not in _flags(cmd)


def test_ctx_precedence_over_context_size() -> None:
    cmd = build_llama_command(
        {"context": {"ctx": 2048, "context_size": 4096}},
        model_path="/m/x.gguf",
        port=1,
    )
    assert cmd[cmd.index("--ctx-size") + 1] == "2048"


def test_value_flags_from_sections() -> None:
    cfg: dict[str, Any] = {
        "context": {"keep": 512, "predict": 256},
        "kv_cache": {
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "cache_reuse": 4,
            "ctx_checkpoints": 64,
            "checkpoint_min_step": 16,
            "cache_ram": 2048,
            "slot_save_path": "/tmp/slots",
        },
        "gpu": {
            "gpu_layers": 99,
            "device": "cuda0",
            "split_mode": "layer",
            "tensor_split": "1,2",
            "main_gpu": 0,
            "fit": "on",
            "fit_target": "256,512",
            "fit_ctx": 4096,
        },
        "speculative": {"spec_draft_p_min": 0.1},
        "sampling": {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": 42,
        },
        "reasoning": {"reasoning": "on", "reasoning_budget": 1024},
        "server": {
            "threads": 8,
            "threads_batch": 4,
            "ubatch_size": 128,
            "parallel": 2,
        },
    }
    cmd = build_llama_command(cfg, model_path="/m/x.gguf", port=1)
    for key, flag in {
        "threads": "--threads",
        "threads_batch": "--threads-batch",
        "ubatch_size": "--ubatch-size",
        "keep": "--keep",
        "predict": "--predict",
        "cache_type_k": "--cache-type-k",
        "cache_type_v": "--cache-type-v",
        "cache_reuse": "--cache-reuse",
        "ctx_checkpoints": "--ctx-checkpoints",
        "checkpoint_min_step": "--checkpoint-min-step",
        "cache_ram": "--cache-ram",
        "slot_save_path": "--slot-save-path",
        "gpu_layers": "--n-gpu-layers",
        "device": "--device",
        "split_mode": "--split-mode",
        "tensor_split": "--tensor-split",
        "main_gpu": "--main-gpu",
        "fit": "--fit",
        "fit_target": "--fit-target",
        "fit_ctx": "--fit-ctx",
        "temperature": "--temperature",
        "top_k": "--top-k",
        "top_p": "--top-p",
        "min_p": "--min-p",
        "repeat_penalty": "--repeat-penalty",
        "presence_penalty": "--presence-penalty",
        "frequency_penalty": "--frequency-penalty",
        "seed": "--seed",
        "parallel": "--parallel",
        "reasoning": "--reasoning",
        "reasoning_budget": "--reasoning-budget",
        "spec_draft_p_min": "--spec-draft-p-min",
    }.items():
        assert flag in cmd, key
        assert cmd[cmd.index(flag) + 1] == str(flatten_args(cfg)[key]), key


def test_legacy_flat_args_still_supported() -> None:
    cmd = build_llama_command(
        {"args": {"ctx": 8192, "gpu_layers": 20, "threads": 4}},
        model_path="/m/x.gguf",
        port=1,
    )
    assert cmd[cmd.index("--ctx-size") + 1] == "8192"
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "20"
    assert cmd[cmd.index("--threads") + 1] == "4"


def test_section_wins_over_legacy_flat_args() -> None:
    cmd = build_llama_command(
        {"args": {"ctx": 1024}, "context": {"ctx": 8192}},
        model_path="/m/x.gguf",
        port=1,
    )
    assert cmd[cmd.index("--ctx-size") + 1] == "8192"


def test_none_value_flags_skipped() -> None:
    cmd = build_llama_command(
        {"server": {"threads": None}, "sampling": {"seed": None}},
        model_path="/m/x.gguf",
        port=1,
    )
    assert "--threads" not in cmd
    assert "--seed" not in cmd


def test_boolean_flag_tuples() -> None:
    on = build_llama_command(
        {
            "kv_cache": {
                "kv_offload": True,
                "cache_prompt": True,
                "cache_idle_slots": True,
            },
            "server": {
                "cont_batching": True,
                "warmup": True,
                "context_shift": True,
            },
        },
        model_path="/m/x.gguf",
        port=1,
    )
    off = build_llama_command(
        {
            "kv_cache": {
                "kv_offload": False,
                "cache_prompt": False,
                "cache_idle_slots": False,
            },
            "server": {
                "cont_batching": False,
                "warmup": False,
                "context_shift": False,
            },
        },
        model_path="/m/x.gguf",
        port=1,
    )
    assert "--kv-offload" in on and "--no-kv-offload" not in on
    assert "--no-kv-offload" in off and "--kv-offload" not in off
    assert "--cache-prompt" in on and "--no-cache-prompt" not in on
    assert "--no-cache-prompt" in off and "--cache-prompt" not in off
    assert "--cont-batching" in on and "--no-cont-batching" in off
    assert "--warmup" in on and "--no-warmup" in off
    assert "--context-shift" in on and "--no-context-shift" in off
    # Canonical positive form: cache_idle_slots true → --cache-idle-slots.
    assert "--cache-idle-slots" in on and "--no-cache-idle-slots" not in on
    assert "--no-cache-idle-slots" in off and "--cache-idle-slots" not in off
    # `no_mmap` is REMOVED in modern llama.cpp (loading modes live in
    # --load-mode); no form of it may ever be emitted.
    assert "--no-mmap" not in on and "--mmap" not in on
    assert "--no-mmap" not in off and "--mmap" not in off


def test_legacy_no_cache_idle_slots_inverted_synonym() -> None:
    on = build_llama_command(
        {"args": {"no_cache_idle_slots": True}}, model_path="/m/x.gguf", port=1
    )
    off = build_llama_command(
        {"args": {"no_cache_idle_slots": False}}, model_path="/m/x.gguf", port=1
    )
    assert "--no-cache-idle-slots" in on and "--cache-idle-slots" not in on
    assert "--cache-idle-slots" in off and "--no-cache-idle-slots" not in off


def test_canonical_key_wins_over_legacy_synonyms() -> None:
    # checkpoint_min_step canonical: one --checkpoint-min-step only.
    cmd = build_llama_command(
        {
            "args": {"checkpoint_every": 8},
            "kv_cache": {"checkpoint_min_step": 16, "cache_idle_slots": False},
        },
        model_path="/m/x.gguf",
        port=1,
    )
    assert cmd.count("--checkpoint-min-step") == 1
    assert cmd[cmd.index("--checkpoint-min-step") + 1] == "16"
    assert "--no-cache-idle-slots" in cmd and "--cache-idle-slots" not in cmd


def test_canonical_none_falls_back_to_legacy_synonym() -> None:
    # A None canonical value must NOT drop the legacy synonym's value.
    cmd = build_llama_command(
        {
            "args": {"checkpoint_every": 8},
            "kv_cache": {"checkpoint_min_step": None, "no_cache_idle_slots": True},
        },
        model_path="/m/x.gguf",
        port=1,
    )
    assert cmd.count("--checkpoint-min-step") == 1
    assert cmd[cmd.index("--checkpoint-min-step") + 1] == "8"
    # cache_idle_slots None → legacy inverted synonym honored.
    assert "--no-cache-idle-slots" in cmd
    assert "--cache-idle-slots" not in cmd


def test_canonical_none_falls_back_within_same_section() -> None:
    flat = flatten_args(
        {"kv_cache": {"checkpoint_min_step": None, "checkpoint_every": 12}}
    )
    assert flat.get("checkpoint_min_step") is None
    assert flat.get("checkpoint_every") == 12
    cmd = build_llama_command(
        {"kv_cache": {"cache_idle_slots": None, "no_cache_idle_slots": False}},
        model_path="/m/x.gguf",
        port=1,
    )
    # no_cache_idle_slots False → positive flag.
    assert "--cache-idle-slots" in cmd
    assert "--no-cache-idle-slots" not in cmd


def test_bool_only_flags_emitted_only_when_true() -> None:
    on = build_llama_command(
        {"kv_cache": {"swa_full": True, "kv_unified": True}},
        model_path="/m/x.gguf",
        port=1,
    )
    off = build_llama_command(
        {"kv_cache": {"swa_full": False, "kv_unified": False}},
        model_path="/m/x.gguf",
        port=1,
    )
    assert "--swa-full" in on
    assert "--kv-unified" in on
    assert "--swa-full" not in off
    assert "--kv-unified" not in off


def test_flash_attn_value_flag_bool_string_and_auto() -> None:
    for value, expected in (
        (True, "on"),
        (False, "off"),
        ("on", "on"),
        ("off", "off"),
        ("ON", "on"),
        ("false", "off"),
        ("auto", "auto"),
    ):
        cmd = build_llama_command(
            {"server": {"flash_attn": value}}, model_path="/m/x.gguf", port=1
        )
        assert "--flash-attn" in cmd, value
        assert cmd[cmd.index("--flash-attn") + 1] == expected, value


def test_flash_attn_none_skipped() -> None:
    cmd = build_llama_command(
        {"server": {"flash_attn": None}}, model_path="/m/x.gguf", port=1
    )
    assert "--flash-attn" not in cmd
    cmd = build_llama_command({}, model_path="/m/x.gguf", port=1)
    assert "--flash-attn" not in cmd


def test_prompt_cache_never_emitted() -> None:
    # The obsolete llama.cpp flag must NEVER appear, whatever the config says.
    for args in (
        {},
        {"cache_prompt": True},
        {"cache_prompt": False},
        {"prompt_cache": True},
        {"prompt_cache": False},
        {"flash_attn": "on"},
    ):
        cmd = build_llama_command({"args": args}, model_path="/m/x.gguf", port=1)
        assert "--prompt-cache" not in cmd
        assert "--prompt-cache" not in " ".join(cmd)


def test_draft_model_selects_dflash_spec() -> None:
    cmd = build_llama_command(
        {"speculative": {"mtp_draft_max": 4}},
        model_path="/m/x.gguf",
        port=1,
        draft_path="/m/draft.gguf",
    )
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dflash"
    assert "-md" in cmd
    assert cmd[cmd.index("-md") + 1] == "/m/draft.gguf"
    # dflash path does not add the MTP n-max flag.
    assert "--spec-draft-n-max" not in cmd


def test_mtp_draft_max_selects_mtp_spec() -> None:
    cmd = build_llama_command(
        {"speculative": {"mtp_draft_max": 4}}, model_path="/m/x.gguf", port=1
    )
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "4"
    assert "-md" not in cmd


def test_explicit_spec_type_mtp_uses_spec_draft_n_max() -> None:
    cmd = build_llama_command(
        {"speculative": {"spec_type": "draft-mtp", "spec_draft_n_max": 5}},
        model_path="/m/x.gguf",
        port=1,
    )
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "5"


def test_explicit_spec_type_mtp_zero_n_max_not_emitted() -> None:
    cmd = build_llama_command(
        {"speculative": {"spec_type": "draft-mtp", "spec_draft_n_max": 0}},
        model_path="/m/x.gguf",
        port=1,
    )
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert "--spec-draft-n-max" not in cmd


def test_mtp_zero_or_none_no_spec_flags() -> None:
    for value in (0, None):
        cmd = build_llama_command(
            {"speculative": {"mtp_draft_max": value}}, model_path="/m/x.gguf", port=1
        )
        assert "--spec-type" not in cmd
        assert "--spec-draft-n-max" not in cmd


def test_load_mode_explicit_value() -> None:
    """`--mmap` was removed in modern llama.cpp; loading modes live in
    `--load-mode` (auto|mmap|mlock|none). Explicit non-auto values emit the
    flag; auto/absent emits nothing (server default)."""
    cmd = build_llama_command(
        {"loading": {"load_mode": "mlock"}}, model_path="/m/x.gguf", port=1
    )
    assert "--load-mode" in cmd
    assert cmd[cmd.index("--load-mode") + 1] == "mlock"
    # auto or absent: no flag (server default is auto).
    assert "--load-mode" not in build_llama_command(
        {"loading": {"load_mode": "auto"}}, model_path="/m/x.gguf", port=1
    )
    assert "--load-mode" not in build_llama_command({}, model_path="/m/x.gguf", port=1)


def test_removed_flags_never_emitted() -> None:
    """Regression guard for flags this deployed llama-server no longer has:
    --mmap/--no-mmap (→ --load-mode) and --spec-mtp-strict-qwen."""
    cmd = build_llama_command(
        {
            "args": {
                "no_mmap": True,
                "strict_mtp_qwen": True,
                "mtp_draft_max": 4,
            }
        },
        model_path="/m/x.gguf",
        port=1,
    )
    for flag in ("--mmap", "--no-mmap", "--spec-mtp-strict-qwen"):
        assert flag not in cmd
    # mtp still selects its spec mode; strict_mtp_qwen no longer coerces
    # parallel (the flag it existed for is gone).
    assert "--spec-type" in cmd


def test_jinja_default_true() -> None:
    assert "--jinja" in build_llama_command({}, model_path="/m/x.gguf", port=1)
    assert "--jinja" not in build_llama_command(
        {"reasoning": {"jinja": False}}, model_path="/m/x.gguf", port=1
    )


def test_mmproj_flag() -> None:
    cmd = build_llama_command(
        {}, model_path="/m/x.gguf", port=1, mmproj_path="/m/mm.gguf"
    )
    assert cmd[cmd.index("--mmproj") + 1] == "/m/mm.gguf"
