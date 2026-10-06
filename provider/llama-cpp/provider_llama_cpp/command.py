"""llama-server command construction, ported from the legacy agent.

Maps a `backend_config` dict (see provider/README.md) to the argv list
for a `llama.cpp llama-server` invocation. The binary path comes from
`ProviderSettings.LLAMA_SERVER_PATH` (env), never from backend_config.

Note on cache flags: `--prompt-cache` is OBSOLETE and must never be
emitted. `--cache-prompt` / `--no-cache-prompt` is a DIFFERENT, valid
modern flag and is ported as-is.
"""

from typing import Any

# Config key -> CLI flag for options that take a value.
VALUE_FLAGS: dict[str, str] = {
    "threads": "--threads",
    "threads_batch": "--threads-batch",
    "ubatch_size": "--ubatch-size",
    "keep": "--keep",
    "predict": "--predict",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "cache_reuse": "--cache-reuse",
    "ctx_checkpoints": "--ctx-checkpoints",
    "checkpoint_every": "--checkpoint-min-step",
    "cache_ram": "--cache-ram",
    "slot_save_path": "--slot-save-path",
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
}

# Config key -> flag (bool-only) or (true_flag, false_flag) tuple.
#
# NOTE on the tuple shapes: modern llama.cpp defaults these features ON and
# only documents the `--no-…` opt-out (e.g. `--warmup, --no-warmup`). The
# positive twin is accepted by this build, but the negative form is what
# actually changes behavior; the map below keeps both to stay compatible
# with either build.
#
# REMOVED flags (verified against the deployed llama-server --help):
#   --mmap / --no-mmap  — mmap is no longer a flag; loading modes live in
#     `--load-mode` (auto|mmap|mlock|none). `no_mmap` is therefore
#     ignored (mmap stays auto). Use options.load_mode for explicit control.
BOOLEAN_FLAGS: dict[str, Any] = {
    "swa_full": "--swa-full",
    "kv_offload": ("--kv-offload", "--no-kv-offload"),
    # Valid modern flag; NOT the obsolete `--prompt-cache`.
    "cache_prompt": ("--cache-prompt", "--no-cache-prompt"),
    "cont_batching": ("--cont-batching", "--no-cont-batching"),
    "warmup": ("--warmup", "--no-warmup"),
    "context_shift": ("--context-shift", "--no-context-shift"),
    "kv_unified": "--kv-unified",
    "no_cache_idle_slots": ("--no-cache-idle-slots", "--cache-idle-slots"),
    # NOTE: `strict_mtp_qwen` (legacy --spec-mtp-strict-qwen) is absent in
    # this build — silently ignored; use spec_type/spec args instead.
    # NOTE: `no_mmap` ignored (see above); explicit control via `load_mode`.
}


def _flash_attn_value(value: Any) -> str | None:
    """Normalize the flash_attn config value to 'on'/'off'; None = skip."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"on", "true", "1", "yes"}:
        return "on"
    if text in {"off", "false", "0", "no"}:
        return "off"
    return None


def build_llama_command(
    cfg: dict[str, Any],
    *,
    model_path: str,
    port: int,
    binary: str = "llama-server",
    mmproj_path: str | None = None,
    draft_path: str | None = None,
) -> list[str]:
    """Build the llama-server argv from a backend_config dict."""
    args: dict[str, Any] = dict(cfg.get("args") or {})

    # `ctx` and `context_size` are accepted synonyms.
    ctx_size = args.get("ctx", args.get("context_size", 4096))

    # Explicit loading mode (auto|mmap|mlock|none); replaces the removed
    # --mmap/--no-mmap flags. auto is the server default.
    load_mode = args.get("load_mode")
    if load_mode is not None and load_mode != "auto":
        cmd_load_mode = ["--load-mode", str(load_mode)]
    else:
        cmd_load_mode = []

    cmd: list[str] = [
        binary,
        "--model",
        model_path,
        "--port",
        str(port),
        "--n-gpu-layers",
        str(args.get("gpu_layers", 35)),
        "--ctx-size",
        str(ctx_size),
        "--batch-size",
        str(args.get("batch_size", 512)),
        "--metrics",
    ]

    for name, flag in VALUE_FLAGS.items():
        if name in args and args[name] is not None:
            cmd.extend([flag, str(args[name])])

    for name, flags in BOOLEAN_FLAGS.items():
        if name not in args or args[name] is None:
            continue
        if isinstance(flags, tuple):
            cmd.append(flags[0] if args[name] else flags[1])
        elif args[name]:
            cmd.append(flags)

    # --flash-attn takes a VALUE (on|off); never the obsolete --prompt-cache.
    flash = _flash_attn_value(args.get("flash_attn"))
    if flash is not None:
        cmd.extend(["--flash-attn", flash])

    # Speculative decoding: an explicit draft model wins over MTP.
    mtp_draft_max = args.get("mtp_draft_max")
    has_mtp = mtp_draft_max is not None and int(mtp_draft_max) > 0
    if draft_path or has_mtp:
        cmd.extend(["--spec-type", "draft-dflash" if draft_path else "draft-mtp"])
        if has_mtp and not draft_path:
            cmd.extend(["--spec-draft-n-max", str(int(mtp_draft_max))])

    if args.get("jinja", True):
        cmd.append("--jinja")

    if mmproj_path:
        cmd.extend(["--mmproj", str(mmproj_path)])

    if draft_path:
        cmd.extend(["-md", str(draft_path)])

    if cmd_load_mode:
        cmd.extend(cmd_load_mode)

    return cmd
