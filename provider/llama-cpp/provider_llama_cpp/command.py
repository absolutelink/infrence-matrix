"""llama-server command construction, ported from the legacy agent.

Maps a sectioned `backend_config` dict (see provider/README.md and
`provider_llama_cpp/schema.json`, Phase 12) to the argv list for a
`llama.cpp llama-server` invocation. The binary path comes from
`ProviderSettings.LLAMA_SERVER_PATH` (env), never from backend_config.

The flag maps below consume a FLAT args dict. `flatten_args()` produces
it from the sectioned config (every section except `artifacts`, which
the driver resolves separately); for backward compatibility the legacy
flat `cfg["args"]` dict is merged in as well, with section values
winning on key conflicts.

Note on cache flags: `--prompt-cache` is OBSOLETE and must never be
emitted. `--cache-prompt` / `--no-cache-prompt` is a DIFFERENT, valid
modern flag and is ported as-is.
"""

from typing import Any

# backend_config sections whose leaf keys map to CLI flags. `artifacts`
# is excluded (resolved by the driver, not passed to llama-server).
CONFIG_SECTIONS: tuple[str, ...] = (
    "context",
    "kv_cache",
    "loading",
    "gpu",
    "speculative",
    "sampling",
    "reasoning",
    "vision",
    "server",
    "endpoints",
    "logging",
)


def flatten_args(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the sectioned backend_config into a single args dict.

    Section leaves are merged in `CONFIG_SECTIONS` order; the legacy flat
    `cfg["args"]` (pre-Phase-12 shape) is merged first so section values
    take precedence. The legacy top-level `checkpoint_every` /
    `context_size` synonyms ride along via the `kv_cache` / `context`
    sections or `cfg["args"]` (both accepted downstream).
    """
    flat: dict[str, Any] = dict(cfg.get("args") or {})
    for section in CONFIG_SECTIONS:
        values = cfg.get(section)
        if isinstance(values, dict):
            flat.update(values)
    # Canonical key wins when both it and its legacy synonym are present
    # with a usable (non-None) value, so a single CLI flag is emitted;
    # a None canonical falls back to the synonym.
    if flat.get("checkpoint_min_step") is not None:
        flat.pop("checkpoint_every", None)
    if flat.get("cache_idle_slots") is not None:
        flat.pop("no_cache_idle_slots", None)
    return flat


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
    "checkpoint_min_step": "--checkpoint-min-step",
    # Legacy synonym of checkpoint_min_step (pre-Phase-12 key name).
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
    # Canonical positive form (Phase 12 schema).
    "cache_idle_slots": ("--cache-idle-slots", "--no-cache-idle-slots"),
    # Legacy inverted synonym of cache_idle_slots (pre-Phase-12 key name).
    "no_cache_idle_slots": ("--no-cache-idle-slots", "--cache-idle-slots"),
    # NOTE: `strict_mtp_qwen` (legacy --spec-mtp-strict-qwen) is absent in
    # this build — silently ignored; use spec_type/spec args instead.
    # NOTE: `no_mmap` ignored (see above); explicit control via `load_mode`.
}


def _flash_attn_value(value: Any) -> str | None:
    """Normalize the flash_attn config value to 'on'/'off'/'auto'; None = skip.

    Tri-state since Phase 12 (upstream default is 'auto'); booleans and
    truthy strings still map to 'on'/'off' for legacy configs.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
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
    modality: str = "llm",
) -> list[str]:
    """Build the llama-server argv from a sectioned backend_config dict.

    Values are read from the flattened sections (see `flatten_args`);
    `model_path` / `mmproj_path` / `draft_path` are the driver-resolved
    local paths of the `artifacts` section descriptors. `modality` is the
    endpoint kind this backend serves (Phase 18): `"embedding"` auto-adds
    `--embedding`; anything else (default `"llm"`) does not.
    """
    args: dict[str, Any] = flatten_args(cfg)

    # `ctx` and `context_size` are accepted synonyms (canonical: ctx).
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
        # Upstream default is 'auto' (fit to VRAM), not a fixed count.
        str(args.get("gpu_layers", "auto")),
        "--ctx-size",
        str(ctx_size),
        "--batch-size",
        # Aligned with the schema default (upstream llama.cpp default).
        str(args.get("batch_size", 2048)),
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

    # Phase 18: embedding-modality backends boot llama-server in embedding
    # mode. Derived from the driver's modality, never a user config field.
    # --pooling is embedding-only: it selects the pooling type used to build
    # the embedding vector, so it is meaningless (and rejected by some
    # llama-server builds) for chat backends.
    if modality == "embedding":
        cmd.append("--embedding")
        pooling = args.get("pooling")
        if pooling is not None:
            cmd.extend(["--pooling", str(pooling)])

    # Speculative decoding: an explicit draft artifact wins over MTP.
    # The MTP path is selected by the legacy `mtp_draft_max > 0` or an
    # explicit `spec_type: "draft-mtp"`; n-max comes from mtp_draft_max
    # when set, else `spec_draft_n_max`. Other spec_type modes are not
    # yet wired and are ignored here (see schema x-supported notes).
    mtp_draft_max = args.get("mtp_draft_max")
    has_mtp = mtp_draft_max is not None and int(mtp_draft_max) > 0
    explicit_mtp = str(args.get("spec_type") or "") == "draft-mtp"
    if draft_path:
        cmd.extend(["--spec-type", "draft-dflash"])
    elif has_mtp or explicit_mtp:
        cmd.extend(["--spec-type", "draft-mtp"])
        n_max = mtp_draft_max if has_mtp else args.get("spec_draft_n_max")
        if n_max:
            cmd.extend(["--spec-draft-n-max", str(int(n_max))])

    # --jinja applies the model's embedded chat template to inputs. That is
    # correct for chat backends but wrong for embeddings, which embed raw text
    # (the template tokens would pollute the vector). Default ON for llm
    # (unchanged), OFF for embedding; an explicit `jinja` config value wins.
    if args.get("jinja", modality != "embedding"):
        cmd.append("--jinja")

    if mmproj_path:
        cmd.extend(["--mmproj", str(mmproj_path)])

    if draft_path:
        cmd.extend(["-md", str(draft_path)])

    if cmd_load_mode:
        cmd.extend(cmd_load_mode)

    return cmd
