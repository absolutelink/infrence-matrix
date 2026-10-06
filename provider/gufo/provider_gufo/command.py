"""``gufo serve llm`` command construction, ported from the legacy agent.

Maps a sectioned `backend_config` dict (see provider/README.md and
`provider_gufo/schema.json`, Phase 12) to the argv list for a
``gufo serve llm`` invocation. The binary path comes from
`ProviderSettings.GUFO_SERVER_PATH` (env), never from backend_config.

Unlike llama.cpp, gufo serves multiple models on one instance and takes
its per-model knobs (mmproj, dflash, dspark, mtp) as *paths* in the
flattened options; those live in the `artifacts` section since Phase 12
and are resolved through `provider_lib.downloader` before the process
starts and rewritten into the option values here.

The flag maps below consume a FLAT options dict. `flatten_args()`
produces it from the sectioned config (every section except `artifacts`,
which the driver resolves and injects); for backward compatibility the
legacy flat `cfg["options"]` dict is merged in as well, with section
values winning on key conflicts.
"""

from pathlib import Path
from typing import Any

# backend_config sections whose leaf keys map to CLI flags. `artifacts`
# is excluded (resolved by the driver; the resolved aux paths are
# injected into the flattened options explicitly).
CONFIG_SECTIONS: tuple[str, ...] = (
    "context",
    "disk_cache",
    "speculative",
    "sampling",
    "reasoning",
    "limits",
    "server",
)


def flatten_args(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the sectioned backend_config into a single options dict.

    Mirrors llama-cpp's `flatten_args()`: section leaves are merged in
    `CONFIG_SECTIONS` order; the legacy flat `cfg["options"]`
    (pre-Phase-12 shape) is merged first so section values take
    precedence on key conflicts.
    """
    flat: dict[str, Any] = dict(cfg.get("options") or {})
    for section in CONFIG_SECTIONS:
        values = cfg.get(section)
        if isinstance(values, dict):
            flat.update(values)
    return flat


# Config key (flattened options.*) -> CLI flag for options that take
# a value. Every flag is optional; omitting it lets gufo apply its own
# default. None values are skipped (never emitted as the string "None").
# The aux spec keys (mmproj / dflash_model / dspark_model / mtp_model)
# carry the driver-resolved LOCAL PATHS of the `artifacts` descriptors.
VALUE_FLAGS: dict[str, str] = {
    # Model & context
    "context": "--context",
    "served_model_name": "--served-model-name",
    "mmproj": "--mmproj",
    # Sampling defaults
    "max_tokens": "--max-tokens",
    "temperature": "--temperature",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "min_p": "--min-p",
    "min_keep": "--min-keep",
    "seed": "--seed",
    "repeat_penalty": "--repeat-penalty",
    "repeat_last_n": "--repeat-last-n",
    "frequency_penalty": "--frequency-penalty",
    "presence_penalty": "--presence-penalty",
    # Reasoning defaults (tri-state / enum values)
    "think": "--think",
    "reasoning_effort": "--reasoning-effort",
    "preserve_thinking": "--preserve-thinking",
    # Speculative decoding
    "speculative": "--speculative",
    "dflash_model": "--dflash-model",
    "dspark_model": "--dspark-model",
    "mtp_model": "--mtp-model",
    "draft_policy": "--draft-policy",
    "draft_tokens": "--draft-tokens",
    "min_draft_tokens": "--min-draft-tokens",
    # Scheduling and server-protection limits
    "prefill_chunk": "--prefill-chunk",
    "max_pending": "--max-pending",
    "max_pending_per_client": "--max-pending-per-client",
    "request_timeout_ms": "--request-timeout-ms",
    "max_output_bytes": "--max-output-bytes",
    "max_buffered_output_bytes": "--max-buffered-output-bytes",
    "max_buffered_output_total": "--max-buffered-output-total",
    # Disk cache
    "cache_disk_bytes": "--cache-disk-bytes",
    "cache_disk_staging_bytes": "--cache-disk-staging-bytes",
    # Server options
    "sessions": "--sessions",
    "max_connections": "--max-connections",
    "max_request_bytes": "--max-request-bytes",
    "api_key": "--api-key",
}

# Config key -> boolean-only flag. False/None are omitted so gufo keeps
# its own default (gufo has no --no-x twins for these).
BOOL_FLAGS: dict[str, str] = {
    "verbose": "--verbose",
    "log_progress": "--log-progress",
}

DEFAULT_SESSIONS = 1


def build_command(
    options: dict[str, Any],
    *,
    model_path: str,
    port: int,
    binary: str = "gufo",
    cache_dir: Path | None = None,
    instance_id: str = "default",
) -> list[str]:
    """Assemble the ``gufo serve llm`` argv from backend_config.options.

    ``cache_dir`` is the provider's CACHE_DIR: when ``options.cache_disk``
    is True, a per-instance subdirectory ``<cache_dir>/<instance_id>`` is
    created and passed via ``--cache-disk``. The driver passes the REAL
    admin-issued registration id (MACHINE_UID only pre-registration), so
    the disk cache never shares a directory across instances.
    """
    cmd: list[str] = [
        binary,
        "serve",
        "llm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model_path,
    ]
    opts = options or {}
    for key, flag in VALUE_FLAGS.items():
        value = opts.get(key)
        if value is not None:
            cmd.extend([flag, str(value)])
    if opts.get("cache_disk") is True and cache_dir is not None:
        inst_cache = Path(cache_dir) / instance_id
        inst_cache.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--cache-disk", str(inst_cache)])
    for key, flag in BOOL_FLAGS.items():
        if opts.get(key) is True:
            cmd.append(flag)
    return cmd


def effective_capacity(options: dict[str, Any]) -> int:
    """Gufo concurrency is the number of preallocated GPU sessions."""
    opts = options or {}
    try:
        return max(int(opts.get("sessions", DEFAULT_SESSIONS)), 1)
    except (TypeError, ValueError):  # fmt: skip
        return DEFAULT_SESSIONS
