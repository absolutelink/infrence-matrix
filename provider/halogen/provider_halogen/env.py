"""HALOGEN_* environment construction, ported from the legacy agent.

Unlike llama.cpp/gufo (argv-configured), the halogen backend is
configured almost entirely through `HALOGEN_*` environment variables.
`backend_config` holds the semantic config (model artifacts + options);
this module maps it to the env dict for the subprocess.

The fixed wiring (ports, bind, checkpoint, tokenizer) is always emitted;
each semantic option is emitted only when present and not None (None-
skip: the engine keeps its own default).
"""

from typing import Any

# Fixed env always set for a halogen process.
FIXED_ENV: dict[str, str] = {
    "HALOGEN_BIND": "127.0.0.1",
}

# backend_config.options key -> HALOGEN_* env var (value flags).
# Every entry is optional; None / absent is never emitted.
ENV_MAP: dict[str, str] = {
    "drafter": "HALOGEN_DRAFTER",
    "cache_align": "HALOGEN_CACHE_ALIGN",
    "kv_slots": "HALOGEN_KV_SLOTS",
    "slot_ctx": "HALOGEN_SLOT_CTX",
    "cache_mb": "HALOGEN_CACHE_MB",
    "cache_reserve_mb": "HALOGEN_CACHE_RESERVE_MB",
    "max_tokens_cap": "HALOGEN_MAX_TOKENS_CAP",
    "queue_timeout": "HALOGEN_QUEUE_TIMEOUT",
    "w4a4": "HALOGEN_W4A4",
    "w4a4_excl": "HALOGEN_W4A4_EXCL",
    "keepalive_timeout": "HALOGEN_KEEPALIVE_TIMEOUT",
    "sse_keepalive_s": "HALOGEN_SSE_KEEPALIVE_S",
}

DEFAULT_KV_SLOTS = 1


def build_env(
    options: dict[str, Any],
    *,
    api_port: int,
    engine_port: int,
    checkpoint_path: str,
    tokenizer_path: str,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map a halogen backend_config into the subprocess environment.

    Returns the full env dict (copy of ``base_env`` or os.environ with
    the halogen variables layered on). The caller never passes halogen
    config via argv — only the entrypoint word ``all``.
    """
    import os

    env = dict(base_env if base_env is not None else os.environ)
    env["HALOGEN_API_PORT"] = str(api_port)
    env["HALOGEN_PORT"] = str(engine_port)
    env["HALOGEN_BIND"] = FIXED_ENV["HALOGEN_BIND"]
    env["HALOGEN_ENGINE"] = f"127.0.0.1:{engine_port}"
    env["HALOGEN_CHECKPOINT"] = str(checkpoint_path)
    env["HALOGEN_TOKENIZER"] = str(tokenizer_path)
    opts = options or {}
    for key, variable in ENV_MAP.items():
        if key in opts and opts[key] is not None:
            env[variable] = str(opts[key])
    return env


def effective_capacity(options: dict[str, Any]) -> int:
    """Halogen request slots = HALOGEN_KV_SLOTS (options.kv_slots)."""
    opts = options or {}
    try:
        return max(int(opts.get("kv_slots", DEFAULT_KV_SLOTS)), 1)
    except TypeError, ValueError:
        return DEFAULT_KV_SLOTS


def build_argv(binary: str) -> list[str]:
    """Argv for the halogen entrypoint: line-buffered stdio + the `all` word.

    Legacy ran `stdbuf -oL -eL <entrypoint> all` so native diagnostics
    stay visible while stdout is captured for log forwarding.
    """
    return ["stdbuf", "-oL", "-eL", binary, "all"]
