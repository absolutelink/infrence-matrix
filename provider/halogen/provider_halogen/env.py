"""HALOGEN_* environment construction, ported from the legacy agent.

Unlike llama.cpp/gufo (argv-configured), the halogen backend is
configured almost entirely through `HALOGEN_*` environment variables.
Since Phase 12 the `backend_config` is a **sectioned** object (see
`provider_halogen/schema.json`); `flatten_options()` assembles the flat
semantic-options dict that `ENV_MAP` / `build_env` consume, so the env
mapping itself is unchanged. The legacy flat `cfg["options"]` blob is
still merged (section values win on key conflicts) for hand-rolled
configs. None-skip semantics hold: absent or None values are never
emitted; the engine keeps its own default.

Port rule: halogen exposes an API port and a private engine port. When
not pinned they default to `PROVIDER_PORT + 1` / `PROVIDER_PORT + 2`
from the container env. Resolution is ATOMIC (Phase 12 D3 lesson): the
pair is honored only when BOTH ports come from the same source, so a
half-migrated config can never mix a `networking` api_port with a
legacy top-level engine_port.
"""

from typing import Any

# Fixed env always set for a halogen process.
FIXED_ENV: dict[str, str] = {
    "HALOGEN_BIND": "127.0.0.1",
}

# backend_config semantic-option key -> HALOGEN_* env var (value flags).
# Every entry is optional; None / absent is never emitted. Keys arrive
# via `flatten_options()` (Phase 12 sections) or the legacy flat
# `cfg["options"]` blob.
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

# backend_config sections whose leaf keys flatten into the semantic
# `options` dict consumed by ENV_MAP / build_env (Phase 12). `artifacts`
# is excluded: the driver resolves those descriptors to local paths and
# passes them to build_env explicitly. `networking` ports ride along
# harmlessly — ENV_MAP has no key for them; the driver reads ports via
# resolve_ports().
CONFIG_SECTIONS: tuple[str, ...] = (
    "cache",
    "concurrency",
    "quantization",
    "speculative",
    "server",
    "networking",
)

DEFAULT_KV_SLOTS = 1


def flatten_options(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assemble the flat semantic-options dict from a sectioned config.

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


def resolve_ports(cfg: dict[str, Any], provider_port: int) -> tuple[int, int]:
    """Resolve the (api_port, engine_port) pair for this instance.

    Each source is ATOMIC — a pair is honored only when both ports come
    from the same source, so a half-migrated config can never mix a
    `networking` api_port with a legacy top-level engine_port.
    Resolution order:

    1. `networking` section with BOTH `api_port` and `engine_port`.
    2. Legacy top-level `api_port` (or its `backend_port` alias) AND
       `engine_port` (both present).
    3. The default pair derived from the container's PROVIDER_PORT
       (api = PROVIDER_PORT + 1, engine = PROVIDER_PORT + 2).
    """
    networking = cfg.get("networking")
    if isinstance(networking, dict):
        api = networking.get("api_port")
        engine = networking.get("engine_port")
        if api is not None and engine is not None:
            return int(api), int(engine)
    api = cfg.get("api_port")
    if api is None:
        api = cfg.get("backend_port")
    engine = cfg.get("engine_port")
    if api is not None and engine is not None:
        return int(api), int(engine)
    return provider_port + 1, provider_port + 2


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
    config via argv — only the entrypoint word ``all``. ``options`` is
    the flat semantic dict — pass ``flatten_options(cfg)`` for a
    Phase 12 sectioned config (or a legacy ``cfg["options"]`` blob).
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
    except (TypeError, ValueError):  # fmt: skip
        return DEFAULT_KV_SLOTS


def build_argv(binary: str) -> list[str]:
    """Argv for the halogen entrypoint: line-buffered stdio + the `all` word.

    Legacy ran `stdbuf -oL -eL <entrypoint> all` so native diagnostics
    stay visible while stdout is captured for log forwarding.
    """
    return ["stdbuf", "-oL", "-eL", binary, "all"]
