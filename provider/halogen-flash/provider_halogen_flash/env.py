"""HALOGEN_* environment construction for halogen-flash, ported from the
legacy agent.

Halogen Flash is configured almost entirely through `HALOGEN_*`
environment variables (43 semantic options plus the fixed wiring).
`backend_config.options` holds the semantic config; this module maps it
to the env dict for the subprocess with None-skip semantics (absent or
None values are never emitted; the engine keeps its own default).

Static-port rule: the Flash engine hashes its server variables — ports
included — into its on-disk prompt-cache fingerprint. Random per-start
ports would invalidate the disk cache on every boot, so the (api,
engine) pair must be **stable across restarts**: either pinned in
`backend_config` (`api_port`/`engine_port`) or derived deterministically
from `MACHINE_UID` inside the dedicated FLASH range (8200–8289, kept
clear of the ephemeral llama-server range). See `derive_static_ports()`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

# backend_config.options key -> HALOGEN_* env var. Every entry is
# optional; None / absent is never emitted.
ENV_MAP: dict[str, str] = {
    # KV cache & admission
    "kv_slots": "HALOGEN_KV_SLOTS",
    "kv_pool_positions": "HALOGEN_KV_POOL_POSITIONS",
    "kv_pool_fit": "HALOGEN_KV_POOL_FIT",
    "host_reserve_gib": "HALOGEN_HOST_RESERVE_GIB",
    # Context / length limits
    "ctx": "HALOGEN_CTX",
    "max_tok": "HALOGEN_MAX_TOK",
    "rope_yarn": "HALOGEN_ROPE_YARN",
    "admit_chunk": "HALOGEN_ADMIT_CHUNK",
    "indexer_budget": "HALOGEN_INDEXER_BUDGET",
    "max_tokens_cap": "HALOGEN_MAX_TOKENS_CAP",
    "max_tokens_default": "HALOGEN_MAX_TOKENS_DEFAULT",
    # Timeouts / keepalive
    "queue_timeout": "HALOGEN_QUEUE_TIMEOUT",
    "keepalive_timeout": "HALOGEN_KEEPALIVE_TIMEOUT",
    "sse_keepalive_s": "HALOGEN_SSE_KEEPALIVE_S",
    # Sampling defaults
    "temperature": "HALOGEN_TEMPERATURE",
    "top_p": "HALOGEN_TOP_P",
    "top_k": "HALOGEN_TOP_K",
    "min_p": "HALOGEN_MIN_P",
    "presence_penalty": "HALOGEN_PRESENCE_PENALTY",
    "frequency_penalty": "HALOGEN_FREQUENCY_PENALTY",
    # Reasoning
    "reasoning_effort": "HALOGEN_REASONING_EFFORT",
    "enable_thinking": "HALOGEN_ENABLE_THINKING",
    "max_thinking_tokens": "HALOGEN_MAX_THINKING_TOKENS",
    "thinking_answer_room": "HALOGEN_THINKING_ANSWER_ROOM",
    # Speculative decoding
    "drafter_default": "HALOGEN_DRAFTER_DEFAULT",
    "mtp_depth": "HALOGEN_MTP_DEPTH",
    "pld": "HALOGEN_PLD",
    "spec_adapt": "HALOGEN_SPEC_ADAPT",
    # Prompt cache
    "prompt_cache": "HALOGEN_PROMPT_CACHE",
    "cache_inplace": "HALOGEN_CACHE_INPLACE",
    "prefill_chunk": "HALOGEN_PREFILL_CHUNK",
    "cache_entries": "HALOGEN_CACHE_ENTRIES",
    "cache_branches": "HALOGEN_CACHE_BRANCHES",
    "cache_snap3": "HALOGEN_CACHE_SNAP3",
    "cache_full": "HALOGEN_CACHE_FULL",
    "cache_disk_gib": "HALOGEN_CACHE_DISK_GIB",
    "cache_prune_old": "HALOGEN_CACHE_PRUNE_OLD",
    # Composable context
    "composable_context": "HALOGEN_COMPOSABLE_CONTEXT",
    "composable_context_floor": "HALOGEN_COMPOSABLE_CONTEXT_FLOOR",
    "composable_context_bytes": "HALOGEN_COMPOSABLE_CONTEXT_BYTES",
    # Structured output / vision
    "grammar": "HALOGEN_GRAMMAR",
    "vision_tower": "HALOGEN_VISION_TOWER",
    "vision_max_pixels": "HALOGEN_VISION_MAX_PIXELS",
}

# Static port range dedicated to Flash servers (api + engine pair). Kept
# clear of the ephemeral llama-server range so fingerprints stay stable
# and ports never collide with dynamically allocated servers.
FLASH_PORT_START = 8200
FLASH_PORT_END = 8290
FLASH_PORT_PAIRS = (FLASH_PORT_END - FLASH_PORT_START) // 2  # 45 slots

DEFAULT_KV_SLOTS = 1


def derive_static_ports(machine_uid: str) -> tuple[int, int]:
    """Deterministic (api_port, engine_port) pair from the machine UID.

    The legacy backend allocated a free pair on first start and persisted
    it (the provider has no DB now); hashing MACHINE_UID into the static
    range gives the same stability property: the same machine always
    boots Flash on the same pair, so its disk-cache fingerprint survives
    restarts. Collisions between two Flash instances on one machine are
    impossible by construction (one backend per provider instance, and
    the pair is machine-scoped).
    """
    digest = hashlib.sha256(machine_uid.encode("utf-8")).digest()
    slot = int.from_bytes(digest[:4], "big") % FLASH_PORT_PAIRS
    api_port = FLASH_PORT_START + slot * 2
    engine_port = api_port + 1
    return api_port, engine_port


def resolve_ports(cfg: dict[str, Any], machine_uid: str) -> tuple[int, int]:
    """Resolve the pinned (api, engine) pair for this instance.

    Priority: explicit `api_port`/`engine_port` in backend_config; else
    the deterministic static pair derived from MACHINE_UID. (The
    instance's own PROVIDER_PORT is its admin-facing surface and never
    participates in the engine's fingerprint pair.)
    """
    api = cfg.get("api_port")
    engine = cfg.get("engine_port")
    if api is not None and engine is not None:
        return int(api), int(engine)
    return derive_static_ports(machine_uid)


def build_env(
    options: dict[str, Any],
    *,
    api_port: int,
    engine_port: int,
    checkpoint_path: str,
    tokenizer_path: str,
    cache_dir: Path | None = None,
    npu_models: list[str] | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map a halogen-flash backend_config into the subprocess environment.

    `cache_dir` is the per-instance disk-cache directory: it is exported
    as HALOGEN_CACHE_DIR only when `options.cache_dir_enabled` is true
    (legacy semantics). `npu_models` (already NPU-probe-gated by the
    caller) joins with commas so the engine reads a bare id list, not a
    Python repr.
    """
    env = dict(base_env if base_env is not None else os.environ)
    env["HALOGEN_API_PORT"] = str(api_port)
    env["HALOGEN_PORT"] = str(engine_port)
    env["HALOGEN_BIND"] = "127.0.0.1"
    env["HALOGEN_ENGINE"] = f"127.0.0.1:{engine_port}"
    env["HALOGEN_CHECKPOINT"] = str(checkpoint_path)
    env["HALOGEN_TOKENIZER"] = str(tokenizer_path)
    opts = options or {}
    for key, variable in ENV_MAP.items():
        if opts.get(key) is not None:
            env[variable] = str(opts[key])
    if opts.get("cache_dir_enabled") is True and cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        env["HALOGEN_CACHE_DIR"] = str(cache_dir)
    if npu_models:
        env["HALOGEN_NPU_MODELS"] = ",".join(str(m) for m in npu_models)
    return env


def effective_capacity(options: dict[str, Any]) -> int:
    """Flash request slots = HALOGEN_KV_SLOTS (options.kv_slots)."""
    opts = options or {}
    try:
        return max(int(opts.get("kv_slots", DEFAULT_KV_SLOTS)), 1)
    except TypeError, ValueError:
        return DEFAULT_KV_SLOTS


def build_argv(binary: str) -> list[str]:
    """Argv: line-buffered stdio + the entrypoint word `all` (legacy)."""
    return ["stdbuf", "-oL", "-eL", binary, "all"]
