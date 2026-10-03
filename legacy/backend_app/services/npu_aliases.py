"""Halogen-flash NPU small-model aliases.

The engine recognises only its upstream NPU model ids in a request's
``model`` field. The broker exposes each model enabled on a flash instance
as ``<instance-alias>-<suffix>`` so names are scoped per instance and never
collide. ``ServerInstance.engine_options["npu_models"]`` stores the upstream
ids; this module turns them into client-facing names and back.

Mirrors ``agent/app/services/npu_models.py`` (the two packages do not share
imports; keep the tables in sync).
"""

# upstream model id -> client-facing suffix
NPU_SUFFIXES: dict[str, str] = {
    "qwen3-embedding-0.6b": "embed",
    "qwen3-reranker-0.6b": "rerank",
    "qwen3.5-2b": "nano",
    "decider-0.8b": "decide",
    "qwen3guard-gen-0.6b": "guard",
}

UPSTREAM_BY_SUFFIX: dict[str, str] = {v: k for k, v in NPU_SUFFIXES.items()}

STOCK_NPU_MODEL_IDS: tuple[str, ...] = tuple(NPU_SUFFIXES)

# which upstream id serves each broker route kind
ROUTE_DEFAULT_MODEL: dict[str, str] = {
    "embeddings": "qwen3-embedding-0.6b",
    "rerank": "qwen3-reranker-0.6b",
    "decisions": "decider-0.8b",
    "moderations": "qwen3guard-gen-0.6b",
}


def client_name(alias: str, upstream_id: str) -> str:
    """Return the ``<alias>-<suffix>`` public name for an enabled NPU model."""
    return f"{alias}-{NPU_SUFFIXES[upstream_id]}"


def suffix_for(upstream_id: str) -> str | None:
    return NPU_SUFFIXES.get(upstream_id)


def upstream_for_suffix(suffix: str) -> str | None:
    return UPSTREAM_BY_SUFFIX.get(suffix)
