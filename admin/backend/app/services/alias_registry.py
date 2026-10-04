"""litellm model-alias registry (Phase 6).

Every provider alias must be registered with litellm before it is used in a
call, or native streaming is not selected and litellm "fake streams" the
SSE body into a non-streaming JSON parse that fails with a confusing
``APIError`` (see ``spike/litellm-fidelity/FINDINGS.md``).

Registration is idempotent and cached process-wide per alias. The
provider-definition create/update flow (Phase 9/10) should also call
:meth:`ensure_registered` so aliases are warm before first use; the
inference route calls it defensively regardless.
"""

import logging

import litellm

logger = logging.getLogger("admin.alias_registry")

# Aliases already registered in this process. litellm.register_model is
# idempotent, but the cache keeps the call off the hot path.
_registered: set[str] = set()


def ensure_registered(alias: str) -> None:
    """Register ``alias`` for native OpenResponses streaming (idempotent)."""
    if alias in _registered:
        return
    litellm.register_model(
        {
            alias: {
                "supports_native_streaming": True,
                "litellm_provider": "openai",
                "mode": "responses",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
        }
    )
    _registered.add(alias)
    logger.debug("registered litellm model alias %s", alias)


def registered_aliases() -> set[str]:
    """Snapshot of aliases registered by this module (copy)."""
    return set(_registered)
