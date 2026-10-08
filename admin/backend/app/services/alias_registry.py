"""litellm model-alias registry (Phase 6/7).

Every provider alias must be registered with litellm before it is used in a
call, or native streaming is not selected and litellm "fake streams" the
SSE body into a non-streaming JSON parse that fails with a confusing
``APIError`` (see ``spike/litellm-fidelity/FINDINGS.md``).

**One registration covers BOTH public APIs** (``aresponses`` and
``acompletion``) with ``mode: "chat"`` + ``supports_native_streaming:
True``. Verified against litellm 1.103.x source (Phase 7 probe):

- ``aresponses`` native streaming is decided *only* by
  ``supports_native_streaming`` (``OpenAIResponsesAPIConfig.should_fake_stream``
  -> ``litellm.utils.supports_native_streaming``); the ``mode`` value is
  never consulted on the responses dispatch path. A ``mode: "chat"``
  registration streams the Responses API natively (confirmed empirically:
  stream + non-stream both hit ``{api_base}/responses``).
- ``acompletion`` **bridges to the Responses API whenever
  ``model_cost[alias]["mode"] == "responses"``** (``litellm/main.py``
  ``responses_api_bridge_check`` + the bridge dispatch). That is exactly
  what breaks chat for a Phase 6-style ``mode: "responses"``
  registration: the chat request would be sent to the provider's
  ``/v1/responses``, not ``/v1/chat/completions``. Registering
  ``mode: "chat"`` keeps ``acompletion`` on the native chat path, where
  ``supports_native_streaming: True`` selects real SSE (no fake-stream
  aggregation).
- The chat route additionally passes ``_skip_responses_api_bridge=True``
  as a belt-and-suspenders guard so a future gpt-5-style conditional
  bridge (reasoning + tools on an OpenAI-looking endpoint) can never
  reroute our chat calls to ``/v1/responses``.

Registration is idempotent and cached process-wide per alias. The
provider-definition create/update flow (Phase 9/10) should also call
:meth:`ensure_registered` so aliases are warm before first use; the
inference routes call it defensively regardless.
"""

import logging

import litellm

logger = logging.getLogger("admin.alias_registry")

# Aliases already registered in this process, mapped to the mode they were
# registered with. litellm.register_model is idempotent, but the cache keeps
# the call off the hot path. The value matters: a definition's modality can be
# PATCHed (llm<->embedding) while no backends are attached, so a cached entry
# must be re-registered when the requested mode differs (a stale chat entry
# would leave an embedding alias mis-declared, and — worse — a stale embedding
# entry lacks supports_native_streaming, so the first chat/responses call
# fake-streams into the APIError this registry exists to prevent).
_registered: dict[str, str] = {}


def ensure_registered(alias: str, mode: str = "chat") -> None:
    """Register ``alias`` with litellm (idempotent per ``(alias, mode)``).

    ``mode`` selects the endpoint family the alias serves (Phase 18):

    - ``"chat"`` (default): native streaming on BOTH /v1/responses and
      /v1/chat/completions. ``mode: "chat"`` is deliberate: it is inert for
      the responses API (which keys off ``supports_native_streaming`` only)
      and it prevents ``acompletion`` from bridging the request to
      ``/v1/responses``.
    - ``"embedding"``: /v1/embeddings. Embeddings are non-streaming, so no
      ``supports_native_streaming`` / ``supports_reasoning`` are declared;
      ``mode: "embedding"`` keeps ``aembedding`` on the native embedding
      path.

    The cache is keyed by alias but stores the registered mode: a modality
    change re-registers (litellm overwrites) rather than early-returning.
    """
    if _registered.get(alias) == mode:
        return
    if mode == "embedding":
        litellm.register_model(
            {
                alias: {
                    "litellm_provider": "openai",
                    "mode": "embedding",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                }
            }
        )
    else:
        litellm.register_model(
            {
                alias: {
                    "supports_native_streaming": True,
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                    # Declare reasoning support: litellm refuses `reasoning.*`
                    # params with UnsupportedParamsError when the model info
                    # lacks `supports_reasoning` (production: /v1/responses
                    # reasoning.effort -> 502 on a perfectly healthy backend).
                    # Self-hosted backends accept-and-ignore the param, so
                    # blanket-declaring it is safe; reasoning models then get
                    # real effort control and instruct models simply ignore it.
                    "supports_reasoning": True,
                }
            }
        )
    _registered[alias] = mode
    logger.debug("registered litellm model alias %s (mode=%s)", alias, mode)


def registered_aliases() -> set[str]:
    """Snapshot of aliases registered by this module (copy)."""
    return set(_registered)
