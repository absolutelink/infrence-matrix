"""Usage normalization override for halogen-flash.

THE canonical provider-override example (see provider/README.md
"Overriding usage normalization").

The halogen-flash backend is **not spec-compliant on usage**: depending
on engine version and endpoint it may emit OpenAI chat-style counts
(`prompt_tokens`/`completion_tokens`), llama.cpp native timing keys at
the top level (`tokens_evaluated`/`tokens_predicted`, `timings.cache_n`),
report only *some* detail fields, or report nothing in-stream at all. The
admin's `persist_turn` needs a single spec-shaped usage dict
(`input_tokens`/`output_tokens`/`total_tokens` +
`input_tokens_details.cached_tokens` +
`output_tokens_details.reasoning_tokens` + optional rate fields).

`calculate_usage` normalizes every observed raw shape into that spec
dict **before the driver yields the terminal event**, so the admin's
persisted token counts are correct regardless of what the backend
reported. Rules (ported from the legacy translator/router):

- Reported values are trusted verbatim — including reported **zeros**
  (a reported 0-cached must not be replaced by a timing fallback).
- Missing counts are *estimated*: output from accumulated streamed
  characters (chars // 4) only when the backend reported no count keys
  at all.
- `timings.cache_n` is the cached-token fallback when no details object
  carries `cached_tokens`.
- Reasoning tokens prefer the reported detail over the driver's counted
  reasoning deltas.
- Rate fields (`prompt_per_second` / `predicted_per_second`) from the
  raw `timings` are carried into `completion_tokens_details` so the
  admin's TokenUsageSample gets them.
"""

from typing import Any

# Keys that indicate the backend reported real token counts.
_COUNT_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "tokens_evaluated",
    "tokens_predicted",
)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def calculate_usage(
    raw: dict[str, Any] | None,
    *,
    fallback_chars: int = 0,
    reasoning_tokens: int = 0,
) -> dict[str, Any]:
    """Normalize a raw halogen-flash usage blob into a spec usage dict.

    ``raw``: the usage object (or native timing chunk) the backend
    emitted; may be None/empty. ``fallback_chars``: characters streamed
    in the terminal response, used to estimate output tokens only when
    the backend reported no counts. ``reasoning_tokens``: the driver's
    own count of reasoning-delta tokens, used unless the backend
    reported its own reasoning detail.
    """
    raw = raw if isinstance(raw, dict) else {}
    timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}

    reported_counts = any(k in raw for k in _COUNT_KEYS)

    input_tokens = _int(
        next(
            (
                raw[key]
                for key in ("input_tokens", "prompt_tokens", "tokens_evaluated")
                if key in raw
            ),
            0,
        )
    )
    output_tokens = _int(
        next(
            (
                raw[key]
                for key in ("output_tokens", "completion_tokens", "tokens_predicted")
                if key in raw
            ),
            0,
        )
    )
    if not reported_counts:
        # Nothing in-stream: estimate the generation from streamed chars.
        output_tokens = max(int(fallback_chars) // 4, 0)

    total = raw.get("total_tokens")
    total_tokens = _int(total) if total is not None else input_tokens + output_tokens

    # Cached tokens: reported details win (even when 0); timings.cache_n
    # is only the fallback.
    prompt_details = raw.get("prompt_tokens_details") or {}
    input_details = raw.get("input_tokens_details") or {}
    if "cached_tokens" in prompt_details:
        cached = _int(prompt_details["cached_tokens"])
    elif "cached_tokens" in input_details:
        cached = _int(input_details["cached_tokens"])
    else:
        cached = _int(timings.get("cache_n"))

    # Reasoning tokens: reported detail wins over the driver's count.
    completion_details = raw.get("completion_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    reported_reasoning = (
        completion_details.get("reasoning_tokens")
        if "reasoning_tokens" in completion_details
        else output_details.get("reasoning_tokens")
    )
    reasoning = (
        _int(reported_reasoning)
        if reported_reasoning is not None
        else _int(reasoning_tokens)
    )

    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }

    # Rate passthrough: the admin's persist_turn reads these from
    # completion_tokens_details.
    rates: dict[str, float] = {}
    for key in ("prompt_per_second", "predicted_per_second"):
        value = raw.get(key)
        if value is None:
            value = timings.get(key)
        if value is not None:
            try:
                rates[key] = float(value)
            except TypeError, ValueError:
                pass
    if rates:
        usage["completion_tokens_details"] = rates
    return usage
