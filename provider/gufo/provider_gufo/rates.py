"""Gufo rate-gauge parsing, ported from the legacy agent's token_stats.

Gufo exposes llama.cpp-style Prometheus counters on ``GET /metrics`` but
no ``*_seconds_total`` totals: it reports instantaneous rates directly as
gauges. These map straight onto the admin's TokenUsageSample rate fields
and take precedence over anything computed from counters.
"""

from typing import Any

_COUNTER_NAMES = {
    "llamacpp:prompt_tokens_total": "prompt_tokens",
    "llamacpp:prompt_seconds_total": "prompt_seconds",
    "llamacpp:tokens_predicted_total": "predicted_tokens",
    "llamacpp:tokens_predicted_seconds_total": "predicted_seconds",
}

# Gufo's instantaneous rate gauges -> admin TokenUsageSample field names.
_DIRECT_RATE_NAMES = {
    "llamacpp:prompt_tokens_seconds": "prompt_per_second",
    "llamacpp:predicted_tokens_seconds": "predicted_per_second",
}


def parse_rate_gauges(metrics_text: str) -> dict[str, float]:
    """Extract prompt/generation token rates from a Prometheus text body.

    Returns ``{prompt_per_second, predicted_per_second}`` for whatever the
    backend reported. Gauges take precedence; when a gauge is missing but
    the matching ``*_seconds_total`` counter exists, the lifetime-average
    rate is derived. Zero/negative durations and unparseable lines are
    ignored; the result may be empty.
    """
    counters: dict[str, float] = {}
    rates: dict[str, float] = {}
    for line in metrics_text.splitlines():
        fields = line.strip().split()
        if len(fields) < 2:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if fields[0] in _COUNTER_NAMES:
            counters[_COUNTER_NAMES[fields[0]]] = value
        elif fields[0] in _DIRECT_RATE_NAMES:
            rates[_DIRECT_RATE_NAMES[fields[0]]] = value

    if "prompt_per_second" not in rates:
        prompt_seconds = counters.get("prompt_seconds", 0)
        if prompt_seconds > 0:
            rates["prompt_per_second"] = (
                counters.get("prompt_tokens", 0) / prompt_seconds
            )
    if "predicted_per_second" not in rates:
        predicted_seconds = counters.get("predicted_seconds", 0)
        if predicted_seconds > 0:
            rates["predicted_per_second"] = (
                counters.get("predicted_tokens", 0) / predicted_seconds
            )
    return rates


def inject_rates(usage: dict[str, Any], rates: dict[str, float]) -> dict[str, Any]:
    """Merge scraped rates into a spec usage dict's completion details.

    The admin's `persist_turn` reads `prompt_per_second` /
    `predicted_per_second` from `usage.completion_tokens_details`, so
    that's where the gufo gauges land. Non-dict usage is returned as-is.
    """
    if not rates or not isinstance(usage, dict):
        return usage
    details = dict(usage.get("completion_tokens_details") or {})
    details.update(rates)
    usage["completion_tokens_details"] = details
    return usage
