"""Rate-gauge parsing tests (canned gufo /metrics shapes)."""

from provider_gufo.rates import inject_rates, parse_rate_gauges

GUFO_METRICS = """# HELP llamacpp:prompt_tokens_total Total prompt tokens processed
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 7920
# HELP llamacpp:tokens_predicted_total Total tokens generated
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 120
# HELP llamacpp:prompt_tokens_seconds Prompt processing speed in tokens per second
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 1412.21
# HELP llamacpp:predicted_tokens_seconds Generation speed in tokens per second
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 36.2534
# HELP llamacpp:kv_cache_usage_ratio KV cache usage ratio
# TYPE llamacpp:kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.0
"""


def test_maps_gufo_gauge_rates() -> None:
    assert parse_rate_gauges(GUFO_METRICS) == {
        "prompt_per_second": 1412.21,
        "predicted_per_second": 36.2534,
    }


def test_prefers_gauge_over_computed_counter() -> None:
    metrics = """llamacpp:prompt_tokens_total 1000
llamacpp:prompt_seconds_total 100
llamacpp:prompt_tokens_seconds 95.5
"""
    assert parse_rate_gauges(metrics) == {"prompt_per_second": 95.5}


def test_derives_rate_from_counters_when_gauge_missing() -> None:
    metrics = """llamacpp:prompt_tokens_total 1000
llamacpp:prompt_seconds_total 100
llamacpp:tokens_predicted_total 50
llamacpp:tokens_predicted_seconds_total 2.5
"""
    assert parse_rate_gauges(metrics) == {
        "prompt_per_second": 10.0,
        "predicted_per_second": 20.0,
    }


def test_zero_duration_ratios_ignored() -> None:
    metrics = """llamacpp:prompt_tokens_total 200
llamacpp:prompt_seconds_total 0
llamacpp:tokens_predicted_total 10
llamacpp:tokens_predicted_seconds_total 0
"""
    assert parse_rate_gauges(metrics) == {}


def test_unparseable_lines_skipped() -> None:
    metrics = """# comment
garbage
llamacpp:prompt_tokens_seconds notanumber
llamacpp:predicted_tokens_seconds 12.5
"""
    assert parse_rate_gauges(metrics) == {"predicted_per_second": 12.5}


def test_empty_text() -> None:
    assert parse_rate_gauges("") == {}


def test_inject_rates_merges_into_completion_details() -> None:
    usage = {"input_tokens": 3, "output_tokens": 8}
    out = inject_rates(usage, {"prompt_per_second": 100.0})
    assert out["completion_tokens_details"] == {"prompt_per_second": 100.0}
    assert out["input_tokens"] == 3


def test_inject_rates_preserves_existing_details() -> None:
    usage = {
        "input_tokens": 3,
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    out = inject_rates(usage, {"predicted_per_second": 30.0})
    assert out["completion_tokens_details"] == {
        "reasoning_tokens": 2,
        "predicted_per_second": 30.0,
    }


def test_inject_rates_noop_on_empty_rates() -> None:
    usage = {"input_tokens": 3}
    assert inject_rates(usage, {}) == usage
