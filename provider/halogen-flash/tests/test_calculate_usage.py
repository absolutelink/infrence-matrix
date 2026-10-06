"""Table tests for the halogen-flash usage-normalization override.

Each case: (raw backend usage shape) -> (normalized spec usage). These
are the shapes the legacy translator/router had to cope with; the
normalized dict is exactly what the admin's `persist_turn` reads.
"""

import pytest

from provider_halogen_flash.usage import calculate_usage


def _spec(inp, outp, total, cached, reasoning, rates=None):
    usage = {
        "input_tokens": inp,
        "output_tokens": outp,
        "total_tokens": total,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }
    if rates:
        usage["completion_tokens_details"] = rates
    return usage


CASES = [
    # Already spec-shaped: pass through, totals recomputed if absent.
    (
        "spec passthrough",
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 1},
        },
        0,
        0,
        _spec(10, 5, 15, 2, 1),
    ),
    # Chat-style OpenAI counts -> spec names.
    (
        "chat-style usage",
        {
            "prompt_tokens": 100,
            "completion_tokens": 12,
            "total_tokens": 112,
            "prompt_tokens_details": {"cached_tokens": 80, "audio_tokens": 0},
        },
        0,
        0,
        _spec(100, 12, 112, 80, 0),
    ),
    # llama.cpp native timing chunk (no usage object at all).
    (
        "native timings chunk",
        {"tokens_evaluated": 42, "tokens_predicted": 7, "timings": {"cache_n": 30}},
        0,
        0,
        _spec(42, 7, 49, 30, 0),
    ),
    # Reported zero cache must NOT be replaced by the timing fallback.
    (
        "reported zero cache beats timings fallback",
        {
            "prompt_tokens": 50,
            "completion_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 0},
            "timings": {"cache_n": 33},
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
        0,
        0,
        _spec(50, 9, 59, 0, 7),
    ),
    # Nothing reported: estimate output from streamed chars (chars // 4).
    (
        "no counts: estimate from chars",
        {},
        100,
        3,
        _spec(0, 25, 25, 0, 3),
    ),
    (
        "None raw: estimate from chars",
        None,
        8,
        0,
        _spec(0, 2, 2, 0, 0),
    ),
    # Partial: prompt reported, completion not — legacy semantics keep
    # the reported prompt and leave output 0 (the char estimate applies
    # only when NO count key is present at all).
    (
        "partial counts",
        {"prompt_tokens": 20},
        40,
        5,
        _spec(20, 0, 20, 0, 5),
    ),
    # Reasoning: reported detail wins over the driver's counted deltas.
    (
        "reported reasoning wins",
        {
            "input_tokens": 5,
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 4},
        },
        0,
        99,
        _spec(5, 10, 15, 0, 4),
    ),
    # Rate fields carried into completion_tokens_details.
    (
        "rates from timings",
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "timings": {
                "prompt_per_second": 1412.21,
                "predicted_per_second": 36.2534,
            },
        },
        0,
        0,
        _spec(
            100,
            10,
            110,
            0,
            0,
            rates={
                "prompt_per_second": 1412.21,
                "predicted_per_second": 36.2534,
            },
        ),
    ),
    # total_tokens reported explicitly wins.
    (
        "explicit total wins",
        {"input_tokens": 3, "output_tokens": 4, "total_tokens": 100},
        0,
        0,
        _spec(3, 4, 100, 0, 0),
    ),
]


@pytest.mark.parametrize(
    "_name, raw, fallback_chars, reasoning_tokens, expected",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_calculate_usage_table(_name, raw, fallback_chars, reasoning_tokens, expected):
    assert (
        calculate_usage(
            raw, fallback_chars=fallback_chars, reasoning_tokens=reasoning_tokens
        )
        == expected
    )


def test_estimate_only_when_no_count_keys_at_all() -> None:
    # Reported counts (even zeros) suppress the char estimate.
    usage = calculate_usage(
        {"prompt_tokens": 0, "completion_tokens": 0}, fallback_chars=1000
    )
    assert usage["output_tokens"] == 0
    # A raw with only details (no counts) still triggers the estimate.
    usage = calculate_usage(
        {"prompt_tokens_details": {"cached_tokens": 5}}, fallback_chars=400
    )
    assert usage["output_tokens"] == 100
    assert usage["input_tokens"] == 0
    assert usage["input_tokens_details"] == {"cached_tokens": 5}


def test_garbage_rates_ignored() -> None:
    usage = calculate_usage(
        {"input_tokens": 1, "output_tokens": 1, "timings": {"prompt_per_second": "x"}}
    )
    assert "completion_tokens_details" not in usage


def test_reported_zero_counts_are_not_replaced_by_fallback_keys() -> None:
    usage = calculate_usage(
        {
            "input_tokens": 0,
            "prompt_tokens": 12,
            "output_tokens": 0,
            "completion_tokens": 9,
        },
        fallback_chars=100,
    )
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
