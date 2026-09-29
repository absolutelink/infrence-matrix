"""Payload and usage normalization tests used by Halogen Flash."""

from app.api.routes.v1.responses.router import _build_usage
from app.api.routes.v1.v1_chat_completions import _extract_usage
from app.api.routes.v1.v1_completions import CompletionRequest, _build_payload


def test_completion_payload_omits_none_optional_fields() -> None:
    payload = _build_payload(
        CompletionRequest(model="flash", prompt="hello", max_tokens=None),
        stream=False,
    )

    assert "max_tokens" not in payload
    assert payload["prompt"] == "hello"


def test_flash_usage_preserves_cached_prompt_tokens() -> None:
    usage = _extract_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
            "timings": {"cache_n": 75},
        }
    )

    assert usage is not None
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 12
    assert usage.total_tokens == 112
    assert usage.prompt_tokens_details == {"cached_tokens": 80, "audio_tokens": 0}


def test_flash_usage_prefers_reported_zero_cache_over_timing_fallback() -> None:
    usage = _extract_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
            "timings": {"cache_n": 75},
        }
    )

    assert usage is not None
    assert usage.prompt_tokens_details["cached_tokens"] == 0
    assert usage.completion_tokens_details["reasoning_tokens"] == 7


def test_responses_usage_maps_reported_cache_and_reasoning_stats() -> None:
    usage = _build_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 7},
            "timings": {"cache_n": 75},
        },
        fallback_chars=200,
        reasoning_tokens=5,
    )

    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 12
    assert usage["total_tokens"] == 112
    assert usage["input_tokens_details"] == {"cached_tokens": 80}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 7}


def test_responses_usage_does_not_replace_reported_zero_counts() -> None:
    usage = _build_usage(
        {"prompt_tokens": 0, "completion_tokens": 0},
        fallback_chars=200,
        reasoning_tokens=0,
    )

    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
