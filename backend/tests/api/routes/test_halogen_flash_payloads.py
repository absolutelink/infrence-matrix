"""Payload and usage normalization tests used by Halogen Flash."""

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
