"""Tests for quantization tag extraction and /models/files enrichment."""

from typing import Any

from app.services.models import extract_quantization, is_aux_type


def test_quantization_from_filename() -> None:
    assert (
        extract_quantization("Qwen3-7B-Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf")
        == "UD-Q4_K_XL"
    )


def test_quantization_from_subfolder() -> None:
    assert extract_quantization("Q8_0/model.gguf") == "Q8_0"


def test_quantization_from_split_in_subfolder() -> None:
    assert extract_quantization("UD-Q4_K_XL/model-00001-of-00003.gguf") == "UD-Q4_K_XL"


def test_quantization_iq_and_bf16_tokens() -> None:
    assert extract_quantization("foo-IQ2_XXS.gguf") == "IQ2_XXS"
    assert extract_quantization("bar-BF16.gguf") == "BF16"
    assert extract_quantization("baz-Q5_K_M.gguf") == "Q5_K_M"


def test_card_data_explicit_mapping_overrides_filename() -> None:
    card = {"model-00001-of-00003.gguf": "Q6_K"}
    assert extract_quantization("UD-Q4_K_XL/model-00001-of-00003.gguf", card) == "Q6_K"


def test_card_data_inverse_mapping_label_key() -> None:
    # dict keyed by quantization label whose value lists files
    card = {"Q8_0": ["big/model-00001-of-00002.gguf"]}
    assert extract_quantization("big/model-00001-of-00002.gguf", card) == "Q8_0"


def test_card_data_list_of_entries() -> None:
    card = [
        {"filename": "a.gguf", "quantization": "Q4_K_M"},
        {"filename": "b.gguf", "quantization": "Q8_0"},
    ]
    assert extract_quantization("b.gguf", card) == "Q8_0"
    assert extract_quantization("a.gguf", card) == "Q4_K_M"


def test_no_tag_returns_none() -> None:
    assert extract_quantization("random-file.gguf") is None
    assert extract_quantization("models/foo.gguf") is None


def test_empty_and_none_path() -> None:
    assert extract_quantization("") is None
    assert extract_quantization(None) is None


def test_malformed_card_data_does_not_crash() -> None:
    # Unexpected shapes must fall back to filename parsing, never raise.
    weird_inputs: list[Any] = [
        "not-a-dict",
        123,
        [1, 2, 3],
        {"unexpected": None},
        [{"no_name": "x"}],
        {"file.gguf": 42},
        {42: "Q8_0"},
    ]
    for weird in weird_inputs:
        # Should not raise; falls back to filename-derived tag (none here).
        result = extract_quantization("plain.gguf", weird)
        assert result is None


def test_malformed_card_data_still_uses_filename_fallback() -> None:
    # Mapping misses this file, but the filename has a tag.
    card = {"other.gguf": "Q2_K"}
    assert extract_quantization("model-Q4_K_M.gguf", card) == "Q4_K_M"


def test_is_aux_type() -> None:
    assert is_aux_type("mmproj") is True
    assert is_aux_type("dflash") is True
    assert is_aux_type("mtp") is True
    assert is_aux_type("llm") is False
    assert is_aux_type(None) is False
