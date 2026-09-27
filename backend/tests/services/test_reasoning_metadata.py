"""Tests for reasoning capability metadata parsing and enforcement."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.services.reasoning_metadata import (
    ReasoningMetadata,
    ReasoningPolicyError,
    allowed_reasoning_efforts,
    default_reasoning_effort,
    resolve_reasoning_effort,
)


def _instance(metadata: dict | None) -> SimpleNamespace:
    return SimpleNamespace(model_metadata=metadata)


class TestReasoningMetadataValidation:
    def test_valid_block(self) -> None:
        meta = ReasoningMetadata(
            supported=True, efforts=["low", "medium"], default="medium"
        )
        assert meta.efforts == ["low", "medium"]

    def test_default_must_be_in_efforts(self) -> None:
        with pytest.raises(ValidationError, match="default reasoning effort"):
            ReasoningMetadata(supported=True, efforts=["low"], default="high")

    def test_unknown_effort_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningMetadata(supported=True, efforts=["ultra"])

    def test_unsupported_rejects_levels(self) -> None:
        with pytest.raises(ValidationError, match="not supported"):
            ReasoningMetadata(supported=False, efforts=["low"])

    def test_duplicate_efforts_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            ReasoningMetadata(supported=True, efforts=["low", "low"])


class TestAllowedReasoningEfforts:
    def test_no_metadata_is_unknown(self) -> None:
        assert allowed_reasoning_efforts(_instance(None)) is None
        assert allowed_reasoning_efforts(_instance({})) is None

    def test_reasoning_block_takes_priority(self) -> None:
        instance = _instance(
            {
                "reasoning": {"supported": True, "efforts": ["low", "high"]},
                "capabilities": {"reasoning": False},
            }
        )
        assert allowed_reasoning_efforts(instance) == ["low", "high"]

    def test_explicitly_unsupported(self) -> None:
        instance = _instance({"reasoning": {"supported": False}})
        assert allowed_reasoning_efforts(instance) == []

    def test_legacy_capabilities_true_allows_all(self) -> None:
        instance = _instance({"capabilities": {"reasoning": True}})
        assert allowed_reasoning_efforts(instance) == [
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
        ]

    def test_legacy_capabilities_false(self) -> None:
        instance = _instance({"capabilities": {"reasoning": False}})
        assert allowed_reasoning_efforts(instance) == []

    def test_invalid_block_falls_back_to_legacy(self) -> None:
        instance = _instance(
            {"reasoning": {"efforts": ["ultra"]}, "capabilities": {"reasoning": True}}
        )
        assert allowed_reasoning_efforts(instance) == [
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
        ]


class TestResolveReasoningEffort:
    def test_unknown_support_passthrough(self) -> None:
        instance = _instance({})
        assert resolve_reasoning_effort(instance, "high") == "high"
        assert resolve_reasoning_effort(instance, None) is None

    def test_allowed_effort_passthrough(self) -> None:
        instance = _instance(
            {"reasoning": {"supported": True, "efforts": ["low", "medium"]}}
        )
        assert resolve_reasoning_effort(instance, "medium") == "medium"

    def test_disallowed_effort_raises(self) -> None:
        instance = _instance(
            {"reasoning": {"supported": True, "efforts": ["low", "medium"]}}
        )
        with pytest.raises(ReasoningPolicyError, match="supported efforts: low, medium"):
            resolve_reasoning_effort(instance, "xhigh")

    def test_none_on_unsupported_model_is_noop(self) -> None:
        instance = _instance({"reasoning": {"supported": False}})
        assert resolve_reasoning_effort(instance, "none") is None

    def test_unsupported_model_raises_for_levels(self) -> None:
        instance = _instance({"reasoning": {"supported": False}})
        with pytest.raises(ReasoningPolicyError, match="supported efforts: none"):
            resolve_reasoning_effort(instance, "low")

    def test_default_applied_when_omitted(self) -> None:
        instance = _instance(
            {
                "reasoning": {
                    "supported": True,
                    "efforts": ["low", "high"],
                    "default": "high",
                }
            }
        )
        assert resolve_reasoning_effort(instance, None) == "high"

    def test_default_ignored_when_request_given(self) -> None:
        instance = _instance(
            {
                "reasoning": {
                    "supported": True,
                    "efforts": ["low", "high"],
                    "default": "high",
                }
            }
        )
        assert resolve_reasoning_effort(instance, "low") == "low"

    def test_default_reasoning_effort_helper(self) -> None:
        assert (
            default_reasoning_effort(
                _instance({"reasoning": {"supported": True, "efforts": [], "default": None}})
            )
            is None
        )
        assert (
            default_reasoning_effort(
                _instance({"reasoning": {"supported": False, "default": None}})
            )
            is None
        )
