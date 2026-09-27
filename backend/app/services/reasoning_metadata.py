"""Reasoning capability metadata for models and server instances.

Reasoning metadata lives in ``ServerInstance.model_metadata`` under the
``reasoning`` key and records which reasoning effort levels the model
actually supports. The legacy ``capabilities.reasoning`` boolean remains
readable as a fallback so existing rows keep working.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]

# All effort levels a reasoning-capable model can expose.
ALL_REASONING_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")


class ReasoningMetadata(BaseModel):
    """User-editable reasoning support block for a model's metadata."""

    supported: bool = True
    efforts: list[ReasoningEffort] = Field(default_factory=list)
    default: ReasoningEffort | None = None

    @model_validator(mode="after")
    def validate_efforts(self) -> ReasoningMetadata:
        if not self.supported and (self.efforts or self.default is not None):
            raise ValueError(
                "reasoning is not supported; efforts and default must be empty"
            )
        if len(self.efforts) != len(set(self.efforts)):
            raise ValueError("duplicate reasoning effort levels")
        if self.default is not None and self.default not in self.efforts:
            raise ValueError(
                "default reasoning effort must be one of the allowed efforts"
            )
        return self


def _reasoning_block(metadata: dict[str, Any] | None) -> ReasoningMetadata | None:
    """Parse the ``reasoning`` metadata block, or None when absent/invalid."""
    if not isinstance(metadata, dict):
        return None
    block = metadata.get("reasoning")
    if not isinstance(block, dict):
        return None
    try:
        return ReasoningMetadata.model_validate(block)
    except ValueError:
        return None


def allowed_reasoning_efforts(instance: Any) -> list[str] | None:
    """Effort levels permitted for this instance.

    Returns None when reasoning support is unknown (no metadata), meaning
    the caller should not restrict requests. An empty list means the model
    explicitly does not support reasoning.
    """
    metadata = getattr(instance, "model_metadata", None)
    block = _reasoning_block(metadata)
    if block is not None:
        return list(block.efforts) if block.supported else []
    legacy = (
        metadata.get("capabilities", {}).get("reasoning")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(legacy, bool):
        legacy = (
            metadata.get("supports_reasoning") if isinstance(metadata, dict) else None
        )
    if legacy is True:
        return list(ALL_REASONING_EFFORTS)
    if legacy is False:
        return []
    return None


def default_reasoning_effort(instance: Any) -> str | None:
    """Configured default effort for requests that omit reasoning, else None."""
    block = _reasoning_block(getattr(instance, "model_metadata", None))
    if block is not None and block.supported:
        return block.default
    return None


class ReasoningPolicyError(Exception):
    """Requested reasoning effort is not allowed for this model."""


def resolve_reasoning_effort(instance: Any, requested: str | None) -> str | None:
    """Map a requested effort to the effective effort for this instance.

    - No request: apply the configured default (may be None).
    - Unknown support (no metadata): permissive passthrough.
    - Requested effort allowed: passthrough.
    - "none" on a non-reasoning model: treated as a no-op (None).
    - Anything else: ReasoningPolicyError listing supported levels.
    """
    allowed = allowed_reasoning_efforts(instance)
    if requested is None:
        return default_reasoning_effort(instance)
    if allowed is None:
        return requested
    if requested in allowed:
        return requested
    if requested == "none" and not allowed:
        return None
    supported = ", ".join(allowed) if allowed else "none"
    raise ReasoningPolicyError(
        f"reasoning effort '{requested}' is not supported by this model; "
        f"supported efforts: {supported}"
    )
