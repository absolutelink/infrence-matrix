"""Admin SQLModel tables.

Real tables (Machine, ProviderDefinition, ProviderInstance, ResponseRecord,
TokenUsageSample) land in Phase 2 and are created by a single squashed
Alembic migration. This stub keeps the scaffold importable in the interim.
"""

from sqlmodel import SQLModel  # noqa: F401  (re-exported for alembic metadata)


class Message(SQLModel):
    """Generic message response."""

    message: str
