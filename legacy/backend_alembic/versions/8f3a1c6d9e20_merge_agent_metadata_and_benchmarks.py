"""merge agent metadata and benchmark migration branches

Revision ID: 8f3a1c6d9e20
Revises: 6b9d2e3f5071, 7d2e4f6a8b10
"""

from collections.abc import Sequence

revision: str = "8f3a1c6d9e20"
down_revision: tuple[str, str] = ("6b9d2e3f5071", "7d2e4f6a8b10")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Merge the migration branches without changing schema."""


def downgrade() -> None:
    """Merge revisions have no independent schema changes."""
