"""Drop the obsolete model context length column.

Revision ID: c4e8f2a1b6d0
Revises: 9d3e7f1a2b4c
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c4e8f2a1b6d0"
down_revision: str | Sequence[str] | None = "9d3e7f1a2b4c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Remove the legacy column retained by databases created before 8a6b8f2."""
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("models")}
    if "context_length" in columns:
        op.drop_column("models", "context_length")


def downgrade() -> None:
    """Restore the legacy column for downgrade compatibility."""
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("models")}
    if "context_length" not in columns:
        op.add_column(
            "models",
            sa.Column(
                "context_length", sa.Integer(), nullable=False, server_default="4096"
            ),
        )
