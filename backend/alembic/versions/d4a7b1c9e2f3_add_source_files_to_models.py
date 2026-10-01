"""add source_files to models

Revision ID: d4a7b1c9e2f3
Revises: cb2140869090
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d4a7b1c9e2f3"
down_revision: str | Sequence[str] | None = "cb2140869090"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column("source_files", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("models", "source_files")
