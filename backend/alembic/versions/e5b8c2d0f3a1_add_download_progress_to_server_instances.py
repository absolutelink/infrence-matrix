"""add download_progress to server_instances

Revision ID: e5b8c2d0f3a1
Revises: d4a7b1c9e2f3
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e5b8c2d0f3a1"
down_revision: str | Sequence[str] | None = "d4a7b1c9e2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column("download_progress", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("server_instances", "download_progress")
