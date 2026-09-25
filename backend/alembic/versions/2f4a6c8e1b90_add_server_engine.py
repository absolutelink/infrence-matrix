"""add server engine metadata

Revision ID: 2f4a6c8e1b90
Revises: 4c7e9a1b2d3f
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "2f4a6c8e1b90"
down_revision: str | Sequence[str] | None = "4c7e9a1b2d3f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column("engine", sa.String(length=32), nullable=False, server_default="llamacpp"),
    )
    op.add_column(
        "server_instances",
        sa.Column("engine_options", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("server_instances", "engine_options")
    op.drop_column("server_instances", "engine")
