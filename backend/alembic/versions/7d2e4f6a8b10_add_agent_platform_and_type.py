"""Add platform and type metadata to agents.

Revision ID: 7d2e4f6a8b10
Revises: b1f8c2a47d90
Create Date: 2026-09-24

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "7d2e4f6a8b10"
down_revision: str | None = "b1f8c2a47d90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agents",
        sa.Column("platform", sa.String(), nullable=False, server_default="llamacpp"),
    )
    op.add_column(
        "agents",
        sa.Column("type", sa.String(), nullable=False, server_default="generic"),
    )
    op.alter_column("agents", "platform", server_default=None)
    op.alter_column("agents", "type", server_default=None)


def downgrade() -> None:
    op.drop_column("agents", "type")
    op.drop_column("agents", "platform")
