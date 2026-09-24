"""Add MTP draft maximum to server instances.

Revision ID: 3cc6291025fe
Revises: f3a9c2d41b07
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "3cc6291025fe"
down_revision: str | Sequence[str] | None = "f3a9c2d41b07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column("mtp_draft_max", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("server_instances", "mtp_draft_max")
