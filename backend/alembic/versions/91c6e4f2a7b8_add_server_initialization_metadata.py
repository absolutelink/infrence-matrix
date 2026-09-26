"""add server initialization metadata

Revision ID: 91c6e4f2a7b8
Revises: 2f4a6c8e1b90
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "91c6e4f2a7b8"
down_revision: str | Sequence[str] | None = "2f4a6c8e1b90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column("model_metadata", sa.JSON(), nullable=False, server_default="{}"),
    )
    # Rows created before initialization existed must be initialized before
    # they can be started or advertised as stopped models.
    op.execute(
        "UPDATE server_instances SET status = 'uninitialized' "
        "WHERE status = 'stopped'"
    )


def downgrade() -> None:
    op.drop_column("server_instances", "model_metadata")
