"""add server instance config columns

Revision ID: 3caae8fa9527
Revises: b7f2d3e9c1a4
Create Date: 2026-09-22 23:47:37.284863

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3caae8fa9527"
down_revision: str | Sequence[str] | None = "b7f2d3e9c1a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add per-instance server config columns."""
    op.add_column(
        "server_instances",
        sa.Column("gpu_layers", sa.Integer(), nullable=False, server_default="35"),
    )
    op.add_column(
        "server_instances",
        sa.Column("context_size", sa.Integer(), nullable=False, server_default="4096"),
    )
    op.add_column(
        "server_instances",
        sa.Column("flash_attn", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    """Remove per-instance server config columns."""
    op.drop_column("server_instances", "flash_attn")
    op.drop_column("server_instances", "context_size")
    op.drop_column("server_instances", "gpu_layers")
