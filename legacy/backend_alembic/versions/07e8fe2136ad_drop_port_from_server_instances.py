"""drop port from server instances

Revision ID: 07e8fe2136ad
Revises: 348f7cef7f91
Create Date: 2026-09-23 00:58:11.399352

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "07e8fe2136ad"
down_revision: str | Sequence[str] | None = "348f7cef7f91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Port is agent-allocated per start; it is not persisted entity data."""
    op.drop_index("uq_server_instances_active_port", table_name="server_instances")
    op.drop_index("idx_server_instances_port", table_name="server_instances")
    op.drop_column("server_instances", "port")


def downgrade() -> None:
    """Restore the port column."""
    op.add_column(
        "server_instances",
        sa.Column("port", sa.Integer(), autoincrement=False, nullable=False),
    )
    op.create_index(
        "idx_server_instances_port", "server_instances", ["port"], unique=False
    )
    op.create_index(
        "uq_server_instances_active_port",
        "server_instances",
        ["agent_id", "port"],
        unique=True,
        postgresql_where=sa.text("status IN ('starting', 'running')"),
    )
