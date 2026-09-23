"""add alias to server instances

Revision ID: 34a95f86cb0b
Revises: 07e8fe2136ad
Create Date: 2026-09-23 15:20:05.033377

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "34a95f86cb0b"
down_revision: str | Sequence[str] | None = "07e8fe2136ad"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _instance_alias_column() -> sa.Column:
    """Alias column definition (required, unique)."""
    return sa.Column("alias", sa.String(length=255), nullable=False)


def upgrade() -> None:
    """Add alias; backfill from model names for pre-existing rows."""
    # Backfill before making it required+unique: use "<model_name>-<short-id>"
    # so existing rows get a stable, collision-free alias.
    op.add_column(
        "server_instances",
        sa.Column("alias", sa.String(length=255), nullable=True),
    )
    op.execute(
        """
        UPDATE server_instances
        SET alias = COALESCE(m.name, 'model') || '-'
            || LEFT(server_instances.id::text, 8)
        FROM models m
        WHERE server_instances.model_id = m.id
          AND server_instances.alias IS NULL
        """
    )
    op.alter_column(
        "server_instances", "alias", existing_type=sa.String(length=255), nullable=False
    )
    op.create_unique_constraint(
        "uq_server_instances_alias", "server_instances", ["alias"]
    )
    op.create_index(
        "idx_server_instances_alias", "server_instances", ["alias"], unique=False
    )


def downgrade() -> None:
    """Remove alias."""
    op.drop_index("idx_server_instances_alias", table_name="server_instances")
    op.drop_constraint("uq_server_instances_alias", "server_instances", type_="unique")
    op.drop_column("server_instances", "alias")
