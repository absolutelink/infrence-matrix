"""add dflash model id to server instances

Revision ID: 9f1a2b3c4d5e
Revises: f3a9c2d41b07
Create Date: 2026-09-25 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "9f1a2b3c4d5e"
down_revision: Union[str, Sequence[str], None] = "c4e8f2a1b6d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column(
            "dflash_model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "idx_server_instances_dflash_model_id",
        "server_instances",
        ["dflash_model_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_server_instances_dflash_model_id", table_name="server_instances")
    op.drop_column("server_instances", "dflash_model_id")
