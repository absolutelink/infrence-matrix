"""add mmproj_model_id to server instances

Revision ID: f3a9c2d41b07
Revises: d5e7a1b9c3f2
Create Date: 2026-09-24 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "f3a9c2d41b07"
down_revision: Union[str, Sequence[str], None] = "d5e7a1b9c3f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add optional per-instance multimodal projector reference."""
    op.add_column(
        "server_instances",
        sa.Column(
            "mmproj_model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "idx_server_instances_mmproj_model_id",
        "server_instances",
        ["mmproj_model_id"],
    )


def downgrade() -> None:
    """Remove the multimodal projector reference."""
    op.drop_index("idx_server_instances_mmproj_model_id", table_name="server_instances")
    op.drop_column("server_instances", "mmproj_model_id")