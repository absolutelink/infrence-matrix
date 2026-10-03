"""Add agent inference slot protocol capability.

Revision ID: e8a4d1c6f203
Revises: b3e1c0a9d7f2
Create Date: 2026-09-29

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e8a4d1c6f203"
down_revision: str | None = "b3e1c0a9d7f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agents",
        sa.Column(
            "inference_slot_protocol",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.alter_column("agents", "inference_slot_protocol", server_default=None)


def downgrade() -> None:
    op.drop_column("agents", "inference_slot_protocol")
