"""Drop agent inference slot protocol capability.

Revision ID: cb2140869090
Revises: f6d2a8c13b04
Create Date: 2026-09-30

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "cb2140869090"
down_revision: str | Sequence[str] | None = "f6d2a8c13b04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("agents", "inference_slot_protocol")


def downgrade() -> None:
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
