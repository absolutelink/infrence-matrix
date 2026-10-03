"""Track llama.cpp slot generations on servers and leases."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "7c1e9f2a4b6d"
down_revision: str | Sequence[str] | None = "5d7e9f1a3c2b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column("slot_generation", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "inference_leases",
        sa.Column("slot_generation", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("inference_leases", "slot_generation")
    op.drop_column("server_instances", "slot_generation")
