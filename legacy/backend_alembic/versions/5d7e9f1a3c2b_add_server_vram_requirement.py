"""add configured VRAM requirement to server instances

Revision ID: 5d7e9f1a3c2b
Revises: 3caae8fa9527
Create Date: 2026-09-26 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "5d7e9f1a3c2b"
down_revision: str | Sequence[str] | None = ("3caae8fa9527", "91c6e4f2a7b8")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column("vram_required_bytes", sa.BigInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("server_instances", "vram_required_bytes")
