"""add typed server options JSON"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9d3e7f1a2b4c"
down_revision: str | Sequence[str] | None = "8f3a1c6d9e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column("server_options", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("server_instances", "server_options")
