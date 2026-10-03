"""partial unique port per active instance

Revision ID: 348f7cef7f91
Revises: 3caae8fa9527
Create Date: 2026-09-23 00:35:48.429718

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "348f7cef7f91"
down_revision: str | Sequence[str] | None = "3caae8fa9527"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Replace global unique port with partial unique per active instance."""
    op.drop_constraint("server_instances_port_key", "server_instances", type_="unique")
    op.create_index(
        "uq_server_instances_active_port",
        "server_instances",
        ["agent_id", "port"],
        unique=True,
        postgresql_where=sa.text("status IN ('starting', 'running')"),
    )


def downgrade() -> None:
    """Restore global unique port constraint."""
    op.drop_index("uq_server_instances_active_port", table_name="server_instances")
    op.create_unique_constraint(
        "server_instances_port_key", "server_instances", ["port"]
    )
