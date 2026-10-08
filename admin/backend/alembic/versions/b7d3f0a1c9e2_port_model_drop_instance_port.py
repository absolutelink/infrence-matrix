"""Port model overhaul: drop ProviderInstance.port

Revision ID: b7d3f0a1c9e2
Revises: a1c7e9b24d50
Create Date: 2026-10-08 12:00:00.000000

Each provider agent now publishes exactly ONE admin-facing HTTP port (its
container env ``PROVIDER_PORT``, recorded on ``ProviderAgent.base_port``) and
routes ``/v1`` to the correct backend by the request's ``model`` (= the
definition alias). The admin no longer allocates or polices per-backend ports,
so the ``provider_instances.port`` column is dropped.

Downgrade re-adds the column with a ``8081`` server default (then clears the
default) so the NOT NULL holds on existing rows.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d3f0a1c9e2"
down_revision: str | Sequence[str] | None = "a1c7e9b24d50"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("provider_instances", "port")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "provider_instances",
        sa.Column("port", sa.Integer(), nullable=False, server_default="8081"),
    )
    op.alter_column("provider_instances", "port", server_default=None)
