"""Phase 14: shell definitions — nullable provider_type + backend_config

Revision ID: b8e4f7d2a9c1
Revises: 7c2f1a9b4d3e
Create Date: 2026-10-06 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b8e4f7d2a9c1'
down_revision: Union[str, Sequence[str], None] = '7c2f1a9b4d3e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Phase 14: a ProviderDefinition may be created as a "shell" — provider_type
    and backend_config NULL. The type is adopted from the container at first
    registration and the config is authored via the UI afterwards (the
    definition stays in the awaiting_config pre-state, unschedulable, until
    then). Existing rows (typed + configured) are untouched by this change.
    """
    op.alter_column(
        'provider_definitions', 'provider_type',
        existing_type=sa.String(length=64), nullable=True,
    )
    op.alter_column(
        'provider_definitions', 'backend_config',
        existing_type=postgresql.JSON(astext_type=sa.Text()), nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema.

    Refuse to downgrade when shell rows exist (they would violate NOT NULL);
    shells must be deleted (typeless configs cannot map back to the old
    semantics) before downgrading.
    """
    conn = op.get_bind()
    shells = conn.execute(
        sa.text(
            "SELECT count(*) FROM provider_definitions "
            "WHERE provider_type IS NULL OR backend_config IS NULL"
        )
    ).scalar_one()
    if shells:
        raise RuntimeError(
            f"cannot downgrade: {shells} shell definition(s) exist; "
            "configure or delete them first"
        )
    op.alter_column(
        'provider_definitions', 'backend_config',
        existing_type=postgresql.JSON(astext_type=sa.Text()), nullable=False,
    )
    op.alter_column(
        'provider_definitions', 'provider_type',
        existing_type=sa.String(length=64), nullable=False,
    )