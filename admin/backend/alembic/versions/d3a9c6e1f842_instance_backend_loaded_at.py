"""Add provider_instances.backend_loaded_at (idle-reaper load clock)

Revision ID: d3a9c6e1f842
Revises: b8e4f7d2a9c1
Create Date: 2026-10-07 07:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd3a9c6e1f842'
down_revision: Union[str, Sequence[str], None] = 'b8e4f7d2a9c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    The idle reaper must not stop a freshly loaded backend whose
    ``last_request_at`` predates the boot (or is NULL): the load clock
    starts when the backend enters running/in_use. Additive nullable
    column; existing rows keep NULL and fall back to the previous
    behavior until their next status transition.
    """
    op.add_column(
        'provider_instances',
        sa.Column(
            'backend_loaded_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('provider_instances', 'backend_loaded_at')
