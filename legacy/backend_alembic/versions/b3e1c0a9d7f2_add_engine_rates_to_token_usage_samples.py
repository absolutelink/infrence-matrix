"""add engine-reported rates to token usage samples

Revision ID: b3e1c0a9d7f2
Revises: cad923eb5186
Create Date: 2026-09-28 10:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3e1c0a9d7f2'
down_revision: Union[str, Sequence[str], None] = 'cad923eb5186'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'token_usage_samples',
        sa.Column(
            'prompt_per_second',
            sa.Float(),
            nullable=False,
            server_default='0',
        ),
    )
    op.add_column(
        'token_usage_samples',
        sa.Column(
            'predicted_per_second',
            sa.Float(),
            nullable=False,
            server_default='0',
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('token_usage_samples', 'predicted_per_second')
    op.drop_column('token_usage_samples', 'prompt_per_second')
