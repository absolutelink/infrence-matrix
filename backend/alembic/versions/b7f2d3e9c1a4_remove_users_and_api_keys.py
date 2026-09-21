"""remove users and api keys

Revision ID: b7f2d3e9c1a4
Revises: ee52a27d7836
Create Date: 2026-09-21 16:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7f2d3e9c1a4'
down_revision: Union[str, Sequence[str], None] = 'ee52a27d7836'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = :name)"
        ),
        {"name": table_name},
    )
    return result.scalar()


def upgrade() -> None:
    """Upgrade schema."""
    # Drop legacy tables no longer used by the application.
    # Use IF EXISTS-style guards since some deployments may not have them.
    if _table_exists("item"):
        op.drop_table("item")
    if _table_exists("api_keys"):
        op.drop_table("api_keys")
    if _table_exists("user"):
        op.drop_index(op.f('ix_user_email'), table_name='user')
        op.drop_table("user")


def downgrade() -> None:
    """Downgrade schema."""
    # Recreating auth tables is not supported; this migration is irreversible.
    raise NotImplementedError("Users/API keys cannot be restored after removal.")