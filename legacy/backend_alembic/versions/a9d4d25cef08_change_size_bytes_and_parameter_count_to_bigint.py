"""change size_bytes and parameter_count to BigInteger

Revision ID: a9d4d25cef08
Revises: c0c91c9ed06d
Create Date: 2026-09-19 19:41:05.434571

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a9d4d25cef08'
down_revision: Union[str, Sequence[str], None] = 'c0c91c9ed06d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Change size_bytes and parameter_count from Integer to BigInteger
    op.alter_column('models', 'size_bytes',
               existing_type=sa.Integer(),
               type_=sa.BigInteger(),
               existing_nullable=False)
    op.alter_column('models', 'parameter_count',
               existing_type=sa.Integer(),
               type_=sa.BigInteger(),
               existing_nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    # Revert BigInteger back to Integer
    op.alter_column('models', 'size_bytes',
               existing_type=sa.BigInteger(),
               type_=sa.Integer(),
               existing_nullable=False)
    op.alter_column('models', 'parameter_count',
               existing_type=sa.BigInteger(),
               type_=sa.Integer(),
               existing_nullable=True)
