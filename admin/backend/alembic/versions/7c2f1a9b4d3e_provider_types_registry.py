"""provider_types registry + reported_schema_fingerprint (Phase 12)

Revision ID: 7c2f1a9b4d3e
Revises: 0140d6ad9f48
Create Date: 2026-10-06 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7c2f1a9b4d3e'
down_revision: Union[str, Sequence[str], None] = '0140d6ad9f48'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('provider_types',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
    sa.Column('schema', postgresql.JSON(astext_type=sa.Text()), nullable=False),
    sa.Column('schema_fingerprint', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
    sa.Column('pending_schema', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('pending_fingerprint', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=True),
    sa.Column('pending_voters', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('status', sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.add_column('provider_instances', sa.Column('reported_schema_fingerprint', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('provider_instances', 'reported_schema_fingerprint')
    op.drop_table('provider_types')
