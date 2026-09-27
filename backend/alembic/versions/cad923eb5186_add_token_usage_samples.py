"""add token usage samples

Revision ID: cad923eb5186
Revises: 1a6c8e3f5b70
Create Date: 2026-09-27 08:31:28.792052

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cad923eb5186'
down_revision: Union[str, Sequence[str], None] = '1a6c8e3f5b70'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('token_usage_samples',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('server_instance_id', sa.Uuid(), nullable=True),
    sa.Column('agent_id', sa.Uuid(), nullable=True),
    sa.Column('model_id', sa.Uuid(), nullable=True),
    sa.Column('prompt_tokens', sa.Integer(), nullable=False),
    sa.Column('cached_tokens', sa.Integer(), nullable=False),
    sa.Column('completion_tokens', sa.Integer(), nullable=False),
    sa.Column('prompt_ms', sa.Float(), nullable=False),
    sa.Column('predicted_ms', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['server_instance_id'], ['server_instances.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_token_usage_samples_created_at', 'token_usage_samples', ['created_at'], unique=False)
    op.create_index('idx_token_usage_samples_server_created', 'token_usage_samples', ['server_instance_id', 'created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_token_usage_samples_server_created', table_name='token_usage_samples')
    op.drop_index('idx_token_usage_samples_created_at', table_name='token_usage_samples')
    op.drop_table('token_usage_samples')
