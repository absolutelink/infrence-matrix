"""Add model_type to models

Revision ID: d5e7a1b9c3f2
Revises: b1f8c2a47d90
Create Date: 2026-09-23 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d5e7a1b9c3f2"
down_revision: Union[str, Sequence[str], None] = "b1f8c2a47d90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add model_type column with llm as default for existing rows."""
    op.add_column(
        "models",
        sa.Column("model_type", sa.String(), nullable=False, server_default="llm"),
    )
    op.create_index("idx_models_model_type", "models", ["model_type"])


def downgrade() -> None:
    """Remove model_type column."""
    op.drop_index("idx_models_model_type", table_name="models")
    op.drop_column("models", "model_type")
