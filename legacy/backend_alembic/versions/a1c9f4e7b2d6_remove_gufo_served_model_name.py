"""remove served_model_name from gufo engine_options

Revision ID: a1c9f4e7b2d6
Revises: e5b8c2d0f3a1
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1c9f4e7b2d6"
down_revision: str | Sequence[str] | None = "e5b8c2d0f3a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # engine_options is a JSON column; the '-' and '?' operators only exist
    # for jsonb, so cast in both directions.
    op.execute(
        """
        UPDATE server_instances
        SET engine_options = (engine_options::jsonb - 'served_model_name')::json
        WHERE engine = 'gufo'
          AND engine_options::jsonb ? 'served_model_name'
        """
    )


def downgrade() -> None:
    # Removed values cannot be reconstructed; the served model name is now
    # always derived from the instance alias at startup.
    pass
