"""convert legacy gufo cache_disk strings to boolean

Revision ID: b2d0a3f5c7e1
Revises: a1c9f4e7b2d6
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b2d0a3f5c7e1"
down_revision: str | Sequence[str] | None = "a1c9f4e7b2d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # cache_disk changed from a user-entered directory string to a boolean
    # enable flag (the agent derives the path from its own CACHE_PATH).
    # Any stored string means the feature was enabled; convert to true.
    op.execute(
        """
        UPDATE server_instances
        SET engine_options = (
            jsonb_set(engine_options::jsonb, '{cache_disk}', 'true'::jsonb)
        )::json
        WHERE engine = 'gufo'
          AND engine_options IS NOT NULL
          AND jsonb_typeof(engine_options::jsonb -> 'cache_disk') = 'string'
        """
    )


def downgrade() -> None:
    # Booleans cannot be mapped back to the directories users had chosen.
    pass
