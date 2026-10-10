"""Phase 25: multi-model definitions (ProviderModel + ProviderType.multi_model)

Revision ID: d5f9b2c8e041
Revises: c4e8a1f7d902
Create Date: 2026-10-10 12:00:00.000000

Two additive changes for ``docs/multi-model-definitions.md`` §3:

  * ``provider_types.multi_model``: bool (default false), read from the shipped
    schema's top-level ``x-multi-model`` at every schema-commit point (like
    ``max_running_backends`` / ``serves_modalities``). Added with a temporary
    ``false`` server_default so existing rows backfill cleanly, then the default
    is dropped so the column matches the model (which carries only a Python-side
    default) — keeping autogenerate parity.
  * ``provider_models``: one row per client-facing served name on a multi-model
    definition (single-model definitions have NO rows). ``definition_id`` FK →
    ``provider_definitions.id`` ON DELETE CASCADE; ``name`` carries a unique
    index (globally unique across served names; the cross-check against
    ``ProviderDefinition.alias`` is app-enforced). ``backend_config`` is a JSON
    object (per-model engine knobs); ``enabled`` gates routing (disabled names
    404).

Downgrade drops the table + column (nothing depends on them yet).
"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5f9b2c8e041"
down_revision: str | Sequence[str] | None = "c4e8a1f7d902"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "provider_types",
        sa.Column("multi_model", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # Drop the temporary backfill default so the column matches the SQLModel
    # declaration (Python-side default only) — autogenerate parity.
    op.alter_column(
        "provider_types",
        "multi_model",
        existing_type=sa.Boolean(),
        server_default=None,
    )

    op.create_table(
        "provider_models",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("definition_id", sa.Uuid(), nullable=False),
        sa.Column(
            "name", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False
        ),
        sa.Column(
            "modality", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False
        ),
        sa.Column(
            "backend_config",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["definition_id"],
            ["provider_definitions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_provider_models_name", "provider_models", ["name"], unique=True
    )
    op.create_index(
        "idx_provider_models_definition", "provider_models", ["definition_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_provider_models_definition", table_name="provider_models")
    op.drop_index("idx_provider_models_name", table_name="provider_models")
    op.drop_table("provider_models")
    op.drop_column("provider_types", "multi_model")
