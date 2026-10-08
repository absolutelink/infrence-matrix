"""Phase 18: definition modality + type serves_modalities

Revision ID: a1c7e9b24d50
Revises: f1b6c2d84a97
Create Date: 2026-10-08 09:00:00.000000

Additive columns (no behavior change yet — Slice 1):

  * provider_definitions.modality: the endpoint kind a definition serves
    (`llm` default | `embedding`; `audio` reserved). Backfills existing rows
    to `llm`.
  * provider_types.serves_modalities: JSON list of modalities the type can
    host (default ["llm"]), declared via a top-level `x-serves-modalities`
    in the shipped schema.json (mirrors `x-max-running-backends`).

Both are safe to drop on downgrade (new columns; nothing depends on them yet).
"""
from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1c7e9b24d50"
down_revision: str | Sequence[str] | None = "f1b6c2d84a97"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "provider_definitions",
        sa.Column(
            "modality",
            sqlmodel.sql.sqltypes.AutoString(length=32),
            nullable=False,
            server_default="llm",
        ),
    )
    op.add_column(
        "provider_types",
        sa.Column(
            "serves_modalities",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[\"llm\"]'::json"),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("provider_types", "serves_modalities")
    op.drop_column("provider_definitions", "modality")
