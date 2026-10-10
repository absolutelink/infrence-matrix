"""Phase 24: audio usage samples + TTS voices

Revision ID: c4e8a1f7d902
Revises: b7d3f0a1c9e2
Create Date: 2026-10-10 09:00:00.000000

Two additive tables for the audio (talkies) data plane:

  * ``audio_usage_samples``: per-request audio telemetry (chars + nullable
    clip seconds) — the sibling of ``token_usage_samples`` for the ``tts`` /
    ``asr`` modalities, which have no tokens. Same nullable instance /
    definition FK context (SET NULL) and timezone-aware ``created_at``.
  * ``tts_voices``: the admin-side UI catalog of saved cloned voices scoped
    to a ``tts`` ``ProviderDefinition`` (the clips themselves live on the
    agent's disk). Unique on ``(definition_id, name)``.

Both are new tables; downgrade drops them (nothing depends on them yet).
"""
from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e8a1f7d902"
down_revision: str | Sequence[str] | None = "b7d3f0a1c9e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "audio_usage_samples",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("provider_instance_id", sa.Uuid(), nullable=True),
        sa.Column("provider_definition_id", sa.Uuid(), nullable=True),
        sa.Column(
            "modality", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False
        ),
        sa.Column("chars", sa.Integer(), nullable=False),
        sa.Column("audio_seconds", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["provider_instance_id"],
            ["provider_instances.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["provider_definition_id"],
            ["provider_definitions.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_audio_usage_samples_instance_created",
        "audio_usage_samples",
        ["provider_instance_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "idx_audio_usage_samples_created_at",
        "audio_usage_samples",
        ["created_at"],
        unique=False,
    )

    op.create_table(
        "tts_voices",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("definition_id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
        sa.Column("ref_text", sa.Text(), nullable=True),
        sa.Column(
            "language", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["definition_id"],
            ["provider_definitions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "definition_id", "name", name="uq_tts_voices_definition_name"
        ),
    )
    op.create_index(
        "idx_tts_voices_definition", "tts_voices", ["definition_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_tts_voices_definition", table_name="tts_voices")
    op.drop_table("tts_voices")
    op.drop_index("idx_audio_usage_samples_created_at", table_name="audio_usage_samples")
    op.drop_index(
        "idx_audio_usage_samples_instance_created", table_name="audio_usage_samples"
    )
    op.drop_table("audio_usage_samples")
