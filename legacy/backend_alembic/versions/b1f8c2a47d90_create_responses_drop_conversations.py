"""Create responses table, drop conversations table and prompt_cache.conversation_id

Revision ID: b1f8c2a47d90
Revises: 34a95f86cb0b
Create Date: 2026-09-23

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b1f8c2a47d90"
down_revision: str | None = "34a95f86cb0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "responses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("response_id", sa.String(length=255), nullable=False),
        sa.Column("previous_response_id", sa.String(length=255), nullable=True),
        sa.Column("input_items", postgresql.JSON(astext_type="text"), nullable=True),
        sa.Column("output_items", postgresql.JSON(astext_type="text"), nullable=True),
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id"),
            nullable=True,
        ),
        sa.Column("parameters", postgresql.JSON(astext_type="text"), nullable=True),
        sa.Column(
            "response_metadata", postgresql.JSON(astext_type="text"), nullable=True
        ),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("error_code", sa.String, nullable=True),
        sa.Column("error_message", sa.String, nullable=True),
        sa.Column("incomplete_reason", sa.String, nullable=True),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("total_tokens", sa.Integer, nullable=False),
        sa.Column("store", sa.Boolean, nullable=False),
        sa.Column("background", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )
    op.create_index(
        "idx_responses_response_id", "responses", ["response_id"], unique=True
    )
    op.create_index(
        "idx_responses_previous_response_id", "responses", ["previous_response_id"]
    )
    op.create_index("idx_responses_model_id", "responses", ["model_id"])
    op.create_index("idx_responses_created_at", "responses", ["created_at"])

    op.drop_index("idx_prompt_cache_conversation_id", table_name="prompt_cache")
    op.drop_column("prompt_cache", "conversation_id")

    # Drop conversations after responses exists (conversation data is not migrated;
    # the stub /v1/responses endpoint that created rows was non-functional).
    op.drop_index("idx_conversations_parent_id", table_name="conversations")
    op.drop_index("idx_conversations_model_id", table_name="conversations")
    op.drop_index("idx_conversations_response_id", table_name="conversations")
    op.drop_index("idx_conversations_created_at", table_name="conversations")
    op.drop_table("conversations")


def downgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("response_id", sa.String, nullable=True),
        sa.Column("input_items", postgresql.JSON(astext_type="text"), nullable=True),
        sa.Column("output_items", postgresql.JSON(astext_type="text"), nullable=True),
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("parameters", postgresql.JSON(astext_type="text"), nullable=True),
        sa.Column(
            "conversation_metadata", postgresql.JSON(astext_type="text"), nullable=True
        ),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("error_message", sa.String, nullable=True),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("total_tokens", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )
    op.create_index("idx_conversations_parent_id", "conversations", ["parent_id"])
    op.create_index("idx_conversations_model_id", "conversations", ["model_id"])
    op.create_index("idx_conversations_response_id", "conversations", ["response_id"])
    op.create_index("idx_conversations_created_at", "conversations", ["created_at"])

    op.add_column(
        "prompt_cache",
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "idx_prompt_cache_conversation_id", "prompt_cache", ["conversation_id"]
    )

    op.drop_index("idx_responses_created_at", table_name="responses")
    op.drop_index("idx_responses_model_id", table_name="responses")
    op.drop_index("idx_responses_previous_response_id", table_name="responses")
    op.drop_index("idx_responses_response_id", table_name="responses")
    op.drop_table("responses")
