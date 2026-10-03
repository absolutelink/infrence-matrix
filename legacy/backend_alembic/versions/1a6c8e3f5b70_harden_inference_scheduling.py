"""Harden inference scheduling state and capacity.

Revision ID: 1a6c8e3f5b70
Revises: 7c1e9f2a4b6d
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "1a6c8e3f5b70"
down_revision: str | Sequence[str] | None = "7c1e9f2a4b6d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "server_instances",
        sa.Column(
            "effective_capacity", sa.Integer(), nullable=False, server_default="1"
        ),
    )
    op.add_column(
        "inference_leases",
        sa.Column("preferred_server_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "inference_leases",
        sa.Column("required_agent_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "inference_leases",
        sa.Column("terminal_reason", sa.String(length=255), nullable=True),
    )
    op.create_foreign_key(
        "fk_inference_leases_preferred_server",
        "inference_leases",
        "server_instances",
        ["preferred_server_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_inference_leases_required_agent",
        "inference_leases",
        "agents",
        ["required_agent_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "idx_inference_leases_queued_model_order",
        "inference_leases",
        ["model_id", "queued_at", "id"],
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.create_index(
        "idx_inference_leases_queued_server_order",
        "inference_leases",
        ["preferred_server_id", "queued_at", "id"],
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.create_index(
        "idx_inference_leases_queued_agent_order",
        "inference_leases",
        ["required_agent_id", "queued_at", "id"],
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.create_check_constraint(
        "ck_inference_leases_status",
        "inference_leases",
        "status IN ('queued', 'active', 'released', 'cancelled', 'expired', 'failed')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_inference_leases_status", "inference_leases", type_="check"
    )
    op.drop_index(
        "idx_inference_leases_queued_agent_order", table_name="inference_leases"
    )
    op.drop_index(
        "idx_inference_leases_queued_server_order", table_name="inference_leases"
    )
    op.drop_index(
        "idx_inference_leases_queued_model_order", table_name="inference_leases"
    )
    op.drop_constraint(
        "fk_inference_leases_required_agent", "inference_leases", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_inference_leases_preferred_server",
        "inference_leases",
        type_="foreignkey",
    )
    op.drop_column("inference_leases", "terminal_reason")
    op.drop_column("inference_leases", "required_agent_id")
    op.drop_column("inference_leases", "preferred_server_id")
    op.drop_column("server_instances", "effective_capacity")
