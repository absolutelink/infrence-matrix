"""Add PostgreSQL-backed inference scheduling leases."""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "4c7e9a1b2d3f"
down_revision: str | None = "9f1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "inference_leases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", sa.String(length=255), nullable=False),
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "server_instance_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("server_instances.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_inference_leases_request_id",
        "inference_leases",
        ["request_id"],
        unique=True,
    )
    op.create_index("idx_inference_leases_status", "inference_leases", ["status"])
    op.create_index(
        "idx_inference_leases_server_status",
        "inference_leases",
        ["server_instance_id", "status"],
    )
    op.create_index(
        "idx_inference_leases_expires_at", "inference_leases", ["lease_expires_at"]
    )


def downgrade() -> None:
    op.drop_index("idx_inference_leases_expires_at", table_name="inference_leases")
    op.drop_index("idx_inference_leases_server_status", table_name="inference_leases")
    op.drop_index("idx_inference_leases_status", table_name="inference_leases")
    op.drop_index("ix_inference_leases_request_id", table_name="inference_leases")
    op.drop_table("inference_leases")
