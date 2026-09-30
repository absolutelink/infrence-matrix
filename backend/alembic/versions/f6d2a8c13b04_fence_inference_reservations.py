"""Fence pending inference reservations across scheduler processes.

Revision ID: f6d2a8c13b04
Revises: e8a4d1c6f203
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "f6d2a8c13b04"
down_revision: str | Sequence[str] | None = "e8a4d1c6f203"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "inference_leases", sa.Column("reservation_id", postgresql.UUID(as_uuid=True))
    )
    op.add_column(
        "inference_leases",
        sa.Column("reservation_expires_at", sa.DateTime(timezone=True)),
    )
    op.drop_constraint("ck_inference_leases_status", "inference_leases", type_="check")
    op.create_check_constraint(
        "ck_inference_leases_status",
        "inference_leases",
        "status IN ('queued', 'reserving', 'active', 'released', 'cancelled', 'expired', 'failed')",
    )
    op.create_index(
        "idx_inference_leases_reservation_expiry",
        "inference_leases",
        ["reservation_expires_at"],
        postgresql_where=sa.text("status = 'reserving'"),
    )


def downgrade() -> None:
    op.execute(
        "UPDATE inference_leases SET status = 'queued' WHERE status = 'reserving'"
    )
    op.drop_index(
        "idx_inference_leases_reservation_expiry", table_name="inference_leases"
    )
    op.drop_constraint("ck_inference_leases_status", "inference_leases", type_="check")
    op.create_check_constraint(
        "ck_inference_leases_status",
        "inference_leases",
        "status IN ('queued', 'active', 'released', 'cancelled', 'expired', 'failed')",
    )
    op.drop_column("inference_leases", "reservation_expires_at")
    op.drop_column("inference_leases", "reservation_id")
