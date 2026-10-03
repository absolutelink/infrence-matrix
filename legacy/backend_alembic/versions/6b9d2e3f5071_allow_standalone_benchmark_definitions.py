"""allow benchmark definitions without a source server

Revision ID: 6b9d2e3f5071
Revises: 5a8c1d2e4f60
"""

from collections.abc import Sequence

from alembic import op

revision: str = "6b9d2e3f5071"
down_revision: str | Sequence[str] | None = "5a8c1d2e4f60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "benchmark_definitions_source_server_instance_id_fkey",
        "benchmark_definitions",
        type_="foreignkey",
    )
    op.alter_column(
        "benchmark_definitions",
        "source_server_instance_id",
        nullable=True,
    )
    op.create_foreign_key(
        "benchmark_definitions_source_server_instance_id_fkey",
        "benchmark_definitions",
        "server_instances",
        ["source_server_instance_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "benchmark_definitions_source_server_instance_id_fkey",
        "benchmark_definitions",
        type_="foreignkey",
    )
    op.alter_column(
        "benchmark_definitions",
        "source_server_instance_id",
        nullable=False,
    )
    op.create_foreign_key(
        "benchmark_definitions_source_server_instance_id_fkey",
        "benchmark_definitions",
        "server_instances",
        ["source_server_instance_id"],
        ["id"],
        ondelete="CASCADE",
    )
