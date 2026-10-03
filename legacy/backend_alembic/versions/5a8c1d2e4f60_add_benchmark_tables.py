"""add benchmark definitions and runs

Revision ID: 5a8c1d2e4f60
Revises: 3cc6291025fe
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5a8c1d2e4f60"
down_revision: str | Sequence[str] | None = "3cc6291025fe"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "benchmark_definitions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("source_server_instance_id", sa.UUID(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_server_instance_id"],
            ["server_instances.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_benchmark_definitions_name", "benchmark_definitions", ["name"]
    )
    op.create_index(
        "idx_benchmark_definitions_source",
        "benchmark_definitions",
        ["source_server_instance_id"],
    )
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("definition_id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.UUID(), nullable=True),
        sa.Column("server_instance_id", sa.UUID(), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("command", sa.JSON(), nullable=False),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("raw_output", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["definition_id"], ["benchmark_definitions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["server_instance_id"], ["server_instances.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_benchmark_runs_status", "benchmark_runs", ["status"])
    op.create_index(
        "idx_benchmark_runs_definition", "benchmark_runs", ["definition_id"]
    )
    op.create_index(
        "idx_benchmark_runs_created_at", "benchmark_runs", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index("idx_benchmark_runs_created_at", table_name="benchmark_runs")
    op.drop_index("idx_benchmark_runs_definition", table_name="benchmark_runs")
    op.drop_index("idx_benchmark_runs_status", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
    op.drop_index(
        "idx_benchmark_definitions_source", table_name="benchmark_definitions"
    )
    op.drop_index(
        "idx_benchmark_definitions_name", table_name="benchmark_definitions"
    )
    op.drop_table("benchmark_definitions")
