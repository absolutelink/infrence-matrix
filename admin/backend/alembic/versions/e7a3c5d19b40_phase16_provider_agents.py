"""Phase 16 machine-scoped provider agents (additive foundation)

Revision ID: e7a3c5d19b40
Revises: d3a9c6e1f842
Create Date: 2026-10-07 12:00:00.000000

Adds the agent model primitives WITHOUT breaking the existing
registration/instance contract (that re-key lands in a follow-on slice):

  * machines.registration_secret        shared agent auth secret
  * provider_types.max_running_backends per-agent running cap (0 = unlimited)
  * provider_definitions.agent_placement 'any_of_type' | 'specific'
  * provider_agents                     container (machine, type, agent_id)
  * definition_agents                   specific-placement link table

All new columns are additive with server defaults so existing rows are valid.
"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7a3c5d19b40"
down_revision: str | Sequence[str] | None = "d3a9c6e1f842"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "machines",
        sa.Column(
            "registration_secret",
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column(
        "provider_types",
        sa.Column(
            "max_running_backends",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "provider_definitions",
        sa.Column(
            "agent_placement",
            sqlmodel.sql.sqltypes.AutoString(length=32),
            nullable=False,
            server_default="any_of_type",
        ),
    )

    op.create_table(
        "provider_agents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("machine_id", sa.UUID(), nullable=False),
        sa.Column(
            "provider_type", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False
        ),
        sa.Column(
            "agent_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False
        ),
        sa.Column("base_port", sa.Integer(), nullable=False),
        sa.Column(
            "version", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False
        ),
        sa.Column(
            "agent_status", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False
        ),
        sa.Column("websocket_connected", sa.Boolean(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "reported_schema_fingerprint",
            sqlmodel.sql.sqltypes.AutoString(length=64),
            nullable=True,
        ),
        sa.Column(
            "assigned_gpus", postgresql.JSON(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["machine_id"], ["machines.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_provider_agents_machine_type_agent",
        "provider_agents",
        ["machine_id", "provider_type", "agent_id"],
        unique=True,
    )
    op.create_index(
        "idx_provider_agents_type", "provider_agents", ["provider_type"], unique=False
    )
    op.create_index(
        "idx_provider_agents_status", "provider_agents", ["agent_status"], unique=False
    )

    op.create_table(
        "definition_agents",
        sa.Column("provider_definition_id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["provider_definition_id"],
            ["provider_definitions.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["agent_id"], ["provider_agents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("provider_definition_id", "agent_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("definition_agents")
    op.drop_index("idx_provider_agents_status", table_name="provider_agents")
    op.drop_index("idx_provider_agents_type", table_name="provider_agents")
    op.drop_table("provider_agents")
    op.drop_column("provider_definitions", "agent_placement")
    op.drop_column("provider_types", "max_running_backends")
    op.drop_column("machines", "registration_secret")
