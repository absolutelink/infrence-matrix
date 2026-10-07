"""Phase 16 atomic cutover: re-key ProviderInstance onto ProviderAgent

Revision ID: f1b6c2d84a97
Revises: e7a3c5d19b40
Create Date: 2026-10-07 18:00:00.000000

Breaking re-key (the running system moves to the agent model):

  * provider_instances: replace ``machine_id`` with ``agent_id``
    (FK -> provider_agents.id, CASCADE); unique key becomes
    ``(agent_id, provider_definition_id)``. The container-level columns
    (``version``/``instance_status``/``websocket_connected``/``epoch``/
    ``last_seen``/``reported_schema_fingerprint``) move onto
    ``provider_agents`` (added by e7a3c5d19b40) and are dropped here.
  * provider_definitions: DROP ``registration_token``; make
    ``provider_type``/``backend_config`` NOT NULL again (Phase 14 shells
    removed — any shell rows are deleted first).
  * machines: backfill an empty ``registration_secret`` with a minted value.

Data migration: for each existing instance, synthesize one
``ProviderAgent`` ``(machine_id, definition.provider_type,
agent_id='legacy-'+<instance-id-prefix>)`` (PK = the instance id) and
repoint the instance's ``agent_id`` to it. Downgrade reverses the re-key
(re-adding the dropped columns from the agent mirror, regenerating unique
registration tokens, and removing the synthesized agents).

Tested upgrade -> downgrade -> upgrade on a throwaway UTF8 DB.

DOWNGRADE SAFETY: downgrade restores the pre-16 unique constraint
``(machine_id, provider_definition_id)`` on ``provider_instances``. That is
only safe on a freshly-migrated / unused DB. Once the agent model is live a
single machine may host MULTIPLE agents (different ``agent_id``) that each
carry a backend for the SAME definition; collapsing them back onto the
machine-keyed unique row would violate the restored constraint and the
downgrade would fail. Do not downgrade a production DB past this revision.
"""

import secrets
from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f1b6c2d84a97"
down_revision: str | Sequence[str] | None = "e7a3c5d19b40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Retire Phase 14 shells: a definition with no type/config cannot be
    #    re-keyed onto a typed agent and is dropped (cascades to its
    #    instances). On a fresh DB this is a no-op.
    bind.execute(
        sa.text(
            "DELETE FROM provider_definitions "
            "WHERE provider_type IS NULL OR backend_config IS NULL"
        )
    )

    # 2. Add the agent FK column (nullable until backfilled).
    op.add_column(
        "provider_instances",
        sa.Column("agent_id", sa.UUID(), nullable=True),
    )

    # 3. Synthesize one ProviderAgent per existing instance (PK = instance id,
    #    stable agent_id string 'legacy-'+prefix) and repoint the instance.
    bind.execute(
        sa.text(
            """
            INSERT INTO provider_agents (
                id, machine_id, provider_type, agent_id, base_port, version,
                agent_status, websocket_connected, epoch, last_seen,
                reported_schema_fingerprint, assigned_gpus, error_message,
                created_at, updated_at
            )
            SELECT pi.id, pi.machine_id, pd.provider_type,
                   'legacy-' || substr(pi.id::text, 1, 8), pi.port, pi.version,
                   pi.instance_status, pi.websocket_connected, pi.epoch,
                   pi.last_seen, pi.reported_schema_fingerprint,
                   COALESCE(pi.assigned_gpus, '[]'::json), pi.error_message,
                   pi.created_at, pi.updated_at
            FROM provider_instances pi
            JOIN provider_definitions pd ON pd.id = pi.provider_definition_id
            WHERE pd.provider_type IS NOT NULL
            ON CONFLICT DO NOTHING
            """
        )
    )
    bind.execute(
        sa.text("UPDATE provider_instances SET agent_id = id WHERE agent_id IS NULL")
    )

    # 4. Backfill empty machine secrets.
    rows = bind.execute(
        sa.text(
            "SELECT id FROM machines "
            "WHERE registration_secret IS NULL OR registration_secret = ''"
        )
    ).fetchall()
    for (mid,) in rows:
        bind.execute(
            sa.text(
                "UPDATE machines SET registration_secret = :s WHERE id = :id"
            ),
            {"s": secrets.token_urlsafe(32), "id": mid},
        )

    # 5. agent_id NOT NULL + FK (CASCADE).
    op.alter_column("provider_instances", "agent_id", nullable=False)
    op.create_foreign_key(
        "fk_provider_instances_agent_id",
        "provider_instances",
        "provider_agents",
        ["agent_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 6. Drop the old machine-keyed indexes + container-level columns.
    op.drop_index("idx_provider_instances_machine_def", table_name="provider_instances")
    op.drop_index(
        "idx_provider_instances_instance_status", table_name="provider_instances"
    )
    op.drop_column("provider_instances", "machine_id")
    op.drop_column("provider_instances", "version")
    op.drop_column("provider_instances", "instance_status")
    op.drop_column("provider_instances", "websocket_connected")
    op.drop_column("provider_instances", "epoch")
    op.drop_column("provider_instances", "last_seen")
    op.drop_column("provider_instances", "reported_schema_fingerprint")

    # 7. New agent-keyed indexes.
    op.create_index(
        "idx_provider_instances_agent_def",
        "provider_instances",
        ["agent_id", "provider_definition_id"],
        unique=True,
    )
    op.create_index(
        "idx_provider_instances_agent", "provider_instances", ["agent_id"], unique=False
    )

    # 8. provider_definitions: drop registration_token, re-tighten NOT NULLs.
    op.drop_column("provider_definitions", "registration_token")
    op.alter_column(
        "provider_definitions",
        "provider_type",
        existing_type=sqlmodel.sql.sqltypes.AutoString(length=64),
        nullable=False,
    )
    op.alter_column(
        "provider_definitions",
        "backend_config",
        existing_type=postgresql.JSON(astext_type=sa.Text()),
        nullable=False,
        server_default="{}",
    )
    op.alter_column("provider_definitions", "backend_config", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()

    # 8'. provider_definitions: re-allow nulls, re-add unique registration_token.
    op.alter_column(
        "provider_definitions",
        "backend_config",
        existing_type=postgresql.JSON(astext_type=sa.Text()),
        nullable=True,
    )
    op.alter_column(
        "provider_definitions",
        "provider_type",
        existing_type=sqlmodel.sql.sqltypes.AutoString(length=64),
        nullable=True,
    )
    op.add_column(
        "provider_definitions",
        sa.Column(
            "registration_token",
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=True,
        ),
    )
    for (did,) in bind.execute(sa.text("SELECT id FROM provider_definitions")).fetchall():
        bind.execute(
            sa.text(
                "UPDATE provider_definitions SET registration_token = :t WHERE id = :id"
            ),
            {"t": secrets.token_urlsafe(24), "id": did},
        )
    op.alter_column(
        "provider_definitions",
        "registration_token",
        existing_type=sqlmodel.sql.sqltypes.AutoString(length=255),
        nullable=False,
        unique=True,
    )

    # 6'/7'. Drop the agent-keyed indexes + agent_id FK/column.
    op.drop_index("idx_provider_instances_agent", table_name="provider_instances")
    op.drop_index("idx_provider_instances_agent_def", table_name="provider_instances")
    op.drop_constraint(
        "fk_provider_instances_agent_id", "provider_instances", type_="foreignkey"
    )
    op.drop_column("provider_instances", "agent_id")

    # 2'. Re-add the container-level columns on provider_instances and
    #     backfill them from the owning agent (machine_id from the agent).
    op.add_column(
        "provider_instances",
        sa.Column("machine_id", sa.UUID(), nullable=True),
    )
    op.add_column(
        "provider_instances",
        sa.Column(
            "version",
            sqlmodel.sql.sqltypes.AutoString(length=64),
            nullable=False,
            server_default="dev",
        ),
    )
    op.add_column(
        "provider_instances",
        sa.Column(
            "instance_status",
            sqlmodel.sql.sqltypes.AutoString(length=32),
            nullable=False,
            server_default="registering",
        ),
    )
    op.add_column(
        "provider_instances",
        sa.Column(
            "websocket_connected", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )
    op.add_column(
        "provider_instances",
        sa.Column("epoch", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "provider_instances",
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "provider_instances",
        sa.Column(
            "reported_schema_fingerprint",
            sqlmodel.sql.sqltypes.AutoString(length=64),
            nullable=True,
        ),
    )
    bind.execute(
        sa.text(
            """
            UPDATE provider_instances pi
            SET machine_id = pa.machine_id,
                version = pa.version,
                instance_status = pa.agent_status,
                websocket_connected = pa.websocket_connected,
                epoch = pa.epoch,
                last_seen = pa.last_seen,
                reported_schema_fingerprint = pa.reported_schema_fingerprint
            FROM provider_agents pa
            WHERE pa.id = (
                SELECT pa2.id FROM provider_agents pa2
                WHERE pa2.agent_id = 'legacy-' || substr(pi.id::text, 1, 8)
                LIMIT 1
            )
            """
        )
    )
    # Any instance left without a machine (no synthesized agent) falls back to
    # the first machine so the NOT NULL holds on a partially-migrated DB.
    bind.execute(
        sa.text(
            "UPDATE provider_instances SET machine_id = "
            "(SELECT id FROM machines ORDER BY created_at LIMIT 1) "
            "WHERE machine_id IS NULL"
        )
    )
    op.alter_column("provider_instances", "machine_id", nullable=False)
    op.create_foreign_key(
        "provider_instances_machine_id_fkey",
        "provider_instances",
        "machines",
        ["machine_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "idx_provider_instances_machine_def",
        "provider_instances",
        ["machine_id", "provider_definition_id"],
        unique=True,
    )
    op.create_index(
        "idx_provider_instances_instance_status",
        "provider_instances",
        ["instance_status"],
        unique=False,
    )
    for col in ("version", "instance_status", "websocket_connected", "epoch"):
        op.alter_column("provider_instances", col, server_default=None)

    # Remove the agents synthesized by this migration (the table itself stays —
    # it belongs to e7a3c5d19b40).
    bind.execute(
        sa.text("DELETE FROM provider_agents WHERE agent_id LIKE 'legacy-%'")
    )
