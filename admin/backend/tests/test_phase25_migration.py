"""Phase 25 migration round-trip + autogenerate parity on a scratch database.

Mirrors ``test_phase24_migration.py``: provisions a throwaway UTF8 database,
runs the full chain to ``head``, asserts the ``provider_models`` table and the
``provider_types.multi_model`` column exist, downgrades one revision (the Phase
25 migration) and asserts they are gone, then re-upgrades. Also runs an
autogenerate-parity check scoped to the Phase 25 objects (no drift between the
migration and the SQLModel metadata).
"""

import os
from urllib.parse import urlparse

import psycopg
import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect
from sqlmodel import SQLModel

from alembic import command
from app.models import ProviderModel

_SCRATCH_DB = "im_mig_phase25_test"
_PHASE25_TABLES = frozenset({ProviderModel.__tablename__})

_TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres@localhost:5432/inference_matrix_test_utf8",
)


def _admin_url(dbname: str) -> str:
    parsed = urlparse(_TEST_DATABASE_URL)
    return parsed._replace(scheme="postgresql", path=f"/{dbname}").geturl()


def _sqlalchemy_url(dbname: str) -> str:
    parsed = urlparse(_TEST_DATABASE_URL)
    return parsed._replace(scheme="postgresql+psycopg", path=f"/{dbname}").geturl()


@pytest.fixture(scope="module")
def scratch_db() -> str:
    with psycopg.connect(_admin_url("postgres"), autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {_SCRATCH_DB} WITH (FORCE)")
        conn.execute(
            f"CREATE DATABASE {_SCRATCH_DB} TEMPLATE template0 ENCODING 'UTF8'"
        )
    yield _SCRATCH_DB
    with psycopg.connect(_admin_url("postgres"), autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {_SCRATCH_DB} WITH (FORCE)")


def _alembic_config(dbname: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", "alembic")
    cfg.set_main_option("sqlalchemy.url", _sqlalchemy_url(dbname))
    return cfg


def _tables(dbname: str) -> set[str]:
    engine = create_engine(_sqlalchemy_url(dbname))
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _columns(dbname: str, table: str) -> set[str]:
    engine = create_engine(_sqlalchemy_url(dbname))
    try:
        return {c["name"] for c in inspect(engine).get_columns(table)}
    finally:
        engine.dispose()


def _phase25_autogenerate_ops(dbname: str) -> list[tuple]:
    """compare_metadata ops (schema at head vs live models) touching a Phase 25
    object. Empty == the migration reproduces the model exactly."""
    engine = create_engine(_sqlalchemy_url(dbname))
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            diffs = compare_metadata(ctx, SQLModel.metadata)
    finally:
        engine.dispose()
    flagged: list[tuple] = []
    for op in diffs:
        kind = op[0]
        table_name: str | None
        if kind in ("add_table", "remove_table", "modify_table"):
            table_name = op[1].name
        elif kind in (
            "add_column",
            "remove_column",
            "modify_nullable",
            "modify_default",
            "comment",
        ):
            table_name = op[1]
        elif kind in (
            "create_index",
            "drop_index",
            "create_unique_constraint",
            "drop_unique_constraint",
            "create_check_constraint",
            "drop_check_constraint",
            "create_foreign_key",
            "drop_foreign_key",
        ):
            obj = op[1]
            table = getattr(obj, "table", None)
            table_name = table.name if table is not None else None
        else:
            table_name = None
        if table_name in _PHASE25_TABLES:
            flagged.append(op)
        # The new provider_types.multi_model column.
        if table_name == "provider_types" and kind in (
            "add_column",
            "remove_column",
            "modify_nullable",
            "modify_default",
        ):
            if op[2] == "multi_model":
                flagged.append(op)
    return flagged


def test_phase25_migration_roundtrip(scratch_db: str) -> None:
    cfg = _alembic_config(scratch_db)

    command.upgrade(cfg, "head")
    assert ProviderModel.__tablename__ in _tables(scratch_db)
    assert "multi_model" in _columns(scratch_db, "provider_types")

    command.downgrade(cfg, "-1")
    tables = _tables(scratch_db)
    assert ProviderModel.__tablename__ not in tables
    assert "multi_model" not in _columns(scratch_db, "provider_types")
    # The previous head is intact after the single-step downgrade.
    assert "provider_definitions" in tables

    command.upgrade(cfg, "head")
    assert ProviderModel.__tablename__ in _tables(scratch_db)
    assert "multi_model" in _columns(scratch_db, "provider_types")


def test_phase25_autogenerate_parity(scratch_db: str) -> None:
    command.upgrade(_alembic_config(scratch_db), "head")
    ops = _phase25_autogenerate_ops(scratch_db)
    assert ops == [], f"Phase 25 drift between migration and models: {ops}"
