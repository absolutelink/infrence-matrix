"""Phase 24 migration round-trip + autogenerate parity on a scratch database.

There is no persistent alembic-versioned test DB (the suite builds its schema
with ``SQLModel.metadata.create_all``), so this test provisions a throwaway
UTF8 database, runs the full migration chain to ``head``, asserts the two new
Phase 24 tables exist, downgrades one revision (the Phase 24 migration) and
asserts they are gone, then re-upgrades.

It also runs an **autogenerate-parity** check: with the schema at ``head``,
``alembic.autogenerate.compare_metadata`` against the live SQLModel metadata
must report **no** operations touching the two audio tables — i.e. the
migration reproduces exactly what the models declare. The comparison is scoped
to the audio tables (rather than enabling ``compare_type`` globally in
``env.py``) so a latent type diff on an unrelated pre-Phase-24 table can never
make this test flaky; note that with ``compare_type`` off, column *type* diffs
are not compared — nullability / presence / indexes / constraints are.
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
from app.models import AudioUsageSample, TTSVoice

_SCRATCH_DB = "im_mig_phase24_test"
_AUDIO_TABLES = frozenset({AudioUsageSample.__tablename__, TTSVoice.__tablename__})

# Same default the conftest uses (it reads TEST_DATABASE_URL with this
# fallback but does not write it back to os.environ).
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
        conn.execute(f"DROP DATABASE IF EXISTS {_SCRATCH_DB}")
        conn.execute(
            f"CREATE DATABASE {_SCRATCH_DB} TEMPLATE template0 ENCODING 'UTF8'"
        )
    yield _SCRATCH_DB
    with psycopg.connect(_admin_url("postgres"), autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {_SCRATCH_DB}")


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


def _audio_autogenerate_ops(dbname: str) -> list[tuple]:
    """compare_metadata ops (schema at head vs live models) that touch an audio
    table. Empty == the migration reproduces the model exactly."""
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
        if table_name in _AUDIO_TABLES:
            flagged.append(op)
    return flagged


def test_phase24_migration_roundtrip(scratch_db: str) -> None:
    cfg = _alembic_config(scratch_db)

    command.upgrade(cfg, "head")
    tables = _tables(scratch_db)
    assert AudioUsageSample.__tablename__ in tables
    assert TTSVoice.__tablename__ in tables

    command.downgrade(cfg, "-1")
    tables = _tables(scratch_db)
    assert AudioUsageSample.__tablename__ not in tables
    assert TTSVoice.__tablename__ not in tables
    # The previous head is intact after the single-step downgrade.
    assert "provider_definitions" in tables

    command.upgrade(cfg, "head")
    tables = _tables(scratch_db)
    assert AudioUsageSample.__tablename__ in tables
    assert TTSVoice.__tablename__ in tables


def test_phase24_autogenerate_parity(scratch_db: str) -> None:
    # Ensure the schema is at head (the round-trip test may have left it there,
    # but be independent of test ordering).
    command.upgrade(_alembic_config(scratch_db), "head")
    ops = _audio_autogenerate_ops(scratch_db)
    assert ops == [], f"audio tables drift between migration and models: {ops}"
