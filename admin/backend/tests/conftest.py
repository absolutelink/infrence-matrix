"""Test fixtures.

Tests run against a UTF8 PostgreSQL database (the local dev cluster's default
databases are SQL_ASCII, which makes psycopg3 return bytes and breaks
SQLAlchemy). Override with TEST_DATABASE_URL if you point elsewhere.

Settings computes DATABASE_URL from the POSTGRES_* fields, so the test URL
is parsed back into those env vars before the app modules import. Redis for
tests defaults to a dedicated DB index (15) and is flushed per test.
"""

import os
from collections.abc import Generator
from urllib.parse import urlparse

import pytest
import redis as redis_sync
from sqlmodel import Session, SQLModel, delete

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres@localhost:5432/inference_matrix_test_utf8",
)
_parsed = urlparse(TEST_DATABASE_URL)
os.environ["POSTGRES_USER"] = _parsed.username or "postgres"
os.environ["POSTGRES_PASSWORD"] = _parsed.password or ""
os.environ["POSTGRES_HOST"] = _parsed.hostname or "localhost"
os.environ["POSTGRES_PORT"] = str(_parsed.port or 5432)
os.environ["POSTGRES_DB"] = (_parsed.path or "/inference_matrix_test_utf8").lstrip("/")

# Dedicated Redis DB so tests never clobber dev data; overridable.
os.environ.setdefault("TEST_REDIS_URL", "redis://localhost:6379/15")
os.environ["REDIS_URL"] = os.environ["TEST_REDIS_URL"]

# Keep the background presence sweep out of the test process; sweep_once is
# exercised directly where needed.
os.environ["INSTANCE_SWEEP_ENABLED"] = "false"

from app import models  # noqa: E402  (import after env overrides)
from app.core.db import engine  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_schema() -> Generator[None]:
    SQLModel.metadata.create_all(engine)
    yield


_TABLE_ORDER = [
    models.TokenUsageSample,
    models.ResponseRecord,
    models.ProviderInstance,
    models.ProviderDefinition,
    models.ProviderType,
    models.Machine,
]


@pytest.fixture(autouse=True)
def clean_db() -> None:
    for table in _TABLE_ORDER:
        with Session(engine) as session:
            session.exec(delete(table))  # type: ignore[arg-type]
            session.commit()


@pytest.fixture(autouse=True)
def clean_redis() -> Generator[redis_sync.Redis]:
    client = redis_sync.Redis.from_url(
        os.environ["TEST_REDIS_URL"], decode_responses=True
    )
    client.flushdb()
    yield client
    client.close()


@pytest.fixture
def session() -> Generator[Session]:
    with Session(engine) as s:
        yield s
