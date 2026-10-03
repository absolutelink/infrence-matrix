"""Test fixtures.

Tests run against a UTF8 PostgreSQL database (the local dev cluster's default
databases are SQL_ASCII, which makes psycopg3 return bytes and breaks
SQLAlchemy). Override with DATABASE_URL if you point elsewhere.
"""

import os
from collections.abc import Generator

import pytest
from sqlmodel import Session, SQLModel, delete

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres@localhost:5432/inference_matrix_test_utf8",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from app import models  # noqa: E402  (import after DATABASE_URL override)
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
    models.Machine,
]


@pytest.fixture(autouse=True)
def clean_db() -> None:
    for table in _TABLE_ORDER:
        with Session(engine) as session:
            session.exec(delete(table))  # type: ignore[arg-type]
            session.commit()


@pytest.fixture
def session() -> Generator[Session]:
    with Session(engine) as s:
        yield s
