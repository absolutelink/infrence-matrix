from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, delete

from app.core.db import engine
from app.main import app
from app.models import AudioJob, BatchJob, Conversation, File, Model, PromptCache


@pytest.fixture(scope="session", autouse=True)
def db() -> Generator[Session]:
    with Session(engine) as session:
        yield session


@pytest.fixture(autouse=True)
def _clean_db() -> None:
    """Remove test data between tests so each test starts from a known state."""
    with Session(engine) as session:
        for model in (PromptCache, AudioJob, BatchJob, Conversation, File, Model):
            session.exec(delete(model))  # type: ignore[arg-type]
        session.commit()


@pytest.fixture(scope="module")
def client() -> Generator[TestClient]:
    with TestClient(app) as c:
        yield c
