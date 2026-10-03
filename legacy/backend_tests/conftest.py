from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, delete

from app.core.db import engine
from app.main import app
from app.models import (
    Agent,
    AudioJob,
    BatchJob,
    BenchmarkDefinition,
    BenchmarkRun,
    File,
    InferenceLease,
    Model,
    PromptCache,
    ResponseRecord,
    ServerInstance,
    TokenUsageSample,
)


@pytest.fixture(scope="session", autouse=True)
def db() -> Generator[Session]:
    with Session(engine) as session:
        yield session


@pytest.fixture(autouse=True)
def _clean_db() -> None:
    """Remove test data between tests so each test starts from a known state."""
    with Session(engine) as session:
        for model in (
            TokenUsageSample,
            PromptCache,
            InferenceLease,
            AudioJob,
            BatchJob,
            BenchmarkRun,
            BenchmarkDefinition,
            ResponseRecord,
            ServerInstance,
            File,
            Model,
            Agent,
        ):
            session.exec(delete(model))  # type: ignore[arg-type]
        session.commit()


@pytest.fixture(scope="session", autouse=True)
def _quiet_metrics_loop() -> Generator[None]:
    """Keep the lifespan metrics snapshot loop from racing gauge assertions.

    The loop reads the interval constant at each cycle; pushing it far out
    means no background snapshot clears/repopulates gauges mid-test.
    Session scope avoids a restore window between tests.
    """
    from app.api.routes import metrics as metrics_module

    metrics_module.METRICS_SNAPSHOT_INTERVAL_SECONDS = 100_000
    yield


@pytest.fixture(scope="module")
def client() -> Generator[TestClient]:
    with TestClient(app) as c:
        yield c
