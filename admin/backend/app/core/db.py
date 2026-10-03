from sqlmodel import Session, create_engine

from app.core.config import settings

engine = create_engine(
    str(settings.DATABASE_URL),
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30.0,
    connect_args={"options": "-c idle_in_transaction_session_timeout=15000"},
)


def get_session() -> Session:
    with Session(engine) as session:
        yield session
