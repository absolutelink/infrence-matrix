"""Database session management."""

from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings

engine = create_async_engine(
    str(settings.DATABASE_URL),
    pool_pre_ping=True,
    echo=False,
    connect_args={"client_encoding": "utf8"},
)

AsyncSessionMaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
