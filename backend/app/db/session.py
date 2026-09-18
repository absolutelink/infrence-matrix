"""Database session management."""

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.core.config import settings

engine = create_async_engine(str(settings.DATABASE_URL), pool_pre_ping=True, echo=False)

AsyncSessionMaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
