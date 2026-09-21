from sqlmodel import create_engine

from app.core.config import settings

engine = create_engine(
    str(settings.SYNC_DATABASE_URL),
    pool_pre_ping=True,
    connect_args={"client_encoding": "utf8"},
)


# make sure all SQLModel models are imported (app.models) before initializing DB
# otherwise, SQLModel might fail to initialize relationships properly
# for more details: https://github.com/fastapi/full-stack-fastapi-template/issues/28
