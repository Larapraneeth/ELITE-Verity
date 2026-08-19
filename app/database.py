from collections.abc import Generator
import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL
from app.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    except SQLAlchemyError:
        logger.exception("Database operation failed", extra={"event": "database.operation_failed"})
        raise
    finally:
        session.close()


def init_db() -> None:
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_message_source_column()


def _ensure_message_source_column() -> None:
    """Lightweight in-place migration: add the nullable JSON ``source`` column
    to an existing ``messages`` table if it is missing. This leaves pre-existing
    rows untouched (their ``source`` is NULL), so old history stays valid."""
    from sqlalchemy import inspect, text

    with engine.begin() as connection:
        inspector = inspect(connection)
        try:
            columns = [col["name"] for col in inspector.get_columns("messages")]
        except Exception:
            return
        if "source" not in columns:
            connection.execute(
                text("ALTER TABLE messages ADD COLUMN source JSON")
            )
