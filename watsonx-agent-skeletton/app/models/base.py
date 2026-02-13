"""SQLAlchemy declarative base, engine, and session factory."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.config import get_database_credentials

_db_url = get_database_credentials().url

_connect_args: dict = {}
if _db_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(_db_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def get_session() -> Session:
    """Create and return a new database session."""
    return SessionLocal()


def init_db() -> None:
    """Create all tables defined by ORM models."""
    Base.metadata.create_all(bind=engine)
