"""
Canonical store - database engine and session factory.

DATABASE_URL is read from the environment; defaults to SQLite for local
development and testing.  Set DATABASE_URL=postgresql+psycopg2://... for
production.

Phase 2 additions:
  - get_session() FastAPI dependency for replay/quarantine endpoints.
"""
from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///integration/canonical/dev.db",
)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args=_connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a database session, then close it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
