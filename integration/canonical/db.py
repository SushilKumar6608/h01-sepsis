"""
Canonical store - database engine and session factory.

DATABASE_URL is read from the environment; defaults to SQLite for local
development and testing.  Set DATABASE_URL=postgresql+psycopg2://... for
production.

Phase 2 additions:
  - get_session() FastAPI dependency for replay/quarantine endpoints.

Bugfix (post-Phase-3):
  - Restored Base (DeclarativeBase) and init_db() that were dropped in the
    Phase 2 rewrite.  test_phase1.py imports init_db from here; the Phase 2
    quarantine.py also imports Base from here so all ORM models share one
    metadata registry.
"""
from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

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


# Shared declarative base - all ORM models must inherit from this so that
# init_db() creates every table in a single create_all() call.
class Base(DeclarativeBase):
    pass


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a database session, then close it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Alias used by Phase 1 API and tests
get_db = get_session


def init_db() -> None:
    """Create all tables defined across the project.

    Imports every ORM module so Base.metadata is populated before
    create_all() runs.  Safe to call multiple times (DDL is idempotent
    with checkfirst=True implied by create_all).
    """
    # Import ORM models so their tables are registered on Base.metadata
    from integration.canonical.quarantine import QuarantineItem  # noqa: F401
    from integration.canonical.mapping_registry import MappingRegistryEntry  # noqa: F401
    Base.metadata.create_all(bind=engine)
