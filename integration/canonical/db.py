"""
Database engine and session factory
=====================================
SQLAlchemy 2.0 async-compatible setup.
Phase 1 uses synchronous sessions for simplicity;
the engine URL is read from the DATABASE_URL environment variable
(default: SQLite for local dev without PostgreSQL).

Set DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/h01v2
for production / interview demo.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ── engine ────────────────────────────────────────────────────────────────────

_DEFAULT_URL = "sqlite:///integration/h01v2_dev.db"
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_URL)

# SQLite needs check_same_thread=False because FastAPI's get_db dependency
# closes the session in a worker thread different from the one that opened it.
# This flag is safe here — we never share a single connection across threads;
# each request gets its own session from SessionLocal().
# For PostgreSQL the connect_args kwarg is ignored.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args=_connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── declarative base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── helpers ───────────────────────────────────────────────────────────────────

def get_db():
    """
    FastAPI dependency that yields a database session and closes it
    after the request.

    Usage in an endpoint:
        @router.post("/ingest/hl7v2")
        def ingest(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables (used in tests and initial setup)."""
    # Import models so Base knows about them before create_all
    from integration.ingest.raw_store import RawMessage  # noqa: F401
    Base.metadata.create_all(bind=engine)
