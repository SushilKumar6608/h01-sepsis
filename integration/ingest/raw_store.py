"""
Raw message store
==================
SQLAlchemy ORM model for the raw_messages table.

Schema
------
raw_messages
  id          SERIAL PRIMARY KEY
  source      VARCHAR(32)    -- "hl7v2" or "fhir"
  control_id  VARCHAR(128)   -- MSH-10 or FHIR Bundle.id
  received_at TIMESTAMP      -- UTC wall clock at ingest
  payload     TEXT           -- full raw message string
  sha256      CHAR(64)       -- SHA-256 of payload bytes

Idempotency: UNIQUE(source, control_id) — duplicate messages are
detected before INSERT and the existing row is returned instead.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, Session, mapped_column

from integration.canonical.db import Base


class RawMessage(Base):
    __tablename__ = "raw_messages"
    __table_args__ = (
        UniqueConstraint("source", "control_id", name="uq_source_control_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    control_id: Mapped[str] = mapped_column(String(128), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<RawMessage id={self.id} source={self.source!r} "
            f"control_id={self.control_id!r}>"
        )


# ── store helpers ─────────────────────────────────────────────────────────────

class DuplicateMessageError(Exception):
    """Raised when (source, control_id) already exists."""

    def __init__(self, existing: RawMessage):
        self.existing = existing
        super().__init__(
            f"Duplicate message: source={existing.source!r} "
            f"control_id={existing.control_id!r} id={existing.id}"
        )


def store_raw(db: Session, source: str, control_id: str, payload: str) -> RawMessage:
    """
    Persist a raw message.  If (source, control_id) already exists,
    raises DuplicateMessageError with the existing row — no INSERT.

    Parameters
    ----------
    db         : SQLAlchemy session (from get_db() dependency)
    source     : "hl7v2" or "fhir"
    control_id : message identifier (MSH-10 / Bundle.id)
    payload    : full message text

    Returns
    -------
    RawMessage  (newly created)
    """
    existing = (
        db.query(RawMessage)
        .filter_by(source=source, control_id=control_id)
        .first()
    )
    if existing:
        raise DuplicateMessageError(existing)

    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    row = RawMessage(
        source=source,
        control_id=control_id,
        received_at=datetime.utcnow(),
        payload=payload,
        sha256=sha,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
