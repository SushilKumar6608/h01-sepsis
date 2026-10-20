"""
Phase 2 - Quarantine store: ORM model and helper functions.

Quarantined items are observations (or whole messages) that failed
validation. They are stored with enough context to replay them once a
mapping or validation rule has been fixed.

Table: quarantine_items
  - id             : auto PK
  - control_id     : from the original HL7 message
  - patient_id     : from PID segment
  - encounter_id   : from PV1 segment
  - local_code     : the OBX local code that failed
  - loinc_code     : mapped LOINC code, or NULL if unmapped
  - value          : the numeric OBX value
  - unit           : the unit string from OBX
  - reason_code    : e.g. UNMAPPED_CODE, OUT_OF_RANGE, MISSING_REQUIRED
  - reason_detail  : human-readable explanation
  - raw_sha256     : SHA-256 of the original raw HL7 message (for lookup)
  - mapping_version: version string of the code_map.yaml used (FK to
                     mapping_registry.version)
  - created_at     : insert timestamp
  - replayed_at    : NULL until successfully replayed
  - replayed_ok    : NULL / True / False

Design: the quarantine table is separate from raw_messages and the
canonical store. A replayed item that passes validation moves to the
canonical store; its replayed_at and replayed_ok columns are updated.

Bugfix (post-Phase-3): Base is now imported from integration.canonical.db
so all ORM models share one metadata registry and init_db() can create
all tables in a single create_all() call.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, Session

from integration.canonical.db import Base
from integration.canonical.validator import ValidationResult, ValidatedObservation

logger = logging.getLogger(__name__)


class QuarantineItem(Base):
    __tablename__ = "quarantine_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    control_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patient_id: Mapped[str] = mapped_column(String(64), nullable=False)
    encounter_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # Observation-level fields (NULL for message-level reasons)
    local_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    loinc_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Reason
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_detail: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(8), nullable=False, default="ERROR")

    # Linkback to raw message
    raw_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Which mapping version produced this quarantine entry (NULL = pre-registry)
    mapping_version: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("mapping_registry.version", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    replayed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    replayed_ok: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, default=None)


# ---- helper functions ----------------------------------------------------------

def store_quarantine_items(
    session: Session,
    result: ValidationResult,
    raw_sha256: str,
    mapping_version: Optional[str] = None,
) -> list[QuarantineItem]:
    """Persist all ERROR-severity quarantine reasons from a ValidationResult.

    Inserts one row per:
      - quarantined observation (per-observation error reasons)
      - message-level reason (MISSING_REQUIRED, etc.)

    Parameters
    ----------
    session : Session
    result : ValidationResult
    raw_sha256 : str
        SHA-256 of the original raw message.
    mapping_version : str, optional
        Version string from the mapping registry that was active when this
        message was processed.  Stored for experiment tracking.

    Returns
    -------
    list[QuarantineItem]
        The inserted objects (already flushed, not committed).
    """
    items: list[QuarantineItem] = []

    # Per-observation errors
    for vo in result.quarantined:
        for reason in vo.reasons:
            if reason.severity != "ERROR":
                continue
            item = QuarantineItem(
                control_id=result.control_id,
                patient_id=result.patient_id,
                encounter_id=result.encounter_id,
                local_code=vo.observation.local_code,
                loinc_code=vo.observation.loinc_code,
                value=vo.observation.value,
                unit=vo.observation.unit,
                reason_code=reason.code,
                reason_detail=reason.detail,
                severity=reason.severity,
                raw_sha256=raw_sha256,
                mapping_version=mapping_version,
            )
            session.add(item)
            items.append(item)

    # Message-level errors
    for reason in result.message_reasons:
        if reason.severity != "ERROR":
            continue
        item = QuarantineItem(
            control_id=result.control_id,
            patient_id=result.patient_id,
            encounter_id=result.encounter_id,
            local_code=None,
            loinc_code=None,
            value=None,
            unit=None,
            reason_code=reason.code,
            reason_detail=reason.detail,
            severity=reason.severity,
            raw_sha256=raw_sha256,
            mapping_version=mapping_version,
        )
        session.add(item)
        items.append(item)

    session.flush()
    logger.debug(
        "Quarantined %d item(s) for control_id=%s (mapping_version=%s)",
        len(items),
        result.control_id,
        mapping_version,
    )
    return items


def mark_replayed(
    session: Session,
    item_ids: list[int],
    ok: bool,
) -> int:
    """Mark quarantine items as replayed.

    Parameters
    ----------
    session : Session
    item_ids : list[int]
        PKs to update.
    ok : bool
        True if replay passed validation; False if still failing.

    Returns
    -------
    int
        Number of rows updated.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    updated = (
        session.query(QuarantineItem)
        .filter(QuarantineItem.id.in_(item_ids))
        .all()
    )
    for item in updated:
        item.replayed_at = now
        item.replayed_ok = ok
    session.flush()
    return len(updated)
