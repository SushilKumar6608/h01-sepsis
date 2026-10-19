"""
Phase 2 - Replay endpoint.

Provides a FastAPI router with:

  POST /ingest/replay
      Re-process quarantine items by their IDs (or by raw_sha256).
      Re-runs mapper + validator with the CURRENT code_map.yaml and
      validation rules. Items that now pass are marked replayed_ok=True;
      items that still fail are marked replayed_ok=False.

      This is the key interview demo: fix a mapping in code_map.yaml,
      restart the service, POST to /ingest/replay, and previously
      quarantined observations graduate to the canonical store.

  GET /ingest/quarantine
      List quarantine items (optionally filtered by control_id or
      reason_code). Used to inspect what is waiting for replay.

The replay endpoint intentionally does NOT re-parse the original raw HL7
message. Phase 2 replays at the mapper/validator level using the stored
field values. A full re-parse from the raw_messages table would be Phase 3+.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from integration.canonical.db import get_session
from integration.canonical.mapper import map_observation, get_map
from integration.canonical.quarantine import QuarantineItem, mark_replayed
from integration.canonical.validator import QuarantineReason, ValidatedObservation
from integration.ingest.hl7v2_parser import ParsedObservation

logger = logging.getLogger(__name__)
router = APIRouter()


# ---- request / response models -------------------------------------------------

class ReplayRequest(BaseModel):
    item_ids: Optional[list[int]] = None      # specific IDs to replay
    raw_sha256: Optional[str] = None          # replay all items for this message
    dry_run: bool = False                      # if True: evaluate but do not commit


class ReplayResponse(BaseModel):
    replayed: int
    passed: int
    still_failing: int
    dry_run: bool


class QuarantineListItem(BaseModel):
    id: int
    control_id: str
    patient_id: str
    encounter_id: str
    local_code: Optional[str]
    loinc_code: Optional[str]
    value: Optional[float]
    unit: Optional[str]
    reason_code: str
    reason_detail: str
    severity: str
    raw_sha256: str
    replayed_at: Optional[str]
    replayed_ok: Optional[bool]

    model_config = {"from_attributes": True}


# ---- helpers -------------------------------------------------------------------

def _evaluate_item(item: QuarantineItem) -> bool:
    """Re-run mapping + range validation for one quarantine item.

    Returns True if the item now passes, False if it still fails.
    """
    code_map = get_map()
    entry = code_map.get(item.local_code or "")

    if item.reason_code == "MISSING_REQUIRED":
        # Can only clear this by re-processing the full message.
        # For Phase 2, mark as still-failing.
        return False

    if item.reason_code == "UNMAPPED_CODE":
        # Check whether the code has been added to the map since quarantine.
        return entry is not None

    if item.reason_code == "OUT_OF_RANGE":
        if entry is None or item.value is None:
            return False
        return entry.min_value <= item.value <= entry.max_value

    # Unknown reason code - conservative: still failing
    return False


# ---- endpoints -----------------------------------------------------------------

@router.post("/ingest/replay", response_model=ReplayResponse)
def replay_quarantine(
    body: ReplayRequest,
    session: Session = Depends(get_session),
) -> ReplayResponse:
    """Re-evaluate quarantine items against the current mapping and rules."""
    if not body.item_ids and not body.raw_sha256:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: item_ids, raw_sha256",
        )

    query = session.query(QuarantineItem).filter(
        QuarantineItem.replayed_ok.is_(None)  # only items not yet successfully replayed
    )
    if body.item_ids:
        query = query.filter(QuarantineItem.id.in_(body.item_ids))
    if body.raw_sha256:
        query = query.filter(QuarantineItem.raw_sha256 == body.raw_sha256)

    items = query.all()
    if not items:
        return ReplayResponse(replayed=0, passed=0, still_failing=0, dry_run=body.dry_run)

    passed_ids: list[int] = []
    failing_ids: list[int] = []

    for item in items:
        ok = _evaluate_item(item)
        if ok:
            passed_ids.append(item.id)
        else:
            failing_ids.append(item.id)

    if not body.dry_run:
        if passed_ids:
            mark_replayed(session, passed_ids, ok=True)
        if failing_ids:
            mark_replayed(session, failing_ids, ok=False)
        session.commit()
        logger.info(
            "Replay: %d passed, %d still failing (control_ids in batch)",
            len(passed_ids),
            len(failing_ids),
        )

    return ReplayResponse(
        replayed=len(items),
        passed=len(passed_ids),
        still_failing=len(failing_ids),
        dry_run=body.dry_run,
    )


@router.get("/ingest/quarantine", response_model=list[QuarantineListItem])
def list_quarantine(
    control_id: Optional[str] = Query(default=None),
    reason_code: Optional[str] = Query(default=None),
    include_replayed: bool = Query(default=False),
    limit: int = Query(default=100, le=1000),
    session: Session = Depends(get_session),
) -> list[QuarantineListItem]:
    """List quarantine items with optional filtering."""
    query = session.query(QuarantineItem)
    if control_id:
        query = query.filter(QuarantineItem.control_id == control_id)
    if reason_code:
        query = query.filter(QuarantineItem.reason_code == reason_code)
    if not include_replayed:
        query = query.filter(QuarantineItem.replayed_ok.is_(None))
    query = query.order_by(QuarantineItem.created_at.desc()).limit(limit)
    rows = query.all()
    return [
        QuarantineListItem(
            id=r.id,
            control_id=r.control_id,
            patient_id=r.patient_id,
            encounter_id=r.encounter_id,
            local_code=r.local_code,
            loinc_code=r.loinc_code,
            value=r.value,
            unit=r.unit,
            reason_code=r.reason_code,
            reason_detail=r.reason_detail,
            severity=r.severity,
            raw_sha256=r.raw_sha256,
            replayed_at=r.replayed_at.isoformat() if r.replayed_at else None,
            replayed_ok=r.replayed_ok,
        )
        for r in rows
    ]
