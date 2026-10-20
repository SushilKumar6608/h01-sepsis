"""
FastAPI ingest service - Phase 1 + Phase 4
===========================================
Exposes:
  POST /ingest/hl7v2      Accept one HL7v2 ORU^R01 message (plain text body)
  GET  /health            Liveness probe
  POST /score             Run sepsis scoring for an already-ingested message

Run (from repo root, Anaconda Prompt):
    cd D:\\Egna projekt\\h01-sepsis
    set DATABASE_URL=sqlite:///integration/h01v2_dev.db
    python -m uvicorn integration.ingest.api:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from integration.canonical.db import get_db, init_db
from integration.canonical.mapper import map_message
from integration.ingest.hl7v2_parser import HL7v2ParseError, parse_message
from integration.ingest.raw_store import DuplicateMessageError, RawMessage, store_raw
from integration.scoring.fhir_risk_assessment import build_risk_assessment
from integration.scoring.scorer import score_observations

logger = logging.getLogger(__name__)

app = FastAPI(
    title="H01 v2 Ingest Service",
    description="HL7v2 / FHIR ingestion layer for the sepsis early-warning model",
    version="0.1.0",
)


# -- startup ------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    """Create tables if they don't exist (dev / test convenience)."""
    init_db()
    logger.info("Database initialised.")


# -- endpoints ----------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/hl7v2", status_code=status.HTTP_201_CREATED)
async def ingest_hl7v2(request: Request, db: Session = Depends(get_db)):
    """
    Ingest one HL7v2 ORU^R01 message.

    Body
    ----
    Content-Type: text/plain  (raw HL7v2 message string)

    Returns
    -------
    201  { "status": "accepted",  "control_id": "...", "raw_message_id": N,
           "patient_id": "...",   "observations": K,
           "warnings": [...] }
    409  { "detail": "Duplicate", "control_id": "...", "existing_id": N }
    422  { "detail": "Parse error: ..." }
    """
    body_bytes = await request.body()
    raw = body_bytes.decode("utf-8", errors="replace")

    # 1. Parse
    try:
        parsed = parse_message(raw)
    except HL7v2ParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Parse error: {exc}",
        )

    # 2. Persist raw message (idempotency check)
    try:
        row = store_raw(
            db=db,
            source="hl7v2",
            control_id=parsed.control_id,
            payload=raw,
        )
    except DuplicateMessageError as exc:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": "Duplicate",
                "control_id": parsed.control_id,
                "existing_id": exc.existing.id,
            },
        )

    logger.info(
        "Stored hl7v2 message control_id=%s raw_id=%d observations=%d",
        parsed.control_id,
        row.id,
        len(parsed.observations),
    )

    return {
        "status": "accepted",
        "control_id": parsed.control_id,
        "raw_message_id": row.id,
        "patient_id": parsed.patient_id,
        "observations": len(parsed.observations),
        "warnings": parsed.parse_warnings,
    }


@app.post("/score", status_code=status.HTTP_200_OK)
def score(control_id: str, db: Session = Depends(get_db)):
    """
    Run the sepsis scoring pipeline for an already-ingested HL7v2 message.

    The caller supplies the control_id returned by POST /ingest/hl7v2.
    Steps:
      1. Look up the raw HL7v2 text stored in raw_messages.
      2. Re-parse it to get a ParsedMessage.
      3. Map observations to canonical LOINC codes via map_message().
      4. Feed the feature sequence through the LSTM Autoencoder.
      5. Return a FHIR R4 RiskAssessment JSON resource.

    Returns
    -------
    200  FHIR R4 RiskAssessment dict
    404  control_id not found
    422  Raw message fails re-parse (should not happen in practice)
    """
    row: RawMessage | None = (
        db.query(RawMessage)
        .filter_by(source="hl7v2", control_id=control_id)
        .first()
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"control_id '{control_id}' not found",
        )

    try:
        parsed = parse_message(row.payload)
    except HL7v2ParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Re-parse error: {exc}",
        )

    mapped = map_message(parsed)
    risk_score = score_observations(mapped.observations)
    risk_assessment = build_risk_assessment(
        score=risk_score,
        patient_id=parsed.patient_id or control_id,
        encounter_id=control_id,
    )

    logger.info(
        "Scored control_id=%s risk=%s error=%.4f",
        control_id,
        risk_score.risk_level,
        risk_score.reconstruction_error,
    )

    return risk_assessment
