"""
Canonical observation model
============================
Single Pydantic model for one normalised observation after
local-code → LOINC mapping.  Used by the ingest pipeline and
the scoring layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CanonicalObservation(BaseModel):
    """One vital / lab value, mapped to LOINC, ready for the model."""

    # Identifiers
    encounter_id: str
    patient_id: str
    source: str                        # e.g. "hl7v2", "fhir"
    control_id: str                    # original message control ID

    # Observation
    feature_name: str                  # canonical name: HR, O2Sat, …
    loinc_code: str                    # e.g. "8867-4"
    value_raw: float                   # value as received (original units)
    unit_raw: Optional[str] = None     # unit as received (may be None)
    value_canonical: float             # value converted to canonical unit
    unit_canonical: str                # canonical unit from features.yaml

    # Temporal
    observation_ts: datetime
    received_at: datetime = Field(default_factory=datetime.utcnow)

    # Quality
    abnormal_flag: Optional[str] = None
    parse_warnings: list[str] = Field(default_factory=list)

    class Config:
        frozen = True
