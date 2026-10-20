"""
fhir_risk_assessment.py
Wraps a SepsisRiskScore into a FHIR R4 RiskAssessment resource.

The RiskAssessment is returned as a plain Python dict (JSON-serialisable).
The caller (API layer) serialises it with json.dumps / FastAPI's JSONResponse.

FHIR R4 RiskAssessment structure used:
  resourceType      : "RiskAssessment"
  id                : encounter_id
  status            : "final"
  subject           : { reference: "Patient/<patient_id>" }
  encounter         : { reference: "Encounter/<encounter_id>" }
  occurrenceDateTime: ISO-8601 timestamp
  basis             : [ { reference: "Device/lstm-ae-sepsis-v1" } ]
  prediction[0]:
    outcome         : { text: "Sepsis risk - <LEVEL>" }
    probabilityDecimal : reconstruction error (float, 4 d.p.)
    rationale       : "LSTM Autoencoder reconstruction error; threshold LOW<0.10 MODERATE<0.25 HIGH>=0.25"
  note[0]           : { text: "Features used: <n>; features imputed: <n>; model: <version>" }
"""
from __future__ import annotations

from datetime import datetime, timezone

from integration.scoring.scorer import SepsisRiskScore


def build_risk_assessment(
    score: SepsisRiskScore,
    patient_id: str,
    encounter_id: str,
    occurred_at: datetime | None = None,
) -> dict:
    """
    Build a FHIR R4 RiskAssessment dict from a SepsisRiskScore.

    Parameters
    ----------
    score        : SepsisRiskScore returned by scorer.score_observations()
    patient_id   : patient identifier (used in subject.reference)
    encounter_id : encounter / control_id (used as resource id and
                   encounter.reference)
    occurred_at  : timestamp for occurrenceDateTime; defaults to now (UTC)

    Returns
    -------
    dict - FHIR R4 RiskAssessment, ready for json.dumps()
    """
    if occurred_at is None:
        occurred_at = datetime.now(tz=timezone.utc)

    return {
        "resourceType": "RiskAssessment",
        "id": encounter_id,
        "status": "final",
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "encounter": {
            "reference": f"Encounter/{encounter_id}",
        },
        "occurrenceDateTime": occurred_at.isoformat(),
        "basis": [
            {
                "reference": f"Device/lstm-ae-sepsis-{score.model_version}",
            }
        ],
        "prediction": [
            {
                "outcome": {
                    "text": f"Sepsis risk - {score.risk_level}",
                },
                "probabilityDecimal": round(score.reconstruction_error, 4),
                "rationale": (
                    "LSTM Autoencoder reconstruction error; "
                    "thresholds: LOW < 0.10, MODERATE 0.10-0.25, HIGH >= 0.25"
                ),
            }
        ],
        "note": [
            {
                "text": (
                    f"Features used: {score.n_observations_used}; "
                    f"features imputed: {score.n_features_imputed}; "
                    f"model: {score.model_version}"
                )
            }
        ],
    }
