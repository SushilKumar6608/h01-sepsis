"""
Phase 2 - Observation validator.

Takes a MappedMessage (output of mapper.map_message) and checks each
MappedObservation for:
  1. Mapping presence  - observation must have been mapped to a LOINC code.
  2. Value range       - value must be within [min_value, max_value] from the map.
  3. Required fields   - all required local codes must be present.

Returns a ValidationResult that separates observations into:
  - valid        : pass all checks -> go to canonical store
  - quarantined  : fail at least one check -> go to quarantine queue

Design notes:
  - Required-field check fires at the MESSAGE level (not per observation).
    A missing required observation adds a message-level quarantine reason.
  - An unmapped observation is always quarantined (reason: UNMAPPED_CODE).
  - An out-of-range observation is quarantined (reason: OUT_OF_RANGE).
  - A valid-but-unmapped-unit warning is NOT quarantined; it is recorded
    in the observation's reasons list with severity INFO.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from integration.canonical.mapper import MappedMessage, MappedObservation, get_map, get_required_codes

logger = logging.getLogger(__name__)


# ---- result structures ---------------------------------------------------------

@dataclass
class QuarantineReason:
    code: str             # e.g. UNMAPPED_CODE, OUT_OF_RANGE, MISSING_REQUIRED
    detail: str           # human-readable detail
    severity: str = "ERROR"   # ERROR or WARN


@dataclass
class ValidatedObservation:
    observation: MappedObservation
    reasons: list[QuarantineReason] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(r.severity == "ERROR" for r in self.reasons)


@dataclass
class ValidationResult:
    control_id: str
    patient_id: str
    encounter_id: str

    valid: list[ValidatedObservation] = field(default_factory=list)
    quarantined: list[ValidatedObservation] = field(default_factory=list)

    # Message-level issues (e.g. missing required observations)
    message_reasons: list[QuarantineReason] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        """True only if quarantined is empty and no message-level errors exist."""
        has_msg_error = any(r.severity == "ERROR" for r in self.message_reasons)
        return len(self.quarantined) == 0 and not has_msg_error


# ---- validator -----------------------------------------------------------------

def validate_message(mapped: MappedMessage) -> ValidationResult:
    """Validate all observations in a MappedMessage.

    Parameters
    ----------
    mapped : MappedMessage
        Output of mapper.map_message().

    Returns
    -------
    ValidationResult
        Observations split into valid and quarantined lists.
    """
    code_map = get_map()
    required_codes = get_required_codes()

    result = ValidationResult(
        control_id=mapped.control_id,
        patient_id=mapped.patient_id,
        encounter_id=mapped.encounter_id,
    )

    seen_local_codes: set[str] = set()

    for mo in mapped.observations:
        vo = ValidatedObservation(observation=mo)
        seen_local_codes.add(mo.local_code)

        # Check 1: mapping presence
        if not mo.mapped:
            vo.reasons.append(QuarantineReason(
                code="UNMAPPED_CODE",
                detail=f"Local code '{mo.local_code}' has no entry in code_map.yaml",
                severity="ERROR",
            ))
            result.quarantined.append(vo)
            continue

        # Check 2: value range
        entry = code_map.get(mo.local_code)
        if entry is not None and mo.value is not None:
            if mo.value < entry.min_value or mo.value > entry.max_value:
                vo.reasons.append(QuarantineReason(
                    code="OUT_OF_RANGE",
                    detail=(
                        f"{mo.local_code} value {mo.value} is outside "
                        f"[{entry.min_value}, {entry.max_value}]"
                    ),
                    severity="ERROR",
                ))

        # Check 3: unit mismatch (warn only, do not quarantine)
        if entry is not None and mo.unit and mo.unit != entry.unit:
            vo.reasons.append(QuarantineReason(
                code="UNIT_MISMATCH",
                detail=(
                    f"{mo.local_code}: received unit '{mo.unit}', "
                    f"expected '{entry.unit}'"
                ),
                severity="WARN",
            ))

        if vo.is_valid:
            result.valid.append(vo)
        else:
            result.quarantined.append(vo)

    # Check 4: required observations present at message level
    missing_required = required_codes - seen_local_codes
    for code in sorted(missing_required):
        result.message_reasons.append(QuarantineReason(
            code="MISSING_REQUIRED",
            detail=f"Required observation '{code}' is absent from the message",
            severity="ERROR",
        ))

    if result.message_reasons or result.quarantined:
        logger.debug(
            "Validation issues for control_id=%s: %d quarantined obs, "
            "%d message-level reasons",
            mapped.control_id,
            len(result.quarantined),
            len(result.message_reasons),
        )

    return result
