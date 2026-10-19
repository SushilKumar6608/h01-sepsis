"""
Phase 3 - FHIR R4 source parser.

Accepts either:
  - A FHIR R4 Bundle (resourceType="Bundle") containing Observation entries, or
  - A single FHIR R4 Observation resource.

Produces the same ParsedMessage / ParsedObservation types that hl7v2_parser.py
produces, so the mapper, validator, and quarantine pipeline work unchanged
regardless of the source format.

Design notes:
  - Only NM (numeric) value types are extracted; CodeableConcept and string
    values are skipped with a warning (same policy as the HL7v2 parser).
  - LOINC codes are extracted from coding[] where system contains "loinc".
    If no LOINC code is present the raw code text is used as local_code so
    the mapper can quarantine it as UNMAPPED_CODE rather than silently drop it.
  - Patient and encounter IDs are extracted from subject.reference and
    encounter.reference respectively; bare UUIDs are used when the full
    reference is not resolvable in this context.
  - sha256 is computed over the canonical JSON bytes of the input dict so
    the quarantine de-duplication logic works the same way as for HL7v2.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

logger = logging.getLogger(__name__)

LOINC_SYSTEM_SUBSTRING = "loinc"


class FHIRParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_of(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ref_id(reference: Optional[str]) -> str:
    """Extract the id portion of a FHIR reference string.

    'Patient/abc-123' -> 'abc-123'
    'urn:uuid:abc-123' -> 'abc-123'
    'abc-123' -> 'abc-123'
    """
    if not reference:
        return "UNKNOWN"
    for sep in ("/", ":"):
        if sep in reference:
            return reference.rsplit(sep, 1)[-1].strip()
    return reference.strip()


def _loinc_code(coding_list: list[dict]) -> Optional[str]:
    """Return the first LOINC code found in a coding array, or None."""
    for coding in coding_list:
        system = coding.get("system", "")
        if LOINC_SYSTEM_SUBSTRING in system.lower():
            return coding.get("code")
    return None


def _display(coding_list: list[dict]) -> str:
    for coding in coding_list:
        d = coding.get("display", "")
        if d:
            return d
    return ""


def _parse_observation(
    obs: dict,
    seq: int,
    fallback_ts: str,
    warnings: list[str],
) -> Optional[ParsedObservation]:
    """Parse one FHIR Observation dict into a ParsedObservation.

    Returns None if the observation has no usable numeric value (with a
    warning appended).
    """
    coding_list: list[dict] = obs.get("code", {}).get("coding", [])
    loinc = _loinc_code(coding_list)
    display = _display(coding_list)

    # Use LOINC code as local_code so the mapper can look it up directly.
    # If no LOINC is present, fall back to the first code.code value so
    # the mapper quarantines it as UNMAPPED_CODE rather than silently dropping.
    if loinc:
        local_code = loinc
    else:
        fallback_code = coding_list[0].get("code", "") if coding_list else ""
        local_code = fallback_code or "UNKNOWN_CODE"
        warnings.append(
            f"Observation seq={seq}: no LOINC code found; using '{local_code}'"
        )

    # Effective timestamp
    obs_ts = (
        obs.get("effectiveDateTime")
        or obs.get("effectivePeriod", {}).get("start")
        or fallback_ts
    )

    # Value - only numeric (valueQuantity) supported in Phase 3
    if "valueQuantity" in obs:
        vq = obs["valueQuantity"]
        raw_val = vq.get("value")
        unit = vq.get("unit") or vq.get("code") or None
        if raw_val is None:
            warnings.append(f"Observation seq={seq} {local_code}: valueQuantity.value is null - skipping")
            return None
        try:
            value = float(raw_val)
        except (TypeError, ValueError):
            warnings.append(
                f"Observation seq={seq} {local_code}: cannot parse value '{raw_val}' as float - skipping"
            )
            return None
    else:
        # valueString, valueBoolean, valueCodeableConcept etc. - not supported
        vtype = next((k for k in obs if k.startswith("value")), "none")
        warnings.append(
            f"Observation seq={seq} {local_code}: value type '{vtype}' is not numeric - skipping"
        )
        return None

    return ParsedObservation(
        seq=seq,
        local_code=local_code,
        description=display,
        value=value,
        unit=unit,
        abnormal_flag=obs.get("interpretation", [{}])[0]
            .get("coding", [{}])[0]
            .get("code") if obs.get("interpretation") else None,
        observation_ts=obs_ts,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_bundle(bundle: dict) -> ParsedMessage:
    """Parse a FHIR R4 Bundle containing Observation resources.

    The bundle is expected to be a dict (already JSON-decoded).
    Raises FHIRParseError on structural problems.
    """
    if bundle.get("resourceType") != "Bundle":
        raise FHIRParseError(
            f"Expected resourceType='Bundle', got '{bundle.get('resourceType')}'"
        )

    sha = _sha256_of(bundle)
    warnings: list[str] = []

    # Pull metadata from the first Patient/Encounter reference we find
    patient_id = "UNKNOWN"
    encounter_id = "UNKNOWN"
    message_ts = bundle.get("timestamp", "")
    control_id = bundle.get("id", sha[:16])

    observations: list[ParsedObservation] = []
    seq = 1

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue

        # Extract patient / encounter from first Observation we see
        if patient_id == "UNKNOWN":
            patient_id = _ref_id(resource.get("subject", {}).get("reference"))
        if encounter_id == "UNKNOWN":
            encounter_id = _ref_id(resource.get("encounter", {}).get("reference"))

        obs_ts = (
            resource.get("effectiveDateTime")
            or resource.get("effectivePeriod", {}).get("start")
            or message_ts
        )

        parsed_obs = _parse_observation(resource, seq, obs_ts, warnings)
        if parsed_obs is not None:
            observations.append(parsed_obs)
            seq += 1

    if not observations:
        warnings.append("No numeric Observation resources found in Bundle")

    if encounter_id == "UNKNOWN":
        encounter_id = f"ENC_{patient_id}"

    msg = ParsedMessage(
        control_id=control_id,
        sending_application="FHIR-R4",
        sending_facility="FHIR-R4",
        patient_id=patient_id,
        encounter_id=encounter_id,
        message_ts=message_ts,
        observation_ts=message_ts,
        observations=observations,
        extra_segments=[],
        sha256=sha,
        parse_warnings=warnings,
    )

    if warnings:
        logger.debug("FHIR parse warnings for id=%s: %s", control_id, "; ".join(warnings))

    return msg


def parse_observation_resource(obs: dict, control_id: Optional[str] = None) -> ParsedMessage:
    """Parse a single FHIR R4 Observation resource dict.

    Useful when an EHR posts individual Observation resources rather than
    a full Bundle.
    """
    if obs.get("resourceType") != "Observation":
        raise FHIRParseError(
            f"Expected resourceType='Observation', got '{obs.get('resourceType')}'"
        )

    sha = _sha256_of(obs)
    warnings: list[str] = []

    patient_id = _ref_id(obs.get("subject", {}).get("reference"))
    encounter_id = _ref_id(obs.get("encounter", {}).get("reference"))
    if encounter_id == "UNKNOWN":
        encounter_id = f"ENC_{patient_id}"

    obs_ts = (
        obs.get("effectiveDateTime")
        or obs.get("effectivePeriod", {}).get("start")
        or ""
    )
    cid = control_id or obs.get("id", sha[:16])

    parsed_obs = _parse_observation(obs, seq=1, fallback_ts=obs_ts, warnings=warnings)
    observations = [parsed_obs] if parsed_obs is not None else []

    msg = ParsedMessage(
        control_id=cid,
        sending_application="FHIR-R4",
        sending_facility="FHIR-R4",
        patient_id=patient_id,
        encounter_id=encounter_id,
        message_ts=obs_ts,
        observation_ts=obs_ts,
        observations=observations,
        extra_segments=[],
        sha256=sha,
        parse_warnings=warnings,
    )

    if warnings:
        logger.debug("FHIR parse warnings for id=%s: %s", cid, "; ".join(warnings))

    return msg
