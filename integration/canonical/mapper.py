"""
Phase 2 - Code mapper: local Hospital codes -> canonical LOINC codes.

Loads code_map.yaml from the same directory and provides:
  - map_observation()  : map one ParsedObservation to a MappedObservation
  - map_message()      : map all observations in a ParsedMessage
  - get_required_codes(): return the set of required local codes

Design notes:
  - Unknown local codes are NOT silently dropped; they return a MappedObservation
    with mapped=False so the caller (validator / quarantine) can decide.
  - The YAML file is loaded once at import time (module-level singleton).
    Tests that need different mappings can call reload_map().
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

import yaml

from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

logger = logging.getLogger(__name__)

_MAP_PATH = pathlib.Path(__file__).parent / "code_map.yaml"


@dataclass
class CodeMapEntry:
    loinc_code: str
    display_name: str
    unit: str
    min_value: float
    max_value: float
    required: bool


@dataclass
class MappedObservation:
    # Original parsed fields
    seq: int
    local_code: str
    value: Optional[float]
    unit: Optional[str]
    abnormal_flag: Optional[str]
    observation_ts: Optional[str]

    # Canonical fields (populated only when mapped=True)
    mapped: bool
    loinc_code: Optional[str]
    canonical_name: Optional[str]
    expected_unit: Optional[str]


@dataclass
class MappedMessage:
    # Pass-through from ParsedMessage
    control_id: str
    sending_application: str
    sending_facility: str
    patient_id: str
    encounter_id: str
    message_ts: str
    observation_ts: str
    sha256: str

    observations: list[MappedObservation]
    unmapped_codes: list[str]      # local codes with no entry in code_map.yaml
    parse_warnings: list[str]      # forwarded from ParsedMessage


# ---- module-level map singleton ------------------------------------------------

_CODE_MAP: dict[str, CodeMapEntry] = {}


def _load_map(path: pathlib.Path = _MAP_PATH) -> dict[str, CodeMapEntry]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    result: dict[str, CodeMapEntry] = {}
    for local_code, entry in raw.get("mappings", {}).items():
        result[local_code] = CodeMapEntry(
            loinc_code=entry["loinc_code"],
            display_name=entry["display_name"],
            unit=entry["unit"],
            min_value=float(entry["min_value"]),
            max_value=float(entry["max_value"]),
            required=bool(entry.get("required", False)),
        )
    logger.debug("Loaded %d code mappings from %s", len(result), path)
    return result


def reload_map(path: pathlib.Path = _MAP_PATH) -> None:
    """Force reload of the code map. Useful in tests and after YAML edits."""
    global _CODE_MAP
    _CODE_MAP = _load_map(path)


# Load on first import
_CODE_MAP = _load_map()


# ---- public API ----------------------------------------------------------------

def get_map() -> dict[str, CodeMapEntry]:
    """Return the current code map (read-only view)."""
    return dict(_CODE_MAP)


def get_required_codes() -> set[str]:
    """Return the set of local codes that are marked required=true."""
    return {code for code, entry in _CODE_MAP.items() if entry.required}


def map_observation(obs: ParsedObservation) -> MappedObservation:
    """Map a single ParsedObservation to a MappedObservation.

    If the local_code is not in the map, mapped=False and the loinc_*
    fields are None. The caller should route unmapped observations to
    the quarantine queue.
    """
    entry = _CODE_MAP.get(obs.local_code)
    if entry is None:
        return MappedObservation(
            seq=obs.seq,
            local_code=obs.local_code,
            value=obs.value,
            unit=obs.unit,
            abnormal_flag=obs.abnormal_flag,
            observation_ts=obs.observation_ts,
            mapped=False,
            loinc_code=None,
            canonical_name=None,
            expected_unit=None,
        )
    return MappedObservation(
        seq=obs.seq,
        local_code=obs.local_code,
        value=obs.value,
        unit=obs.unit,
        abnormal_flag=obs.abnormal_flag,
        observation_ts=obs.observation_ts,
        mapped=True,
        loinc_code=entry.loinc_code,
        canonical_name=entry.display_name,
        expected_unit=entry.unit,
    )


def map_message(parsed: ParsedMessage) -> MappedMessage:
    """Map all observations in a ParsedMessage.

    Returns a MappedMessage with:
      - observations: all ParsedObservations mapped (mapped=True) or
        flagged (mapped=False) in order.
      - unmapped_codes: deduplicated list of local codes not found in the map.
    """
    mapped_obs: list[MappedObservation] = []
    unmapped: list[str] = []

    for obs in parsed.observations:
        mo = map_observation(obs)
        mapped_obs.append(mo)
        if not mo.mapped and obs.local_code not in unmapped:
            unmapped.append(obs.local_code)

    return MappedMessage(
        control_id=parsed.control_id,
        sending_application=parsed.sending_application,
        sending_facility=parsed.sending_facility,
        patient_id=parsed.patient_id,
        encounter_id=parsed.encounter_id,
        message_ts=parsed.message_ts,
        observation_ts=parsed.observation_ts,
        sha256=parsed.sha256,
        observations=mapped_obs,
        unmapped_codes=unmapped,
        parse_warnings=parsed.parse_warnings,
    )
