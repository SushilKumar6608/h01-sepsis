"""
Phase 1 — HL7v2 message parser
================================
Parses ORU^R01 messages produced by the generator (or any conformant sender)
into a structured ParsedMessage dataclass.

Uses python-hl7 for low-level segment access.  All quirks handled:
  - trailing whitespace stripped from OBX values
  - missing units field → None (not an error)
  - custom ZPT segment → captured in extra_segments
  - empty OBX value → observation skipped (logged)

No database access here — this module is pure parsing logic so it can be
tested without PostgreSQL.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import hl7

logger = logging.getLogger(__name__)


# ── data classes ───────────────────────────────────────────────────────────────

@dataclass
class ParsedObservation:
    seq: int                     # OBX sequence number
    local_code: str              # e.g. WBC_LOC
    description: str             # e.g. White blood cell count
    value: Optional[float]       # None if the OBX value was empty
    unit: Optional[str]          # None if units field was missing
    abnormal_flag: Optional[str] # H / L / N / None
    observation_ts: Optional[str]


@dataclass
class ParsedMessage:
    # Identifiers
    control_id: str
    sending_application: str
    sending_facility: str
    patient_id: str
    encounter_id: str
    message_ts: str              # MSH timestamp (YYYYMMDDHHmmss)
    observation_ts: str          # OBR timestamp

    # Payload
    observations: list[ParsedObservation] = field(default_factory=list)
    extra_segments: list[str] = field(default_factory=list)  # e.g. ZPT lines

    # Integrity
    sha256: str = ""             # SHA-256 of the raw message bytes

    # Parse metadata
    parse_warnings: list[str] = field(default_factory=list)


# ── parser ─────────────────────────────────────────────────────────────────────

class HL7v2ParseError(Exception):
    pass


def _field(segment, index: int, default: str = "") -> str:
    """Safely extract a field from an HL7 segment, stripping whitespace."""
    try:
        val = str(segment[index])
        return val.strip()
    except (IndexError, TypeError):
        return default


def _component(segment, field_index: int, comp_index: int, default: str = "") -> str:
    """Extract a component within an HL7 field (^ separated).

    python-hl7 Field[0] returns a Repetition object (the whole field string),
    so component access must be done by splitting the string on '^'.
    comp_index is 0-based (0 = first component).
    """
    try:
        parts = str(segment[field_index]).split("^")
        val = parts[comp_index]
        return val.strip()
    except (IndexError, TypeError):
        return default


def parse_message(raw: str) -> ParsedMessage:
    """
    Parse one HL7v2 ORU^R01 message string.

    Parameters
    ----------
    raw : str
        Complete HL7v2 message with segments separated by \\r.

    Returns
    -------
    ParsedMessage
        Structured representation.  parse_warnings lists non-fatal issues.

    Raises
    ------
    HL7v2ParseError
        On fatal structural problems (missing MSH, wrong message type).
    """
    raw_bytes = raw.encode("utf-8")
    sha = hashlib.sha256(raw_bytes).hexdigest()

    # Normalise line endings: python-hl7 requires \r as segment separator.
    # Messages arriving with \r\n (Windows) or \n (Unix) must be converted.
    raw = raw.replace("\r\n", "\r").replace("\n", "\r")

    try:
        msg = hl7.parse(raw)
    except Exception as exc:
        raise HL7v2ParseError(f"hl7.parse failed: {exc}") from exc

    warnings: list[str] = []

    # ── MSH ──────────────────────────────────────────────────────────────────
    try:
        msh = msg.segment("MSH")
    except KeyError:
        raise HL7v2ParseError("MSH segment missing")

    msg_type = _field(msh, 9)
    if "ORU" not in msg_type:
        raise HL7v2ParseError(f"Expected ORU^R01, got: {msg_type}")

    control_id        = _field(msh, 10)
    message_ts        = _field(msh, 7)
    sending_app       = _field(msh, 3)
    sending_facility  = _field(msh, 4)

    # ── PID ──────────────────────────────────────────────────────────────────
    try:
        pid = msg.segment("PID")
        patient_id = _component(pid, 3, 0) or _field(pid, 3)
    except KeyError:
        warnings.append("PID segment missing; patient_id unknown")
        patient_id = "UNKNOWN"

    # ── PV1 ──────────────────────────────────────────────────────────────────
    try:
        pv1 = msg.segment("PV1")
        encounter_id = _field(pv1, 18) or _field(pv1, 19) or _field(pv1, 5)
    except KeyError:
        warnings.append("PV1 segment missing; encounter_id unknown")
        encounter_id = f"ENC_{patient_id}"

    # ── OBR ──────────────────────────────────────────────────────────────────
    try:
        obr = msg.segment("OBR")
        observation_ts = _field(obr, 7) or message_ts
    except KeyError:
        warnings.append("OBR segment missing; using MSH timestamp")
        observation_ts = message_ts

    # ── OBX (zero or more) ───────────────────────────────────────────────────
    observations: list[ParsedObservation] = []
    extra_segments: list[str] = []

    for segment in msg:
        seg_id = str(segment[0])

        if seg_id == "OBX":
            seq_str   = _field(segment, 1)
            seq       = int(seq_str) if seq_str.isdigit() else len(observations) + 1
            local_code = _component(segment, 3, 0) or _field(segment, 3)
            description = _component(segment, 3, 1) or ""
            raw_value  = _field(segment, 5)
            unit_raw   = _field(segment, 6)
            abn_flag   = _field(segment, 8) or None
            obs_ts     = _field(segment, 14) or observation_ts

            unit = unit_raw if unit_raw else None
            if not unit:
                warnings.append(f"OBX[{seq}] {local_code}: units field empty")

            if not raw_value:
                warnings.append(f"OBX[{seq}] {local_code}: value is empty — skipping")
                continue

            try:
                value = float(raw_value)
            except ValueError:
                warnings.append(
                    f"OBX[{seq}] {local_code}: cannot parse value '{raw_value}' as float — skipping"
                )
                continue

            observations.append(ParsedObservation(
                seq=seq,
                local_code=local_code,
                description=description,
                value=value,
                unit=unit,
                abnormal_flag=abn_flag,
                observation_ts=obs_ts,
            ))

        elif seg_id not in ("MSH", "PID", "PV1", "OBR"):
            # Custom or unknown segment — capture as raw string
            extra_segments.append(str(segment))

    parsed = ParsedMessage(
        control_id=control_id,
        sending_application=sending_app,
        sending_facility=sending_facility,
        patient_id=patient_id,
        encounter_id=encounter_id,
        message_ts=message_ts,
        observation_ts=observation_ts,
        observations=observations,
        extra_segments=extra_segments,
        sha256=sha,
        parse_warnings=warnings,
    )

    if warnings:
        logger.debug(
            "Parse warnings for control_id=%s: %s", control_id, "; ".join(warnings)
        )

    return parsed
