"""
Phase 3 - Cross-format parity checker.

Compares two ParsedMessage objects representing the same clinical encounter
(one from HL7v2, one from FHIR R4) and produces a ParityReport that
classifies each observation as matched, mismatched, or missing from one
side.

Matching key: local_code (which for FHIR messages is the LOINC code, and
for HL7v2 messages is the local lab code).  A fully matched pair must have
the same local_code AND the same numeric value (within tolerance).

Design choices:
  - Value comparison uses an absolute tolerance of 0.001 and a relative
    tolerance of 0.001 (0.1%), matching typical lab rounding conventions.
  - Unit differences are flagged as warnings, not mismatches, because FHIR
    and HL7v2 may use different unit representations for the same quantity
    (e.g. "g/dL" vs "g/dl").
  - The checker is order-independent: it aligns by local_code, not by seq.
  - Multiple observations with the same local_code in one message are
    handled by comparing the first occurrence on each side; duplicates are
    noted in the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

# Tolerances for numeric value comparison
_ABS_TOL = 0.001
_REL_TOL = 0.001


def _values_equal(a: Optional[float], b: Optional[float]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, abs_tol=_ABS_TOL, rel_tol=_REL_TOL)


@dataclass
class ObservationPair:
    """One aligned pair of observations from two messages."""
    local_code: str
    hl7_obs: Optional[ParsedObservation]
    fhir_obs: Optional[ParsedObservation]
    matched: bool          # True when both present and value agrees
    value_mismatch: bool   # True when both present but values differ
    unit_warning: str      # Non-empty when units differ (informational)
    missing_from: str      # "hl7" | "fhir" | "" when both present


@dataclass
class ParityReport:
    """Result of comparing two ParsedMessage objects."""
    control_id: str
    hl7_control_id: str
    fhir_control_id: str
    pairs: list[ObservationPair] = field(default_factory=list)
    duplicate_codes_hl7: list[str] = field(default_factory=list)
    duplicate_codes_fhir: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Convenience counts (computed by check_parity; not constructor args)
    matched_count: int = 0
    mismatched_count: int = 0
    missing_hl7_count: int = 0
    missing_fhir_count: int = 0

    def _recount(self) -> None:
        self.matched_count = sum(1 for p in self.pairs if p.matched)
        self.mismatched_count = sum(1 for p in self.pairs if p.value_mismatch)
        self.missing_hl7_count = sum(1 for p in self.pairs if p.missing_from == "hl7")
        self.missing_fhir_count = sum(1 for p in self.pairs if p.missing_from == "fhir")

    @property
    def all_matched(self) -> bool:
        return self.mismatched_count == 0 and self.missing_hl7_count == 0 and self.missing_fhir_count == 0


def check_parity(hl7_msg: ParsedMessage, fhir_msg: ParsedMessage) -> ParityReport:
    """Compare one HL7v2-sourced message against one FHIR-sourced message.

    The two messages should represent the same encounter / observation event.
    There is no enforcement of control_id equality here because control IDs
    from HL7v2 (MSH-10) and FHIR (Bundle.id) are typically different strings
    even when they describe the same event.  The caller is responsible for
    presenting matching pairs.
    """
    report = ParityReport(
        control_id=hl7_msg.control_id,
        hl7_control_id=hl7_msg.control_id,
        fhir_control_id=fhir_msg.control_id,
    )

    # Index observations by local_code; note duplicates
    def _index(observations: list[ParsedObservation], dup_list: list[str]) -> dict[str, ParsedObservation]:
        index: dict[str, ParsedObservation] = {}
        for obs in observations:
            if obs.local_code in index:
                dup_list.append(obs.local_code)
            else:
                index[obs.local_code] = obs
        return index

    hl7_index = _index(hl7_msg.observations, report.duplicate_codes_hl7)
    fhir_index = _index(fhir_msg.observations, report.duplicate_codes_fhir)

    all_codes = sorted(set(hl7_index) | set(fhir_index))

    for code in all_codes:
        hl7_obs = hl7_index.get(code)
        fhir_obs = fhir_index.get(code)

        if hl7_obs is None:
            pair = ObservationPair(
                local_code=code,
                hl7_obs=None,
                fhir_obs=fhir_obs,
                matched=False,
                value_mismatch=False,
                unit_warning="",
                missing_from="hl7",
            )
        elif fhir_obs is None:
            pair = ObservationPair(
                local_code=code,
                hl7_obs=hl7_obs,
                fhir_obs=None,
                matched=False,
                value_mismatch=False,
                unit_warning="",
                missing_from="fhir",
            )
        else:
            value_ok = _values_equal(hl7_obs.value, fhir_obs.value)
            unit_warn = ""
            if hl7_obs.unit and fhir_obs.unit:
                if hl7_obs.unit.replace(" ", "") != fhir_obs.unit.replace(" ", ""):
                    unit_warn = (
                        f"unit mismatch for {code}: "
                        f"HL7='{hl7_obs.unit}' FHIR='{fhir_obs.unit}'"
                    )
            pair = ObservationPair(
                local_code=code,
                hl7_obs=hl7_obs,
                fhir_obs=fhir_obs,
                matched=value_ok,
                value_mismatch=not value_ok,
                unit_warning=unit_warn,
                missing_from="",
            )

        report.pairs.append(pair)

    if report.duplicate_codes_hl7:
        report.warnings.append(
            f"Duplicate codes in HL7v2 message (first occurrence used): "
            f"{report.duplicate_codes_hl7}"
        )
    if report.duplicate_codes_fhir:
        report.warnings.append(
            f"Duplicate codes in FHIR message (first occurrence used): "
            f"{report.duplicate_codes_fhir}"
        )

    report._recount()
    return report
