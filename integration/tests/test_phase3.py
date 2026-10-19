"""
Phase 3 tests - FHIR R4 parser and cross-format parity checker.

Run from the project root (D:\\Egna projekt\\h01-sepsis):
    conda activate D:\\conda-envs\\h01-sepsis
    pytest integration/tests/test_phase3.py -v
"""
import pytest

from integration.fhir.fhir_parser import (
    FHIRParseError,
    parse_bundle,
    parse_observation_resource,
)
from integration.canonical.parity import check_parity
from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation


# ---------------------------------------------------------------------------
# Fixtures / helper factories
# ---------------------------------------------------------------------------

def _make_observation(loinc: str, value: float, unit: str, ts: str = "2024-01-01T08:00:00Z") -> dict:
    return {
        "resourceType": "Observation",
        "id": f"obs-{loinc.replace('.','-')}",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": loinc,
                    "display": f"Test {loinc}",
                }
            ]
        },
        "subject": {"reference": "Patient/pt-001"},
        "encounter": {"reference": "Encounter/enc-001"},
        "effectiveDateTime": ts,
        "valueQuantity": {"value": value, "unit": unit},
    }


def _make_bundle(observations: list[dict], bundle_id: str = "bundle-001") -> dict:
    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": "2024-01-01T08:00:00Z",
        "entry": [{"resource": obs} for obs in observations],
    }


def _make_hl7_parsed_message(
    control_id: str,
    observations: list[tuple],  # (local_code, value, unit)
) -> ParsedMessage:
    """Build a minimal ParsedMessage as the HL7v2 parser would produce."""
    obs_list = [
        ParsedObservation(
            seq=i + 1,
            local_code=code,
            description="",
            value=val,
            unit=unit,
            abnormal_flag=None,
            observation_ts="2024-01-01T08:00:00Z",
        )
        for i, (code, val, unit) in enumerate(observations)
    ]
    return ParsedMessage(
        control_id=control_id,
        sending_application="TEST_APP",
        sending_facility="TEST_FAC",
        patient_id="pt-001",
        encounter_id="enc-001",
        message_ts="2024-01-01T08:00:00Z",
        observation_ts="2024-01-01T08:00:00Z",
        observations=obs_list,
    )


# ---------------------------------------------------------------------------
# T3-01: parse a FHIR Bundle with multiple Observations
# ---------------------------------------------------------------------------

def test_parse_bundle_basic():
    """parse_bundle extracts all numeric Observations from a Bundle."""
    bundle = _make_bundle([
        _make_observation("8867-4", 88.0, "/min"),   # Heart rate
        _make_observation("59408-5", 97.0, "%"),      # SpO2
        _make_observation("9279-1", 18.0, "/min"),    # Resp rate
    ])
    msg = parse_bundle(bundle)

    assert msg.sending_application == "FHIR-R4"
    assert msg.patient_id == "pt-001"
    assert msg.encounter_id == "enc-001"
    assert len(msg.observations) == 3
    codes = {o.local_code for o in msg.observations}
    assert codes == {"8867-4", "59408-5", "9279-1"}


# ---------------------------------------------------------------------------
# T3-02: parse a single FHIR Observation resource
# ---------------------------------------------------------------------------

def test_parse_observation_resource_basic():
    """parse_observation_resource wraps a single Observation in a ParsedMessage."""
    obs = _make_observation("8867-4", 72.0, "/min")
    msg = parse_observation_resource(obs, control_id="ctrl-999")

    assert msg.control_id == "ctrl-999"
    assert msg.patient_id == "pt-001"
    assert len(msg.observations) == 1
    assert msg.observations[0].local_code == "8867-4"
    assert msg.observations[0].value == pytest.approx(72.0)
    assert msg.observations[0].unit == "/min"


# ---------------------------------------------------------------------------
# T3-03: Bundle with non-Observation entries is handled gracefully
# ---------------------------------------------------------------------------

def test_parse_bundle_skips_non_observation_entries():
    """Non-Observation resources inside a Bundle are silently skipped."""
    bundle = _make_bundle([_make_observation("8867-4", 90.0, "/min")])
    # Inject a Patient resource entry
    bundle["entry"].insert(
        0,
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pt-001",
                "name": [{"family": "Doe"}],
            }
        },
    )
    msg = parse_bundle(bundle)
    assert len(msg.observations) == 1
    assert msg.observations[0].local_code == "8867-4"


# ---------------------------------------------------------------------------
# T3-04: Observation with no LOINC code gets quarantine-ready local_code
# ---------------------------------------------------------------------------

def test_parse_bundle_unknown_loinc_code():
    """An Observation without a LOINC system code gets its raw code as local_code
    and a parse_warning - so the mapper can quarantine it as UNMAPPED_CODE."""
    obs = {
        "resourceType": "Observation",
        "id": "obs-custom",
        "status": "final",
        "code": {
            "coding": [
                {"system": "http://hospital.example/codes", "code": "CUSTOM_HR", "display": "Custom HR"}
            ]
        },
        "subject": {"reference": "Patient/pt-001"},
        "effectiveDateTime": "2024-01-01T08:00:00Z",
        "valueQuantity": {"value": 82.0, "unit": "/min"},
    }
    bundle = _make_bundle([obs])
    msg = parse_bundle(bundle)

    assert len(msg.observations) == 1
    assert msg.observations[0].local_code == "CUSTOM_HR"
    assert any("no LOINC code" in w for w in msg.parse_warnings)


# ---------------------------------------------------------------------------
# T3-05: Observation with non-numeric value type is skipped with warning
# ---------------------------------------------------------------------------

def test_parse_bundle_non_numeric_value_skipped():
    """valueString observations are skipped; a parse_warning is recorded."""
    obs_hr = _make_observation("8867-4", 78.0, "/min")
    obs_string = {
        "resourceType": "Observation",
        "id": "obs-note",
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "11450-4", "display": "Some note"}]
        },
        "subject": {"reference": "Patient/pt-001"},
        "effectiveDateTime": "2024-01-01T08:00:00Z",
        "valueString": "normal",
    }
    bundle = _make_bundle([obs_hr, obs_string])
    msg = parse_bundle(bundle)

    # Only the numeric observation should come through
    assert len(msg.observations) == 1
    assert msg.observations[0].local_code == "8867-4"
    # And there should be a warning about the skipped one
    assert any("not numeric" in w or "valueString" in w for w in msg.parse_warnings)


# ---------------------------------------------------------------------------
# T3-06: FHIRParseError raised for wrong resourceType on parse_bundle
# ---------------------------------------------------------------------------

def test_parse_bundle_wrong_resource_type():
    """FHIRParseError is raised when the top-level resourceType is not Bundle."""
    with pytest.raises(FHIRParseError, match="resourceType"):
        parse_bundle({"resourceType": "Patient", "id": "pt-001"})


# ---------------------------------------------------------------------------
# T3-07: FHIRParseError raised for wrong resourceType on parse_observation_resource
# ---------------------------------------------------------------------------

def test_parse_observation_resource_wrong_type():
    """FHIRParseError is raised when the dict is not an Observation."""
    with pytest.raises(FHIRParseError, match="resourceType"):
        parse_observation_resource({"resourceType": "Bundle"})


# ---------------------------------------------------------------------------
# T3-08: parity - all observations match
# ---------------------------------------------------------------------------

def test_parity_all_matched():
    """When HL7v2 and FHIR observations share the same LOINC codes and values,
    all pairs are reported as matched."""
    # HL7v2 uses LOINC codes as local_code (as the mapper produces after mapping)
    hl7_msg = _make_hl7_parsed_message(
        "ctrl-hl7-001",
        [("8867-4", 88.0, "/min"), ("59408-5", 97.0, "%")],
    )
    fhir_bundle = _make_bundle([
        _make_observation("8867-4", 88.0, "/min"),
        _make_observation("59408-5", 97.0, "%"),
    ])
    fhir_msg = parse_bundle(fhir_bundle)

    report = check_parity(hl7_msg, fhir_msg)

    assert report.matched_count == 2
    assert report.mismatched_count == 0
    assert report.missing_hl7_count == 0
    assert report.missing_fhir_count == 0
    assert report.all_matched is True


# ---------------------------------------------------------------------------
# T3-09: parity - value mismatch detected
# ---------------------------------------------------------------------------

def test_parity_value_mismatch():
    """When values differ beyond tolerance, the pair is flagged as mismatched."""
    hl7_msg = _make_hl7_parsed_message(
        "ctrl-hl7-002",
        [("8867-4", 88.0, "/min")],
    )
    fhir_bundle = _make_bundle([_make_observation("8867-4", 92.0, "/min")])
    fhir_msg = parse_bundle(fhir_bundle)

    report = check_parity(hl7_msg, fhir_msg)

    assert report.matched_count == 0
    assert report.mismatched_count == 1
    assert report.all_matched is False
    mismatch_pair = next(p for p in report.pairs if p.value_mismatch)
    assert mismatch_pair.local_code == "8867-4"
    assert mismatch_pair.hl7_obs.value == pytest.approx(88.0)
    assert mismatch_pair.fhir_obs.value == pytest.approx(92.0)


# ---------------------------------------------------------------------------
# T3-10: parity - observation present in FHIR but missing from HL7v2
# ---------------------------------------------------------------------------

def test_parity_missing_from_hl7():
    """A FHIR observation that has no counterpart in the HL7v2 message is
    reported as missing_from='hl7'."""
    hl7_msg = _make_hl7_parsed_message(
        "ctrl-hl7-003",
        [("8867-4", 88.0, "/min")],
    )
    fhir_bundle = _make_bundle([
        _make_observation("8867-4", 88.0, "/min"),
        _make_observation("59408-5", 97.0, "%"),  # only in FHIR
    ])
    fhir_msg = parse_bundle(fhir_bundle)

    report = check_parity(hl7_msg, fhir_msg)

    assert report.matched_count == 1
    assert report.missing_hl7_count == 1
    missing_pair = next(p for p in report.pairs if p.missing_from == "hl7")
    assert missing_pair.local_code == "59408-5"
    assert missing_pair.hl7_obs is None
    assert missing_pair.fhir_obs is not None


# ---------------------------------------------------------------------------
# T3-11: parity - observation present in HL7v2 but missing from FHIR
# ---------------------------------------------------------------------------

def test_parity_missing_from_fhir():
    """An HL7v2 observation that has no counterpart in the FHIR message is
    reported as missing_from='fhir'."""
    hl7_msg = _make_hl7_parsed_message(
        "ctrl-hl7-004",
        [("8867-4", 88.0, "/min"), ("9279-1", 18.0, "/min")],  # resp only in HL7
    )
    fhir_bundle = _make_bundle([_make_observation("8867-4", 88.0, "/min")])
    fhir_msg = parse_bundle(fhir_bundle)

    report = check_parity(hl7_msg, fhir_msg)

    assert report.matched_count == 1
    assert report.missing_fhir_count == 1
    missing_pair = next(p for p in report.pairs if p.missing_from == "fhir")
    assert missing_pair.local_code == "9279-1"
    assert missing_pair.fhir_obs is None
    assert missing_pair.hl7_obs is not None


# ---------------------------------------------------------------------------
# T3-12: parity - unit difference is a warning, not a mismatch
# ---------------------------------------------------------------------------

def test_parity_unit_warning_not_mismatch():
    """Unit differences (e.g. 'g/dL' vs 'g/dl') produce a unit_warning on the
    pair but do NOT flip matched to False."""
    hl7_msg = _make_hl7_parsed_message(
        "ctrl-hl7-005",
        [("718-7", 13.5, "g/dL")],
    )
    fhir_bundle = _make_bundle([_make_observation("718-7", 13.5, "g/dl")])
    fhir_msg = parse_bundle(fhir_bundle)

    report = check_parity(hl7_msg, fhir_msg)

    assert report.matched_count == 1
    assert report.mismatched_count == 0
    pair = report.pairs[0]
    assert pair.matched is True
    assert "unit mismatch" in pair.unit_warning


# ---------------------------------------------------------------------------
# T3-13: Bundle with no entries produces a parse_warning, not an exception
# ---------------------------------------------------------------------------

def test_parse_empty_bundle_no_exception():
    """An empty Bundle does not raise; it returns a message with zero
    observations and a parse_warning."""
    bundle = {
        "resourceType": "Bundle",
        "id": "empty-bundle",
        "type": "collection",
        "timestamp": "2024-01-01T00:00:00Z",
        "entry": [],
    }
    msg = parse_bundle(bundle)
    assert len(msg.observations) == 0
    assert any("No numeric" in w for w in msg.parse_warnings)


# ---------------------------------------------------------------------------
# T3-14: sha256 is populated on both parsed message types
# ---------------------------------------------------------------------------

def test_sha256_populated():
    """parse_bundle and parse_observation_resource both set a non-empty sha256."""
    bundle = _make_bundle([_make_observation("8867-4", 88.0, "/min")])
    msg_b = parse_bundle(bundle)
    assert msg_b.sha256 != ""

    obs = _make_observation("8867-4", 88.0, "/min")
    msg_o = parse_observation_resource(obs)
    assert msg_o.sha256 != ""
