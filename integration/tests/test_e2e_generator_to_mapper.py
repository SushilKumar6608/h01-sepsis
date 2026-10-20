"""
End-to-end test: HL7v2 generator -> parser -> mapper.

This test catches local-code mismatches between hospital_profiles.yaml
(what the generator emits) and code_map.yaml (what the mapper expects).
It builds a minimal ORU^R01 message using the same profile the generator
uses, then sends it through the real HL7v2 parser and mapper pipeline.

If a local code in hospital_profiles.yaml is not in code_map.yaml, the
mapper will quarantine the observation as UNMAPPED_CODE instead of
mapping it to LOINC.  Any UNMAPPED_CODE result causes this test to fail,
so a code-name mismatch is caught immediately rather than discovered in
production.

Run from the project root (D:\\Egna projekt\\h01-sepsis):
    conda activate D:\\conda-envs\\h01-sepsis
    pytest integration/tests/test_e2e_generator_to_mapper.py -v
"""
from __future__ import annotations

import pathlib
import textwrap

import pytest
import yaml

from integration.ingest.hl7v2_parser import parse_message
from integration.canonical.mapper import map_message

# ---------------------------------------------------------------------------
# Load the generator's hospital profile so the test stays in sync with it
# ---------------------------------------------------------------------------

_PROFILES_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "simulate" / "hospital_profiles.yaml"
)
_CODE_MAP_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "canonical" / "code_map.yaml"
)


def _load_profile() -> dict:
    with open(_PROFILES_PATH) as f:
        return yaml.safe_load(f)["hospital_A"]


def _load_code_map() -> dict:
    with open(_CODE_MAP_PATH) as f:
        return yaml.safe_load(f)["mappings"]


# ---------------------------------------------------------------------------
# Helper: build a minimal ORU^R01 message for a given feature list
# ---------------------------------------------------------------------------

def _build_oru(profile: dict, feature_values: dict[str, float]) -> str:
    """Build a minimal ORU^R01 using the hospital profile's local codes."""
    local_codes = profile["local_codes"]
    ts = "20260101080000"
    segments = [
        "MSH|^~\\&|HOSP_A_LIS|HOSP_A|INTEGRATION|PLATFORM"
        f"|{ts}||ORU^R01|E2E_TEST_CTRL_001|P|2.5",
        "PID|1||P999999^^^HOSP_A^MR||SYNTHETIC^PATIENT||19700101|U",
        "PV1|1|I|ICU^^^HOSP_A|||||||||||||||ENC_P999999",
        "OBR|1|||VITALS_LABS^ICU vitals and labs^L|||" + ts,
    ]
    for seq, (feature, value) in enumerate(feature_values.items(), start=1):
        info = local_codes[feature]
        code = info["code"]
        desc = info["description"]
        unit = info["unit"]
        segments.append(
            f"OBX|{seq}|NM|{code}^{desc}^L"
            f"||{value:.4f}|{unit}|||N|||F|||{ts}"
        )
    return "\r".join(segments) + "\r"


# ---------------------------------------------------------------------------
# TE2E-01: every feature in hospital_profiles.yaml maps without UNMAPPED_CODE
# ---------------------------------------------------------------------------

def test_all_local_codes_map_to_loinc():
    """Every local code in hospital_profiles.yaml must resolve in code_map.yaml.

    If the generator emits a code the mapper doesn't know, this test fails
    with the list of unmapped codes so the mismatch is immediately visible.
    """
    profile = _load_profile()

    # Use a safe, non-NaN value for each feature
    feature_values = {
        "HR": 80.0,
        "O2Sat": 97.0,
        "Resp": 16.0,
        "SBP": 120.0,
        "MAP": 80.0,
        "Temp": 37.0,
        "Glucose": 100.0,
        "DBP": 70.0,
        "Creatinine": 1.0,
        "BUN": 15.0,
        "Potassium": 4.0,
        "WBC": 8.0,
        "Platelets": 200.0,
        "Hgb": 12.0,
    }

    raw = _build_oru(profile, feature_values)
    parsed = parse_message(raw)

    assert len(parsed.observations) == len(feature_values), (
        f"Expected {len(feature_values)} observations, got {len(parsed.observations)}"
    )

    result = map_message(parsed)

    unmapped = result.unmapped_codes
    assert unmapped == [], (
        f"Local codes not found in code_map.yaml: {unmapped}\n"
        f"Check hospital_profiles.yaml and code_map.yaml for mismatched names."
    )


# ---------------------------------------------------------------------------
# TE2E-02: code_map.yaml has an entry for every local code in hospital_profiles
# ---------------------------------------------------------------------------

def test_code_map_covers_all_generator_codes():
    """Static check: code_map.yaml must contain every local_code in
    hospital_profiles.yaml.  This catches the mismatch without needing to
    run the full parser/mapper pipeline.
    """
    profile = _load_profile()
    code_map = _load_code_map()

    generator_codes = {
        info["code"]
        for info in profile["local_codes"].values()
    }
    mapper_codes = set(code_map.keys())

    missing_from_mapper = generator_codes - mapper_codes
    extra_in_mapper = mapper_codes - generator_codes

    assert missing_from_mapper == set(), (
        f"Generator emits codes not in code_map.yaml: {sorted(missing_from_mapper)}"
    )
    # Extra mapper codes (e.g. for a second hospital) are allowed - not a failure.
    # Just print them for visibility.
    if extra_in_mapper:
        print(
            f"\nINFO: code_map.yaml has additional codes not from Hospital A: "
            f"{sorted(extra_in_mapper)}"
        )


# ---------------------------------------------------------------------------
# TE2E-03: no out-of-range values for normal synthetic data
# ---------------------------------------------------------------------------

def test_normal_values_not_quarantined():
    """Normal-range synthetic values should pass validation without quarantine."""
    profile = _load_profile()

    feature_values = {
        "HR": 80.0,
        "O2Sat": 98.0,
        "Resp": 16.0,
        "SBP": 120.0,
        "MAP": 83.0,
        "Temp": 37.0,
        "Glucose": 100.0,
        "DBP": 70.0,
        "Creatinine": 1.0,
        "BUN": 15.0,
        "Potassium": 4.0,
        "WBC": 8.0,
        "Platelets": 200.0,
        "Hgb": 13.0,
    }

    raw = _build_oru(profile, feature_values)
    parsed = parse_message(raw)
    result = map_message(parsed)

    # map_message returns MappedMessage; unmapped_codes lists codes with no
    # entry in code_map.yaml.  All observations should be mapped.
    assert result.unmapped_codes == [], (
        f"Normal values produced UNMAPPED_CODE: {result.unmapped_codes}"
    )
    not_mapped = [
        mo.observation.local_code
        for mo in result.observations
        if not mo.mapped
    ]
    assert not_mapped == [], (
        f"Normal values were not mapped: {not_mapped}"
    )
