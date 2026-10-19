"""
Phase 1 test suite
===================
Tests parsing, idempotency, and quirk handling.
No external database required — uses SQLite in-memory via init_db().

Run (from repo root, Anaconda Prompt):
    cd D:\\Egna projekt\\h01-sepsis
    python -m pytest integration/tests/test_phase1.py -v
"""

from __future__ import annotations

import os
import textwrap

import pytest

# ── force SQLite in-memory for the test session ────────────────────────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402 (after env var set)

from integration.canonical.db import init_db  # noqa: E402
from integration.ingest.api import app  # noqa: E402
from integration.ingest.hl7v2_parser import (  # noqa: E402
    HL7v2ParseError,
    parse_message,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

MINIMAL_ORU = textwrap.dedent("""\
    MSH|^~\\&|HOSP_A_LIS|HOSP_A|INTEGRATION|PLATFORM|20260101120000||ORU^R01|CTRL001|P|2.5\r
    PID|1||P000001^^^HOSP_A^MR||SYNTHETIC^PATIENT||19700101|U\r
    PV1|1|I|ICU^^^HOSP_A|||||||||||||||ENC_P000001\r
    OBR|1|||VITALS_LABS^ICU vitals and labs^L|||20260101120000\r
    OBX|1|NM|HR_LOC^Heart rate^L||72.0000|bpm|60-100|N|||F|||20260101120000\r
    OBX|2|NM|SPO2_LOC^Oxygen saturation SpO2^L||98.0000|%|95-100|N|||F|||20260101120000\r
    ZPT|1|ICU_BED_05|STABLE\r
""")

ORU_WITH_QUIRKS = textwrap.dedent("""\
    MSH|^~\\&|HOSP_A_LIS|HOSP_A|INTEGRATION|PLATFORM|20260101130000||ORU^R01|CTRL002|P|2.5\r
    PID|1||P000002^^^HOSP_A^MR||SYNTHETIC^PATIENT||19700101|U\r
    PV1|1|I|ICU^^^HOSP_A|||||||||||||||ENC_P000002\r
    OBR|1|||VITALS_LABS^ICU vitals and labs^L|||20260101130000\r
    OBX|1|NM|HR_LOC^Heart rate^L||88.0000  ||60-100|N|||F|||20260101130000\r
    OBX|2|NM|RR_LOC^Respiratory rate^L||||12-20|N|||F|||20260101130000\r
    ZPT|1|ICU_BED_12|STABLE\r
""")

ORU_DUPLICATE = MINIMAL_ORU  # same control_id as MINIMAL_ORU


# ── parser unit tests (no DB) ─────────────────────────────────────────────────

class TestParser:

    def test_minimal_message_parses(self):
        msg = parse_message(MINIMAL_ORU)
        assert msg.control_id == "CTRL001"
        assert msg.patient_id == "P000001"
        assert msg.encounter_id == "ENC_P000001"
        assert len(msg.observations) == 2

    def test_hr_value_correct(self):
        msg = parse_message(MINIMAL_ORU)
        hr = next(o for o in msg.observations if o.local_code == "HR_LOC")
        assert hr.value == pytest.approx(72.0)
        assert hr.unit == "bpm"
        assert hr.abnormal_flag == "N"

    def test_zpt_segment_captured(self):
        msg = parse_message(MINIMAL_ORU)
        assert any("ZPT" in s for s in msg.extra_segments)

    def test_trailing_whitespace_stripped(self):
        """OBX value '88.0000  ' → parsed as 88.0 without error."""
        msg = parse_message(ORU_WITH_QUIRKS)
        hr = next((o for o in msg.observations if o.local_code == "HR_LOC"), None)
        assert hr is not None
        assert hr.value == pytest.approx(88.0)

    def test_missing_units_generates_warning(self):
        msg = parse_message(ORU_WITH_QUIRKS)
        assert any("units field empty" in w for w in msg.parse_warnings)

    def test_empty_obx_value_skipped(self):
        """OBX for RR_LOC has empty value → observation is skipped, warning added."""
        msg = parse_message(ORU_WITH_QUIRKS)
        codes = [o.local_code for o in msg.observations]
        assert "RR_LOC" not in codes
        assert any("value is empty" in w for w in msg.parse_warnings)

    def test_sha256_is_set(self):
        msg = parse_message(MINIMAL_ORU)
        assert len(msg.sha256) == 64

    def test_wrong_message_type_raises(self):
        bad = MINIMAL_ORU.replace("ORU^R01", "ADT^A01")
        with pytest.raises(HL7v2ParseError, match="Expected ORU"):
            parse_message(bad)

    def test_missing_msh_raises(self):
        no_msh = "PID|1||P000001\r"
        with pytest.raises(HL7v2ParseError):
            parse_message(no_msh)


# ── API integration tests (SQLite in-memory) ──────────────────────────────────

@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


class TestAPI:

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ingest_accepted(self, client):
        r = client.post(
            "/ingest/hl7v2",
            content=MINIMAL_ORU,
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "accepted"
        assert body["control_id"] == "CTRL001"
        assert body["observations"] == 2

    def test_duplicate_returns_409(self, client):
        # Second POST with same control_id
        r = client.post(
            "/ingest/hl7v2",
            content=ORU_DUPLICATE,
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 409
        body = r.json()
        assert body["detail"] == "Duplicate"
        assert body["control_id"] == "CTRL001"

    def test_quirky_message_accepted(self, client):
        """Trailing whitespace + missing units + empty value — still 201."""
        r = client.post(
            "/ingest/hl7v2",
            content=ORU_WITH_QUIRKS,
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 201
        body = r.json()
        # HR_LOC parsed; RR_LOC skipped (empty value)
        assert body["observations"] == 1
        assert any("units field empty" in w for w in body["warnings"])
        assert any("value is empty" in w for w in body["warnings"])

    def test_bad_message_returns_422(self, client):
        r = client.post(
            "/ingest/hl7v2",
            content="GARBAGE|NOT|HL7",
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 422
