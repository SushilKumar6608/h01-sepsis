"""
Phase 2 tests - mapping, validation, quarantine, replay.

Run from project root (Anaconda Prompt, conda env h01-sepsis activated):
    cd D:\\Egna projekt\\h01-sepsis
    python -m pytest integration/tests/test_phase2.py -v

All 20 tests must pass before proceeding to Phase 3.
"""
from __future__ import annotations

import pathlib
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ---- helpers -------------------------------------------------------------------

def _make_session(url: str = "sqlite:///:memory:"):
    from integration.canonical.quarantine import Base as QBase
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, echo=False)
    QBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return Session()


# ==================== SECTION 1: mapper =========================================

class TestMapper:
    def setup_method(self):
        from integration.canonical import mapper
        mapper.reload_map()

    def test_known_code_maps_correctly(self):
        """HR_LOC maps to LOINC 8867-4."""
        from integration.canonical.mapper import map_observation
        from integration.ingest.hl7v2_parser import ParsedObservation

        obs = ParsedObservation(
            seq=1, local_code="HR_LOC", description="Heart rate",
            value=72.0, unit="/min", abnormal_flag=None,
            observation_ts="20240101120000",
        )
        mo = map_observation(obs)
        assert mo.mapped is True
        assert mo.loinc_code == "8867-4"
        assert mo.canonical_name == "Heart rate"
        assert mo.expected_unit == "/min"

    def test_unknown_code_not_mapped(self):
        """A code absent from code_map.yaml returns mapped=False."""
        from integration.canonical.mapper import map_observation
        from integration.ingest.hl7v2_parser import ParsedObservation

        obs = ParsedObservation(
            seq=1, local_code="UNKNOWN_CODE", description="",
            value=1.0, unit="x", abnormal_flag=None,
            observation_ts="20240101120000",
        )
        mo = map_observation(obs)
        assert mo.mapped is False
        assert mo.loinc_code is None

    def test_all_14_features_map(self):
        """All 14 CinC 2019 local codes must be in code_map.yaml."""
        from integration.canonical.mapper import get_map

        expected = {
            "HR_LOC", "SPO2_LOC", "RR_LOC", "SBP_LOC", "MAP_LOC",
            "TEMP_LOC", "GLU_LOC", "DBP_LOC", "CREA_LOC", "BUN_LOC",
            "K_LOC", "WBC_LOC", "PLT_LOC", "HGB_LOC",
        }
        assert expected.issubset(get_map().keys())

    def test_map_message_propagates_unmapped(self):
        """map_message() lists unmapped codes in MappedMessage.unmapped_codes."""
        from integration.canonical.mapper import map_message
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

        parsed = ParsedMessage(
            control_id="C001", sending_application="SA",
            sending_facility="SF", patient_id="P001",
            encounter_id="E001", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=[
                ParsedObservation(1, "HR_LOC", "HR", 72.0, "/min", None, "20240101120000"),
                ParsedObservation(2, "GHOST_CODE", "Ghost", 5.0, "x", None, "20240101120000"),
            ],
        )
        mm = map_message(parsed)
        assert "GHOST_CODE" in mm.unmapped_codes
        assert len(mm.observations) == 2
        assert mm.observations[0].mapped is True
        assert mm.observations[1].mapped is False

    def test_get_required_codes_nonempty(self):
        """At least HR_LOC and SPO2_LOC and SBP_LOC must be required."""
        from integration.canonical.mapper import get_required_codes
        req = get_required_codes()
        assert "HR_LOC" in req
        assert "SPO2_LOC" in req
        assert "SBP_LOC" in req


# ==================== SECTION 2: validator ======================================

class TestValidator:
    def setup_method(self):
        from integration.canonical import mapper
        mapper.reload_map()

    def _make_full_parsed(self, overrides: dict | None = None):
        """Return a ParsedMessage with all required observations at normal values."""
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation
        obs_list = [
            ParsedObservation(1,  "HR_LOC",   "HR",  72.0,  "/min",    None, "20240101120000"),
            ParsedObservation(2,  "SPO2_LOC", "SpO2", 98.0, "%",       None, "20240101120000"),
            ParsedObservation(3,  "SBP_LOC",  "SBP", 120.0, "mm[Hg]", None, "20240101120000"),
        ]
        if overrides:
            obs_list = [o for o in obs_list if o.local_code not in overrides]
            for local_code, val in overrides.items():
                obs_list.append(
                    ParsedObservation(99, local_code, "", val, "unit", None, "20240101120000")
                )
        return ParsedMessage(
            control_id="C002", sending_application="SA",
            sending_facility="SF", patient_id="P002",
            encounter_id="E002", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=obs_list,
        )

    def test_valid_message_passes(self):
        """Message with all required obs at in-range values: all_valid=True."""
        from integration.canonical.mapper import map_message
        from integration.canonical.validator import validate_message
        mm = map_message(self._make_full_parsed())
        result = validate_message(mm)
        assert result.all_valid is True
        assert len(result.quarantined) == 0
        assert len(result.message_reasons) == 0

    def test_out_of_range_quarantined(self):
        """HR=5000 (well above 300 max) is quarantined with OUT_OF_RANGE."""
        from integration.canonical.mapper import map_message
        from integration.canonical.validator import validate_message
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

        parsed = ParsedMessage(
            control_id="C003", sending_application="SA",
            sending_facility="SF", patient_id="P003",
            encounter_id="E003", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=[
                ParsedObservation(1, "HR_LOC", "HR", 5000.0, "/min", None, "20240101120000"),
                ParsedObservation(2, "SPO2_LOC", "SpO2", 98.0, "%", None, "20240101120000"),
                ParsedObservation(3, "SBP_LOC", "SBP", 120.0, "mm[Hg]", None, "20240101120000"),
            ],
        )
        mm = map_message(parsed)
        result = validate_message(mm)
        assert result.all_valid is False
        codes = [vo.observation.local_code for vo in result.quarantined]
        assert "HR_LOC" in codes
        reason_codes = [r.code for vo in result.quarantined for r in vo.reasons]
        assert "OUT_OF_RANGE" in reason_codes

    def test_unmapped_code_quarantined(self):
        """An unmapped local code is quarantined with UNMAPPED_CODE."""
        from integration.canonical.mapper import map_message
        from integration.canonical.validator import validate_message
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

        parsed = ParsedMessage(
            control_id="C004", sending_application="SA",
            sending_facility="SF", patient_id="P004",
            encounter_id="E004", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=[
                ParsedObservation(1, "HR_LOC", "HR", 72.0, "/min", None, "20240101120000"),
                ParsedObservation(2, "SPO2_LOC", "SpO2", 98.0, "%", None, "20240101120000"),
                ParsedObservation(3, "SBP_LOC", "SBP", 120.0, "mm[Hg]", None, "20240101120000"),
                ParsedObservation(4, "MYSTERY_CODE", "?", 1.0, "x", None, "20240101120000"),
            ],
        )
        mm = map_message(parsed)
        result = validate_message(mm)
        quarantined_codes = [vo.observation.local_code for vo in result.quarantined]
        assert "MYSTERY_CODE" in quarantined_codes

    def test_missing_required_observation(self):
        """Message missing HR_LOC raises MISSING_REQUIRED message-level reason."""
        from integration.canonical.mapper import map_message
        from integration.canonical.validator import validate_message
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

        parsed = ParsedMessage(
            control_id="C005", sending_application="SA",
            sending_facility="SF", patient_id="P005",
            encounter_id="E005", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=[
                # HR_LOC deliberately missing
                ParsedObservation(2, "SPO2_LOC", "SpO2", 98.0, "%", None, "20240101120000"),
                ParsedObservation(3, "SBP_LOC", "SBP", 120.0, "mm[Hg]", None, "20240101120000"),
            ],
        )
        mm = map_message(parsed)
        result = validate_message(mm)
        assert result.all_valid is False
        msg_reason_codes = [r.code for r in result.message_reasons]
        assert "MISSING_REQUIRED" in msg_reason_codes

    def test_unit_mismatch_is_warn_not_error(self):
        """Unit mismatch is WARN severity - observation still goes to valid."""
        from integration.canonical.mapper import map_message
        from integration.canonical.validator import validate_message
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

        parsed = ParsedMessage(
            control_id="C006", sending_application="SA",
            sending_facility="SF", patient_id="P006",
            encounter_id="E006", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=[
                # HR sent in bpm instead of /min (different string, same meaning)
                ParsedObservation(1, "HR_LOC", "HR", 72.0, "bpm", None, "20240101120000"),
                ParsedObservation(2, "SPO2_LOC", "SpO2", 98.0, "%", None, "20240101120000"),
                ParsedObservation(3, "SBP_LOC", "SBP", 120.0, "mm[Hg]", None, "20240101120000"),
            ],
        )
        mm = map_message(parsed)
        result = validate_message(mm)
        # HR_LOC should be in valid (WARN does not quarantine)
        valid_codes = [vo.observation.local_code for vo in result.valid]
        assert "HR_LOC" in valid_codes
        assert len(result.quarantined) == 0


# ==================== SECTION 3: quarantine store ===============================

class TestQuarantineStore:
    def test_store_quarantine_items_out_of_range(self):
        """store_quarantine_items() persists one row for an OUT_OF_RANGE obs."""
        from integration.canonical.mapper import map_message
        from integration.canonical.validator import validate_message
        from integration.canonical.quarantine import store_quarantine_items, QuarantineItem
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

        session = _make_session()
        parsed = ParsedMessage(
            control_id="C010", sending_application="SA",
            sending_facility="SF", patient_id="P010",
            encounter_id="E010", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=[
                ParsedObservation(1, "HR_LOC", "HR", 9999.0, "/min", None, "20240101120000"),
                ParsedObservation(2, "SPO2_LOC", "SpO2", 98.0, "%", None, "20240101120000"),
                ParsedObservation(3, "SBP_LOC", "SBP", 120.0, "mm[Hg]", None, "20240101120000"),
            ],
        )
        mm = map_message(parsed)
        result = validate_message(mm)
        items = store_quarantine_items(session, result, raw_sha256="abc123")
        session.commit()

        assert len(items) >= 1
        row = session.query(QuarantineItem).filter_by(reason_code="OUT_OF_RANGE").first()
        assert row is not None
        assert row.local_code == "HR_LOC"
        assert row.value == 9999.0
        assert row.replayed_ok is None

    def test_store_missing_required(self):
        """store_quarantine_items() persists MISSING_REQUIRED message-level rows."""
        from integration.canonical.mapper import map_message
        from integration.canonical.validator import validate_message
        from integration.canonical.quarantine import store_quarantine_items, QuarantineItem
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

        session = _make_session()
        parsed = ParsedMessage(
            control_id="C011", sending_application="SA",
            sending_facility="SF", patient_id="P011",
            encounter_id="E011", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=[
                # Only SPO2 and SBP present - HR_LOC missing (required)
                ParsedObservation(2, "SPO2_LOC", "SpO2", 98.0, "%", None, "20240101120000"),
                ParsedObservation(3, "SBP_LOC", "SBP", 120.0, "mm[Hg]", None, "20240101120000"),
            ],
        )
        mm = map_message(parsed)
        result = validate_message(mm)
        items = store_quarantine_items(session, result, raw_sha256="def456")
        session.commit()

        missing_rows = (
            session.query(QuarantineItem)
            .filter_by(reason_code="MISSING_REQUIRED")
            .all()
        )
        assert len(missing_rows) >= 1
        assert any("HR_LOC" in r.reason_detail for r in missing_rows)

    def test_mark_replayed(self):
        """mark_replayed() sets replayed_at and replayed_ok on the row."""
        from integration.canonical.mapper import map_message
        from integration.canonical.validator import validate_message
        from integration.canonical.quarantine import store_quarantine_items, mark_replayed, QuarantineItem
        from integration.ingest.hl7v2_parser import ParsedMessage, ParsedObservation

        session = _make_session()
        parsed = ParsedMessage(
            control_id="C012", sending_application="SA",
            sending_facility="SF", patient_id="P012",
            encounter_id="E012", message_ts="20240101120000",
            observation_ts="20240101120000",
            observations=[
                ParsedObservation(1, "HR_LOC", "HR", 9999.0, "/min", None, "20240101120000"),
                ParsedObservation(2, "SPO2_LOC", "SpO2", 98.0, "%", None, "20240101120000"),
                ParsedObservation(3, "SBP_LOC", "SBP", 120.0, "mm[Hg]", None, "20240101120000"),
            ],
        )
        mm = map_message(parsed)
        result = validate_message(mm)
        items = store_quarantine_items(session, result, raw_sha256="ghi789")
        session.commit()

        item_ids = [i.id for i in items]
        updated = mark_replayed(session, item_ids, ok=True)
        session.commit()

        assert updated == len(items)
        row = session.get(QuarantineItem, item_ids[0])
        assert row.replayed_ok is True
        assert row.replayed_at is not None


# ==================== SECTION 4: replay API =====================================

class TestReplayEndpoint:
    def _make_app_with_session(self):
        """Build a TestClient wired to a shared SQLite session.

        Uses a named temp file instead of :memory: so all connections
        (test code + FastAPI dependency) share the same database.
        """
        import tempfile
        import os
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from integration.ingest.replay import router
        from integration.canonical.quarantine import Base as QBase
        from integration.canonical.db import get_session

        # Named temp file - cleaned up when the test process exits
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        db_url = f"sqlite:///{tmp_path}"

        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        QBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

        app = FastAPI()
        app.include_router(router)

        # Override the get_session dependency to use the same engine
        def override_get_session():
            s = Session()
            try:
                yield s
            finally:
                s.close()

        app.dependency_overrides[get_session] = override_get_session

        return TestClient(app), Session

    def test_replay_clears_unmapped_code_when_map_updated(self):
        """
        Scenario: NEW_CODE_LOC arrives unmapped, gets quarantined.
        After reload_map() adding it, POST /ingest/replay -> passed=1.

        This test patches code_map in memory to simulate a YAML fix.
        """
        import pathlib
        import tempfile
        import yaml as _yaml
        from integration.canonical import mapper

        client, Session = self._make_app_with_session()

        # Build a quarantine item manually (simulating prior ingestion)
        from integration.canonical.quarantine import QuarantineItem
        session = Session()
        item = QuarantineItem(
            control_id="C020",
            patient_id="P020",
            encounter_id="E020",
            local_code="NEW_CODE_LOC",
            loinc_code=None,
            value=5.0,
            unit="mg/dL",
            reason_code="UNMAPPED_CODE",
            reason_detail="Local code 'NEW_CODE_LOC' has no entry in code_map.yaml",
            severity="ERROR",
            raw_sha256="sha_c020",
        )
        session.add(item)
        session.commit()
        item_id = item.id

        # Before map fix: replay should fail
        resp = client.post(
            "/ingest/replay",
            json={"item_ids": [item_id], "dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["still_failing"] == 1
        assert data["passed"] == 0
        assert data["dry_run"] is True

        # Now "fix" the map by injecting the new code into the in-memory map
        mapper._CODE_MAP["NEW_CODE_LOC"] = mapper.CodeMapEntry(
            loinc_code="99999-9",
            display_name="New Code",
            unit="mg/dL",
            min_value=0.0,
            max_value=100.0,
            required=False,
        )

        # After map fix: replay should pass (dry_run=False this time)
        resp2 = client.post(
            "/ingest/replay",
            json={"item_ids": [item_id]},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["passed"] == 1
        assert data2["still_failing"] == 0

        # Restore map to clean state
        del mapper._CODE_MAP["NEW_CODE_LOC"]

    def test_replay_no_ids_returns_400(self):
        """POST /ingest/replay with neither item_ids nor raw_sha256 -> 400."""
        client, _ = self._make_app_with_session()
        resp = client.post("/ingest/replay", json={})
        assert resp.status_code == 400

    def test_quarantine_list_endpoint(self):
        """GET /ingest/quarantine returns the stored item."""
        client, Session = self._make_app_with_session()
        from integration.canonical.quarantine import QuarantineItem

        session = Session()
        session.add(QuarantineItem(
            control_id="C021", patient_id="P021", encounter_id="E021",
            local_code="HR_LOC", loinc_code="8867-4",
            value=9999.0, unit="/min",
            reason_code="OUT_OF_RANGE",
            reason_detail="HR_LOC value 9999.0 is outside [0.0, 300.0]",
            severity="ERROR",
            raw_sha256="sha_c021",
        ))
        session.commit()

        resp = client.get("/ingest/quarantine")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 1
        assert items[0]["reason_code"] == "OUT_OF_RANGE"
        assert items[0]["replayed_ok"] is None

    def test_quarantine_filter_by_reason_code(self):
        """GET /ingest/quarantine?reason_code=UNMAPPED_CODE filters correctly."""
        client, Session = self._make_app_with_session()
        from integration.canonical.quarantine import QuarantineItem

        session = Session()
        session.add(QuarantineItem(
            control_id="C022", patient_id="P022", encounter_id="E022",
            local_code="GHOST", loinc_code=None,
            value=1.0, unit="x",
            reason_code="UNMAPPED_CODE",
            reason_detail="Local code 'GHOST' has no entry in code_map.yaml",
            severity="ERROR",
            raw_sha256="sha_c022",
        ))
        session.add(QuarantineItem(
            control_id="C023", patient_id="P023", encounter_id="E023",
            local_code="HR_LOC", loinc_code="8867-4",
            value=9999.0, unit="/min",
            reason_code="OUT_OF_RANGE",
            reason_detail="HR_LOC value 9999.0 is outside [0.0, 300.0]",
            severity="ERROR",
            raw_sha256="sha_c023",
        ))
        session.commit()

        resp = client.get("/ingest/quarantine?reason_code=UNMAPPED_CODE")
        assert resp.status_code == 200
        items = resp.json()
        assert all(i["reason_code"] == "UNMAPPED_CODE" for i in items)
        assert len(items) == 1
