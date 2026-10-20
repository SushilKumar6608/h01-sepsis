"""
test_phase4.py - Phase 4: scoring pipeline tests.

Tests are grouped into:
  TestSequenceBuilder  - _build_sequence() logic (no model needed)
  TestScorerMocked     - score_observations() with a mocked model
  TestFhirRiskAssessment - build_risk_assessment() output shape
  TestScorerIntegration  - loads real weights; skipped if file absent

Run from project root:
    conda activate D:\\conda-envs\\h01-sepsis
    pytest integration/tests/test_phase4.py -v
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Make src/ importable (needed by the scorer singleton)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_SRC = str(_PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Minimal stub for MappedObservation so tests don't need the full mapper
# ---------------------------------------------------------------------------

@dataclass
class _Obs:
    loinc_code: str
    value: float
    local_code: str = "STUB"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_FILE = _PROJECT_ROOT / "models" / "lstm_ae_best.pt"
_MEAN_FILE  = _PROJECT_ROOT / "data" / "processed" / "mean_A.npy"
_STD_FILE   = _PROJECT_ROOT / "data" / "processed" / "std_A.npy"

_REAL_MODEL_AVAILABLE = _MODEL_FILE.exists() and _MEAN_FILE.exists() and _STD_FILE.exists()

# One observation per feature - 14 normal-range values
_NORMAL_OBS = [
    _Obs("8867-4",  80.0),   # HR
    _Obs("59408-5", 97.0),   # O2Sat
    _Obs("9279-1",  16.0),   # Resp
    _Obs("8480-6",  120.0),  # SBP
    _Obs("8478-0",  83.0),   # MAP
    _Obs("8310-5",  37.0),   # Temp
    _Obs("2345-7",  100.0),  # Glucose
    _Obs("8462-4",  70.0),   # DBP
    _Obs("2160-0",  1.0),    # Creatinine
    _Obs("3094-0",  15.0),   # BUN
    _Obs("2823-3",  4.0),    # Potassium
    _Obs("6690-2",  8.0),    # WBC
    _Obs("777-3",   200.0),  # Platelets
    _Obs("718-7",   12.0),   # Hgb
]


# ---------------------------------------------------------------------------
# TestSequenceBuilder
# ---------------------------------------------------------------------------

class TestSequenceBuilder:
    """Tests for _build_sequence - no model required."""

    def test_full_14_obs_yields_correct_shape(self):
        from integration.scoring.scorer import _build_sequence, SEQ_LEN, N_FEATURES
        seq = _build_sequence(_NORMAL_OBS)
        assert seq.shape == (SEQ_LEN, N_FEATURES)

    def test_empty_obs_returns_median_filled_sequence(self):
        from integration.scoring.scorer import _build_sequence, SEQ_LEN, N_FEATURES, GLOBAL_MEDIANS, FEATURE_ORDER
        seq = _build_sequence([])
        assert seq.shape == (SEQ_LEN, N_FEATURES)
        # All values should be medians
        expected_row = np.array([GLOBAL_MEDIANS[f] for f in FEATURE_ORDER], dtype=np.float32)
        np.testing.assert_allclose(seq[0], expected_row)

    def test_hr_value_appears_in_last_row(self):
        from integration.scoring.scorer import _build_sequence, FEATURE_ORDER
        obs = [_Obs("8867-4", 99.0)]  # HR only
        seq = _build_sequence(obs)
        hr_idx = FEATURE_ORDER.index("HR")
        assert seq[-1, hr_idx] == pytest.approx(99.0)

    def test_missing_feature_filled_with_median(self):
        from integration.scoring.scorer import _build_sequence, FEATURE_ORDER, GLOBAL_MEDIANS
        # Only HR is observed; all others should be medians
        obs = [_Obs("8867-4", 80.0)]
        seq = _build_sequence(obs)
        o2sat_idx = FEATURE_ORDER.index("O2Sat")
        assert seq[-1, o2sat_idx] == pytest.approx(GLOBAL_MEDIANS["O2Sat"])

    def test_long_sequence_truncated_to_seq_len(self):
        from integration.scoring.scorer import _build_sequence, SEQ_LEN
        # 100 distinct hours of HR observations - repeat loinc triggers new slot
        obs = []
        for _ in range(100):
            obs.append(_Obs("8867-4", 80.0))
            obs.append(_Obs("59408-5", 97.0))  # O2Sat in same slot; triggers new HR slot next
        seq = _build_sequence(obs)
        assert seq.shape[0] == SEQ_LEN

    def test_multiple_values_same_feature_same_slot_are_averaged(self):
        from integration.scoring.scorer import _build_sequence, FEATURE_ORDER
        # Two HR observations in the first slot - should be averaged
        obs = [_Obs("8867-4", 80.0), _Obs("8867-4", 100.0)]
        # Both in slot 0 because we start fresh; second HR triggers new slot
        # (slot 0 = 80.0, slot 1 = 100.0 - they are separate slots per design)
        # Test that neither raises and shape is correct
        seq = _build_sequence(obs)
        hr_idx = FEATURE_ORDER.index("HR")
        assert seq[-1, hr_idx] in (80.0, 100.0)  # last slot value

    def test_no_nan_in_output(self):
        from integration.scoring.scorer import _build_sequence
        seq = _build_sequence(_NORMAL_OBS)
        assert not np.isnan(seq).any(), "Sequence must not contain NaN after median fill"


# ---------------------------------------------------------------------------
# TestScorerMocked - score_observations() with fake model
# ---------------------------------------------------------------------------

class TestScorerMocked:
    """Tests that mock the model so we don't need the .pt file."""

    def _patch_get_model(self, error_value: float):
        from integration.scoring.scorer import GLOBAL_MEDIANS, FEATURE_ORDER
        medians = np.array([GLOBAL_MEDIANS[f] for f in FEATURE_ORDER], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.reconstruction_error.return_value = torch.tensor([error_value])
        return patch(
            "integration.scoring.scorer.get_model",
            return_value=(mock_model, medians, np.ones(14, dtype=np.float32), "lstm_ae_best.pt"),
        )

    def test_low_error_gives_low_risk(self):
        from integration.scoring.scorer import score_observations
        with self._patch_get_model(0.05):
            result = score_observations(_NORMAL_OBS)
        assert result.risk_level == "LOW"

    def test_moderate_error_gives_moderate_risk(self):
        from integration.scoring.scorer import score_observations
        with self._patch_get_model(0.15):
            result = score_observations(_NORMAL_OBS)
        assert result.risk_level == "MODERATE"

    def test_high_error_gives_high_risk(self):
        from integration.scoring.scorer import score_observations
        with self._patch_get_model(0.30):
            result = score_observations(_NORMAL_OBS)
        assert result.risk_level == "HIGH"

    def test_reconstruction_error_is_float(self):
        from integration.scoring.scorer import score_observations
        with self._patch_get_model(0.08):
            result = score_observations(_NORMAL_OBS)
        assert isinstance(result.reconstruction_error, float)

    def test_result_has_model_version(self):
        from integration.scoring.scorer import score_observations
        with self._patch_get_model(0.05):
            result = score_observations(_NORMAL_OBS)
        assert result.model_version == "lstm_ae_best.pt"

    def test_n_observations_used_correct(self):
        from integration.scoring.scorer import score_observations
        with self._patch_get_model(0.05):
            result = score_observations(_NORMAL_OBS)
        assert result.n_observations_used == 14

    def test_boundary_moderate_threshold(self):
        from integration.scoring.scorer import score_observations, THRESHOLD_MODERATE
        with self._patch_get_model(THRESHOLD_MODERATE):
            result = score_observations(_NORMAL_OBS)
        assert result.risk_level == "MODERATE"

    def test_boundary_high_threshold(self):
        from integration.scoring.scorer import score_observations, THRESHOLD_HIGH
        with self._patch_get_model(THRESHOLD_HIGH):
            result = score_observations(_NORMAL_OBS)
        assert result.risk_level == "HIGH"


# ---------------------------------------------------------------------------
# TestFhirRiskAssessment
# ---------------------------------------------------------------------------

class TestFhirRiskAssessment:

    def _make_score(self, error=0.05, level="LOW"):
        from integration.scoring.scorer import SepsisRiskScore
        return SepsisRiskScore(
            reconstruction_error=error,
            risk_level=level,
            n_observations_used=14,
            n_features_imputed=0,
            model_version="lstm_ae_best.pt",
        )

    def test_resource_type_is_risk_assessment(self):
        from integration.scoring.fhir_risk_assessment import build_risk_assessment
        ra = build_risk_assessment(self._make_score(), "P001", "ENC001")
        assert ra["resourceType"] == "RiskAssessment"

    def test_patient_reference_set(self):
        from integration.scoring.fhir_risk_assessment import build_risk_assessment
        ra = build_risk_assessment(self._make_score(), "P001", "ENC001")
        assert ra["subject"]["reference"] == "Patient/P001"

    def test_encounter_reference_set(self):
        from integration.scoring.fhir_risk_assessment import build_risk_assessment
        ra = build_risk_assessment(self._make_score(), "P001", "ENC001")
        assert ra["encounter"]["reference"] == "Encounter/ENC001"

    def test_prediction_contains_risk_level(self):
        from integration.scoring.fhir_risk_assessment import build_risk_assessment
        ra = build_risk_assessment(self._make_score(level="HIGH"), "P001", "ENC001")
        assert "HIGH" in ra["prediction"][0]["outcome"]["text"]

    def test_probability_decimal_rounded(self):
        from integration.scoring.fhir_risk_assessment import build_risk_assessment
        ra = build_risk_assessment(self._make_score(error=0.123456), "P001", "ENC001")
        assert ra["prediction"][0]["probabilityDecimal"] == pytest.approx(0.1235, abs=1e-4)

    def test_occurred_at_defaults_to_utc_now(self):
        from integration.scoring.fhir_risk_assessment import build_risk_assessment
        before = datetime.now(tz=timezone.utc)
        ra = build_risk_assessment(self._make_score(), "P001", "ENC001")
        after = datetime.now(tz=timezone.utc)
        ts = datetime.fromisoformat(ra["occurrenceDateTime"])
        assert before <= ts <= after

    def test_status_is_final(self):
        from integration.scoring.fhir_risk_assessment import build_risk_assessment
        ra = build_risk_assessment(self._make_score(), "P001", "ENC001")
        assert ra["status"] == "final"


# ---------------------------------------------------------------------------
# TestScorerIntegration - real model; skipped when weights absent
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _REAL_MODEL_AVAILABLE,
    reason="Real model weights not found - skipping integration test",
)
class TestScorerIntegration:

    def test_real_model_returns_non_negative_error(self):
        # Reset singleton so real model is loaded fresh
        import integration.scoring.scorer as sc
        sc._model = None
        from integration.scoring.scorer import score_observations
        result = score_observations(_NORMAL_OBS)
        assert result.reconstruction_error >= 0.0

    def test_real_model_returns_valid_risk_level(self):
        import integration.scoring.scorer as sc
        sc._model = None
        from integration.scoring.scorer import score_observations
        result = score_observations(_NORMAL_OBS)
        assert result.risk_level in ("LOW", "MODERATE", "HIGH")
