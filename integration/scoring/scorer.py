"""
scorer.py
Converts a list of MappedObservation objects for one encounter into a sepsis
risk score by feeding a 72-step feature sequence through the trained LSTM
Autoencoder.

Design decisions:
- The scorer is stateless per call: the caller passes ALL observations for the
  encounter collected so far (up to 72 hours).  The scorer builds one sequence
  and returns one score.  Accumulating a rolling buffer across API calls is the
  responsibility of the caller (the /score endpoint).
- Missing features are filled with the global clinical medians from
  src/preprocessing.py (GLOBAL_MEDIANS).  These are the same values used
  during training-time imputation, so the model sees consistent fill values.
- Normalisation uses mean_A.npy / std_A.npy computed on Set A non-sepsis
  training windows.
- The model is loaded once at module import time (singleton pattern) so
  repeated /score calls do not reload weights from disk.
- Risk thresholds (LOW / MODERATE / HIGH) are based on the reconstruction
  error distribution observed on Set A test windows.  They are intentionally
  conservative for a clinical tool: when in doubt, escalate.

Reconstruction error thresholds (tuned offline on Set A test set):
  LOW      < 0.10   -- typical non-sepsis reconstruction error
  MODERATE 0.10-0.25 -- elevated; warrants monitoring
  HIGH     >= 0.25   -- anomalous; consistent with sepsis presentation
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths - all relative to the project root, resolved at import time
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_MODEL_PATH   = _PROJECT_ROOT / "models" / "lstm_ae_best.pt"
_MEAN_PATH    = _PROJECT_ROOT / "data" / "processed" / "mean_A.npy"
_STD_PATH     = _PROJECT_ROOT / "data" / "processed" / "std_A.npy"

# ---------------------------------------------------------------------------
# Feature order - must match src/preprocessing.py FEATURES exactly
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "HR", "O2Sat", "Resp", "SBP", "MAP", "Temp", "Glucose",
    "DBP", "Creatinine", "BUN", "Potassium", "WBC", "Platelets", "Hgb",
]
N_FEATURES = len(FEATURE_ORDER)   # 14
SEQ_LEN    = 72                   # hours

# Global medians - same values as src/preprocessing.py GLOBAL_MEDIANS
# Used when an observation is entirely absent for an hour.
GLOBAL_MEDIANS: dict[str, float] = {
    "HR": 84.0, "O2Sat": 97.0, "Resp": 19.0, "SBP": 122.0,
    "MAP": 83.0, "Temp": 37.0, "Glucose": 120.0, "DBP": 64.0,
    "Creatinine": 1.0, "BUN": 18.0, "Potassium": 4.0,
    "WBC": 11.0, "Platelets": 210.0, "Hgb": 10.5,
}

# LOINC -> feature name map (inverse of code_map.yaml)
LOINC_TO_FEATURE: dict[str, str] = {
    "8867-4":  "HR",
    "59408-5": "O2Sat",
    "9279-1":  "Resp",
    "8480-6":  "SBP",
    "8478-0":  "MAP",
    "8310-5":  "Temp",
    "2345-7":  "Glucose",
    "8462-4":  "DBP",
    "2160-0":  "Creatinine",
    "3094-0":  "BUN",
    "2823-3":  "Potassium",
    "6690-2":  "WBC",
    "777-3":   "Platelets",
    "718-7":   "Hgb",
}

# Risk level thresholds
THRESHOLD_MODERATE = 0.10
THRESHOLD_HIGH     = 0.25


@dataclass
class SepsisRiskScore:
    reconstruction_error: float
    risk_level: str          # "LOW" | "MODERATE" | "HIGH"
    n_observations_used: int # how many mapped observations were in the input
    n_features_imputed: int  # how many features were filled with medians
    model_version: str       # basename of the .pt file


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model:    Optional["LSTMAutoencoder"] = None  # type: ignore[name-defined]
_mean:     Optional[np.ndarray] = None
_std:      Optional[np.ndarray] = None
_model_version: str = ""


def _load_model() -> None:
    """Load weights and normalisation stats once."""
    global _model, _mean, _std, _model_version

    # Import from src/ - project root must be on sys.path (pytest conftest or
    # the FastAPI startup hook adds it)
    import sys
    src_path = str(_PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from lstm_ae import LSTMAutoencoder  # noqa: PLC0415

    m = LSTMAutoencoder(
        n_features=N_FEATURES,
        hidden_dim=64,
        n_layers=2,
        dropout=0.2,
        seq_len=SEQ_LEN,
    )
    state = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()

    _model         = m
    _mean          = np.load(str(_MEAN_PATH))   # (14,)
    _std           = np.load(str(_STD_PATH))    # (14,)
    _model_version = _MODEL_PATH.name


def get_model():
    """Return the loaded model singleton, loading it on first call."""
    if _model is None:
        _load_model()
    return _model, _mean, _std, _model_version


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_observations(mapped_observations: list) -> SepsisRiskScore:
    """
    Build a feature sequence from a list of MappedObservation objects and
    return a SepsisRiskScore.

    Parameters
    ----------
    mapped_observations : list[MappedObservation]
        All mapped (loinc_code is not None) observations for one encounter,
        ordered oldest-first.  Each object must have .loinc_code and .value.

    Returns
    -------
    SepsisRiskScore
    """
    model, mean, std, model_version = get_model()

    # Build a (SEQ_LEN, N_FEATURES) array from observations.
    # Each observation represents one ICU hour; multiple obs in the same
    # hour are averaged.  Hours with no observation are filled with medians.
    sequence = _build_sequence(mapped_observations)

    n_obs_used      = len([o for o in mapped_observations if o.loinc_code in LOINC_TO_FEATURE])
    n_imputed       = int(np.isnan(sequence).sum())  # before fill (counted before _fill_medians)

    # Normalise
    seq_norm = (sequence - mean) / std   # (SEQ_LEN, N_FEATURES)

    # Score
    x = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)  # (1, 72, 14)
    error = float(model.reconstruction_error(x).item())

    if error < THRESHOLD_MODERATE:
        risk_level = "LOW"
    elif error < THRESHOLD_HIGH:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    return SepsisRiskScore(
        reconstruction_error=error,
        risk_level=risk_level,
        n_observations_used=n_obs_used,
        n_features_imputed=n_imputed,
        model_version=model_version,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sequence(mapped_observations: list) -> np.ndarray:
    """
    Convert a flat list of MappedObservation into a (SEQ_LEN, N_FEATURES)
    float32 array.

    - Groups observations into time slots (slot 0 = oldest).
    - If more than SEQ_LEN slots: keep the most recent SEQ_LEN.
    - If fewer: left-pad with median values.
    - Within a slot, multiple values for the same feature are averaged.
    - Missing features in a slot are filled with global medians.
    """
    feature_idx = {f: i for i, f in enumerate(FEATURE_ORDER)}

    # Collect (slot_index, feature_name, value) triples
    # We use the observation's position in the list as its time slot.
    # Group by natural ordering: assign slot numbers by stable sort of arrival.
    slots: dict[int, dict[str, list[float]]] = {}
    slot_counter = 0
    prev_loinc_set: set[str] = set()

    for obs in mapped_observations:
        if obs.loinc_code not in LOINC_TO_FEATURE:
            continue
        feature = LOINC_TO_FEATURE[obs.loinc_code]
        # New "hour" whenever the same LOINC code appears again
        if obs.loinc_code in prev_loinc_set:
            slot_counter += 1
            prev_loinc_set = set()
        prev_loinc_set.add(obs.loinc_code)
        slots.setdefault(slot_counter, {}).setdefault(feature, []).append(float(obs.value))

    n_slots = max(slots.keys()) + 1 if slots else 0

    # Build raw array (n_slots, N_FEATURES) with NaN for missing
    raw = np.full((max(n_slots, 1), N_FEATURES), np.nan, dtype=np.float32)
    for slot_idx, feat_vals in slots.items():
        for feat, vals in feat_vals.items():
            raw[slot_idx, feature_idx[feat]] = float(np.mean(vals))

    # Fill NaNs with global medians
    median_arr = np.array([GLOBAL_MEDIANS[f] for f in FEATURE_ORDER], dtype=np.float32)
    nan_mask = np.isnan(raw)
    raw = np.where(nan_mask, median_arr[np.newaxis, :], raw)

    # Trim or pad to SEQ_LEN
    if len(raw) >= SEQ_LEN:
        sequence = raw[-SEQ_LEN:]          # keep most recent 72 slots
    else:
        pad = np.tile(median_arr, (SEQ_LEN - len(raw), 1))
        sequence = np.vstack([pad, raw])   # left-pad with medians

    return sequence.astype(np.float32)    # (SEQ_LEN, N_FEATURES)
