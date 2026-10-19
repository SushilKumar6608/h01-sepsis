"""
Phase 0 — Golden fixture generator
===================================
Reads two raw PhysioNet .psv patient files (one non-sepsis, one sepsis),
runs the same preprocessing as src/preprocessing.py, scores with the
trained LSTM AE, and saves the results to integration/fixtures/.

Run from the repo root with the h01-sepsis conda env active:

    python integration/create_fixtures.py

Outputs
-------
integration/fixtures/raw_nonsepsis.psv       copy of the chosen non-sepsis patient file
integration/fixtures/raw_sepsis.psv          copy of the chosen sepsis patient file
integration/fixtures/golden_windows.parquet  normalized windows for both patients
integration/fixtures/golden_labels.npy       patient-level labels (0 or 1)
integration/fixtures/golden_scores.npy       LSTM AE anomaly scores per window
integration/fixtures/fixture_meta.json       run metadata (patient ids, shapes, AE checkpoint)

These fixtures are the ground truth that Phase 3 must reproduce.
"""

import os
import sys
import json
import shutil
import pathlib
import numpy as np
import pandas as pd
import torch

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent          # h01-sepsis/
SRC         = REPO_ROOT / "src"
DATA_RAW_A  = REPO_ROOT / "data" / "raw" / "training_setA" / "training"
DATA_PROC   = REPO_ROOT / "data" / "processed"
MODEL_PATH  = REPO_ROOT / "models" / "lstm_ae_best.pt"
FIXTURE_DIR = REPO_ROOT / "integration" / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))

# ── feature spec (must match src/preprocessing.py exactly) ─────────────────────
FEATURES = [
    "HR", "O2Sat", "Resp", "SBP", "MAP", "Temp", "Glucose",   # Tier 1
    "DBP", "Creatinine", "BUN", "Potassium", "WBC", "Platelets", "Hgb",  # Tier 2
]
N_FEATURES  = len(FEATURES)
MAX_SEQ_LEN = 72
WINDOW_STEP = 1

GLOBAL_MEDIANS = {
    "HR": 84.0, "O2Sat": 97.0, "Resp": 19.0, "SBP": 122.0,
    "MAP": 83.0, "Temp": 37.0, "Glucose": 120.0, "DBP": 64.0,
    "Creatinine": 1.0, "BUN": 18.0, "Potassium": 4.0,
    "WBC": 11.0, "Platelets": 210.0, "Hgb": 10.5,
}


# ── helpers ────────────────────────────────────────────────────────────────────

def find_patient(psv_dir: pathlib.Path, want_sepsis: bool) -> pathlib.Path | None:
    """Scan psv_dir and return the first file matching the sepsis requirement."""
    for f in sorted(psv_dir.glob("p*.psv")):
        df = pd.read_csv(f, sep="|")
        is_sepsis = df.get("SepsisLabel", pd.Series([0])).max() == 1
        if is_sepsis == want_sepsis:
            return f
    return None


def load_patient(path: pathlib.Path) -> tuple[np.ndarray, int]:
    """
    Load one .psv file and return (array of shape (T, 14), sepsis_label).
    Applies the same imputation as preprocessing.py:
      forward-fill -> backward-fill -> global medians fallback.
    Does NOT normalize here; normalization is applied after.
    """
    df = pd.read_csv(path, sep="|")
    label = int(df.get("SepsisLabel", pd.Series([0])).max())

    df = df[FEATURES].copy()

    # ffill -> bfill
    df = df.ffill().bfill()

    # global medians for columns that are still entirely NaN
    for col in FEATURES:
        if df[col].isna().all():
            df[col] = GLOBAL_MEDIANS[col]
        else:
            df[col] = df[col].fillna(GLOBAL_MEDIANS[col])

    return df.values.astype(np.float32), label


def normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score with the Set-A non-sepsis stats.  std=0 guard: set to 1."""
    safe_std = np.where(std == 0, 1.0, std)
    return (arr - mean) / safe_std


def make_windows(arr: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window segmentation identical to preprocessing.py:
      - if T < 72: zero-pad the front, return one window
      - if T >= 72: slide with step=1
    Returns (windows of shape (N, 72, 14), labels of shape (N,)).
    """
    T = arr.shape[0]
    if T < MAX_SEQ_LEN:
        pad = np.zeros((MAX_SEQ_LEN - T, N_FEATURES), dtype=np.float32)
        win = np.concatenate([pad, arr], axis=0)[np.newaxis]  # (1, 72, 14)
    else:
        indices = range(0, T - MAX_SEQ_LEN + 1, WINDOW_STEP)
        win = np.stack([arr[i:i + MAX_SEQ_LEN] for i in indices])  # (N, 72, 14)

    labels = np.full(len(win), label, dtype=np.int32)
    return win, labels


# ── model loading ──────────────────────────────────────────────────────────────

def load_model():
    """Import LSTMAutoencoder from src/ and load the trained checkpoint."""
    from lstm_ae import LSTMAutoencoder
    model = LSTMAutoencoder(n_features=N_FEATURES, hidden_dim=64, n_layers=2,
                            dropout=0.2, seq_len=MAX_SEQ_LEN)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def score_windows(model, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Returns per-window anomaly scores (mean reconstruction error across
    all time steps and features), shape (N,).
    """
    scores = []
    t = torch.from_numpy(windows)
    with torch.no_grad():
        for i in range(0, len(t), batch_size):
            batch = t[i:i + batch_size]
            recon = model(batch)
            err = ((batch - recon) ** 2).mean(dim=(1, 2))  # (batch,)
            scores.append(err.numpy())
    return np.concatenate(scores)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 0: Golden fixture generator ===\n")

    # 1. Check prereqs
    missing = []
    for p in [DATA_RAW_A, DATA_PROC / "mean_A.npy", DATA_PROC / "std_A.npy", MODEL_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR — missing required files:")
        for m in missing:
            print(f"  {m}")
        print("\nMake sure you have run src/preprocessing.py and src/train_lstm_ae.py first.")
        sys.exit(1)

    # 2. Load normalization stats
    mean_A = np.load(DATA_PROC / "mean_A.npy")
    std_A  = np.load(DATA_PROC / "std_A.npy")
    print(f"Loaded norm stats: mean shape {mean_A.shape}, std shape {std_A.shape}")

    # 3. Find one non-sepsis and one sepsis patient from Set A
    print("Scanning Set A for patients ...")
    path_nonsep = find_patient(DATA_RAW_A, want_sepsis=False)
    path_sep    = find_patient(DATA_RAW_A, want_sepsis=True)

    if path_nonsep is None or path_sep is None:
        print("ERROR — could not find required patient files in", DATA_RAW_A)
        sys.exit(1)

    print(f"  Non-sepsis patient : {path_nonsep.name}")
    print(f"  Sepsis patient     : {path_sep.name}")

    # 4. Copy raw files into fixtures/
    shutil.copy(path_nonsep, FIXTURE_DIR / "raw_nonsepsis.psv")
    shutil.copy(path_sep,    FIXTURE_DIR / "raw_sepsis.psv")
    print("Copied raw .psv files to integration/fixtures/")

    # 5. Preprocess both patients
    all_windows, all_labels = [], []
    for path in [path_nonsep, path_sep]:
        raw_arr, lbl = load_patient(path)
        norm_arr = normalize(raw_arr, mean_A, std_A)
        wins, lbls = make_windows(norm_arr, lbl)
        all_windows.append(wins)
        all_labels.append(lbls)
        print(f"  {path.name}: {raw_arr.shape[0]} hours → {len(wins)} windows, label={lbl}")

    windows = np.concatenate(all_windows, axis=0)    # (N, 72, 14)
    labels  = np.concatenate(all_labels,  axis=0)    # (N,)
    print(f"\nTotal windows: {windows.shape}")

    # 6. Save windows as Parquet (feature × time flattened for easy inspection)
    # Shape (N, 72*14+1): columns = feature_t0, feature_t1, ..., label
    flat = windows.reshape(len(windows), -1)
    col_names = [f"{feat}_t{t}" for t in range(MAX_SEQ_LEN) for feat in FEATURES]
    df_out = pd.DataFrame(flat, columns=col_names)
    df_out["label"] = labels
    parquet_path = FIXTURE_DIR / "golden_windows.parquet"
    df_out.to_parquet(parquet_path, index=False)
    print(f"Saved golden_windows.parquet  ({parquet_path.stat().st_size // 1024} KB)")

    # 7. Save labels as .npy (quick load for Phase 3 parity test)
    np.save(FIXTURE_DIR / "golden_labels.npy", labels)
    print("Saved golden_labels.npy")

    # 8. Score with LSTM AE
    print("\nLoading LSTM AE checkpoint ...")
    model = load_model()
    print("Scoring windows ...")
    scores = score_windows(model, windows)
    np.save(FIXTURE_DIR / "golden_scores.npy", scores)
    print(f"Saved golden_scores.npy  (min={scores.min():.4f}, max={scores.max():.4f}, "
          f"mean={scores.mean():.4f})")

    # 9. Write metadata
    meta = {
        "nonsepsis_patient": path_nonsep.name,
        "sepsis_patient":    path_sep.name,
        "n_windows":         int(len(windows)),
        "window_shape":      list(windows.shape),
        "n_sepsis_windows":  int(labels.sum()),
        "n_nonsepsis_windows": int((labels == 0).sum()),
        "score_min":  float(scores.min()),
        "score_max":  float(scores.max()),
        "score_mean": float(scores.mean()),
        "ae_checkpoint": str(MODEL_PATH),
        "mean_A_npy":    str(DATA_PROC / "mean_A.npy"),
        "std_A_npy":     str(DATA_PROC / "std_A.npy"),
        "features":      FEATURES,
        "n_features":    N_FEATURES,
        "max_seq_len":   MAX_SEQ_LEN,
        "window_step":   WINDOW_STEP,
        "dataset":       "PhysioNet_CinC_2019",
    }
    with open(FIXTURE_DIR / "fixture_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Saved fixture_meta.json")

    print("\n=== Phase 0 DONE ===")
    print("Golden fixture files are in integration/fixtures/")
    print("  raw_nonsepsis.psv")
    print("  raw_sepsis.psv")
    print("  golden_windows.parquet")
    print("  golden_labels.npy")
    print("  golden_scores.npy")
    print("  fixture_meta.json")
    print("\nNext: create branch integration-v2 and commit these files.")


if __name__ == "__main__":
    main()
