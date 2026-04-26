"""
preprocessing.py
Converts raw PhysioNet 2019 .psv files into normalized, windowed numpy arrays
ready for LSTM Autoencoder and TPP model training.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Feature configuration ──────────────────────────────────────────────────
TIER1 = ['HR', 'O2Sat', 'Resp', 'SBP', 'MAP', 'Temp', 'Glucose']
TIER2 = ['DBP', 'Creatinine', 'BUN', 'Potassium', 'WBC', 'Platelets', 'Hgb']
FEATURES = TIER1 + TIER2          # 14 features total
MAX_SEQ_LEN = 72                   # hours — covers ~97% of patients
WINDOW_STEP = 1                    # sliding window step in hours

# ── Paths ──────────────────────────────────────────────────────────────────
SETA_PATH = r"D:\Egna projekt\h01-sepsis\data\raw\training_setA\training"
SETB_PATH = r"D:\Egna projekt\h01-sepsis\data\raw\training_setB\training_setB"
OUT_PATH  = r"D:\Egna projekt\h01-sepsis\data\processed"


def load_patient(filepath: str) -> pd.DataFrame:
    """Load a single .psv file and return a DataFrame with FEATURES + SepsisLabel."""
    df = pd.read_csv(filepath, sep='|')
    cols = FEATURES + ['SepsisLabel']
    return df[cols].copy()

# Global fallback medians — used when a patient has an entirely missing column
GLOBAL_MEDIANS = {
    'HR': 84.0, 'O2Sat': 97.0, 'Resp': 19.0, 'SBP': 122.0,
    'MAP': 83.0, 'Temp': 37.0, 'Glucose': 120.0, 'DBP': 64.0,
    'Creatinine': 1.0, 'BUN': 18.0, 'Potassium': 4.0,
    'WBC': 11.0, 'Platelets': 210.0, 'Hgb': 10.5
}

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      1. Forward-fill (carry last known value forward)
      2. Backward-fill (fill leading NaNs with first valid value)
      3. Fill any remaining NaNs with global clinical medians
    """
    df = df.ffill().bfill()
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(GLOBAL_MEDIANS[col])
    return df


def extract_windows(df: pd.DataFrame, patient_id: str):
    """
    Slide a MAX_SEQ_LEN window across the patient's ICU stay.
    Returns:
      windows      — list of (MAX_SEQ_LEN, n_features) arrays
      labels       — list of ints: 1 if any SepsisLabel==1 in window, else 0
      patient_ids  — list of str for traceability
    """
    feature_arr = df[FEATURES].values        # (T, 14)
    label_arr   = df['SepsisLabel'].values   # (T,)
    T = len(feature_arr)

    windows, labels, ids = [], [], []

    if T <= MAX_SEQ_LEN:
        # Short stay — pad with zeros at the front
        pad_len = MAX_SEQ_LEN - T
        window  = np.vstack([np.zeros((pad_len, len(FEATURES))), feature_arr])
        label   = int(label_arr.max())
        windows.append(window)
        labels.append(label)
        ids.append(patient_id)
    else:
        # Long stay — slide window across entire stay
        for start in range(0, T - MAX_SEQ_LEN + 1, WINDOW_STEP):
            end    = start + MAX_SEQ_LEN
            window = feature_arr[start:end]
            label  = int(label_arr[start:end].max())
            windows.append(window)
            labels.append(label)
            ids.append(f"{patient_id}_w{start}")

    return windows, labels, ids


def normalize(train_windows: np.ndarray):
    """
    Compute mean and std from training windows only.
    Returns normalized array + (mean, std) for reuse on test set.
    """
    # train_windows shape: (N, MAX_SEQ_LEN, n_features)
    flat  = train_windows.reshape(-1, train_windows.shape[-1])
    mean  = flat.mean(axis=0)
    std   = flat.std(axis=0)
    std[std == 0] = 1.0          # avoid division by zero
    normalized = (train_windows - mean) / std
    return normalized, mean, std


def process_folder(folder: str, split: str, mean=None, std=None):
    """
    Process all .psv files in a folder.
    split = 'train' or 'test'
    For 'train': only use non-sepsis patients, compute normalization stats.
    For 'test':  use all patients, apply existing mean/std.
    Returns windows array, labels array, ids list, (mean, std).
    """
    all_windows, all_labels, all_ids = [], [], []

    files = sorted([f for f in os.listdir(folder) if f.endswith('.psv')])

    for fname in tqdm(files, desc=f"{split} — {os.path.basename(folder)}"):
        fpath      = os.path.join(folder, fname)
        patient_id = fname.replace('.psv', '')
        df         = load_patient(fpath)

        # Training: skip sepsis patients entirely
        if split == 'train' and df['SepsisLabel'].max() == 1:
            continue

        patient_has_sepsis = int(df['SepsisLabel'].max())
        df = impute(df)
        windows, labels, ids = extract_windows(df, patient_id)
        # Override: label ALL windows from a sepsis patient as positive
        if patient_has_sepsis == 1:
            labels = [1] * len(labels)
        
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_ids.extend(ids)

    X = np.array(all_windows, dtype=np.float32)   # (N, 72, 14)
    y = np.array(all_labels,  dtype=np.int8)       # (N,)

    if split == 'train':
        X, mean, std = normalize(X)
    else:
        X = (X - mean) / std

    return X, y, all_ids, mean, std


def run():
    os.makedirs(OUT_PATH, exist_ok=True)

    print("=== Processing Set A — Train (non-sepsis only) ===")
    X_train_A, y_train_A, ids_train_A, mean_A, std_A = process_folder(
        SETA_PATH, split='train')
    print(f"  Train windows: {X_train_A.shape}, labels: {y_train_A.sum()} positive (should be 0)")

    print("\n=== Processing Set A — Test (all patients) ===")
    X_test_A, y_test_A, ids_test_A, _, _ = process_folder(
        SETA_PATH, split='test', mean=mean_A, std=std_A)
    print(f"  Test windows:  {X_test_A.shape}, sepsis windows: {y_test_A.sum()}")

    print("\n=== Processing Set B — Test (all patients, using Set A stats) ===")
    X_test_B, y_test_B, ids_test_B, _, _ = process_folder(
        SETB_PATH, split='test', mean=mean_A, std=std_A)
    print(f"  Set B windows: {X_test_B.shape}, sepsis windows: {y_test_B.sum()}")

    print("\nSaving to disk...")
    np.save(os.path.join(OUT_PATH, 'X_train_A.npy'), X_train_A)
    np.save(os.path.join(OUT_PATH, 'y_train_A.npy'), y_train_A)
    np.save(os.path.join(OUT_PATH, 'X_test_A.npy'),  X_test_A)
    np.save(os.path.join(OUT_PATH, 'y_test_A.npy'),  y_test_A)
    np.save(os.path.join(OUT_PATH, 'X_test_B.npy'),  X_test_B)
    np.save(os.path.join(OUT_PATH, 'y_test_B.npy'),  y_test_B)
    np.save(os.path.join(OUT_PATH, 'mean_A.npy'),     mean_A)
    np.save(os.path.join(OUT_PATH, 'std_A.npy'),      std_A)
    print(f"All arrays saved to {OUT_PATH}")
    print("\n=== Preprocessing complete ===")


if __name__ == '__main__':
    run()