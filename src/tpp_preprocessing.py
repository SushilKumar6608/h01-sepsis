"""
tpp_preprocessing.py
Converts raw .psv files into Neural TPP format:
each patient = sequence of (time, event_type) pairs
where an event = a non-NaN measurement of a clinical feature.

Output format compatible with EasyTPP framework.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
FEATURES = ['HR','O2Sat','Resp','SBP','MAP','Temp','Glucose',
            'DBP','Creatinine','BUN','Potassium','WBC','Platelets','Hgb']

# Event type index: 0-13 maps to each feature
EVENT2IDX = {f: i for i, f in enumerate(FEATURES)}
NUM_EVENT_TYPES = len(FEATURES)  # 14

SETA_PATH = r"D:\Egna projekt\h01-sepsis\data\raw\training_setA\training"
SETB_PATH = r"D:\Egna projekt\h01-sepsis\data\raw\training_setB\training_setB"
OUT_PATH  = r"D:\Egna projekt\h01-sepsis\data\tpp"

MIN_SEQ_LEN = 5    # discard patients with fewer than 5 events
MAX_SEQ_LEN = 300  # cap very long sequences


def patient_to_tpp(df: pd.DataFrame, patient_id: str):
    """
    Convert one patient DataFrame to a TPP event sequence.
    Each non-NaN cell = one event at that hour.
    Returns list of dicts: [{time_since_start, time_since_last, event_type}, ...]
    """
    events = []
    for hour, row in df.iterrows():
        for feat in FEATURES:
            if not pd.isna(row[feat]):
                events.append({
                    'hour': float(hour),
                    'event_type': EVENT2IDX[feat]
                })

    # Sort by time (should already be sorted but be safe)
    events.sort(key=lambda x: x['hour'])

    if len(events) < MIN_SEQ_LEN:
        return None

    # Cap length
    events = events[:MAX_SEQ_LEN]

    # Compute inter-event times
    times = [e['hour'] for e in events]
    time_since_start = times  # absolute time from ICU admission
    time_since_last  = [0.0] + [times[i] - times[i-1] for i in range(1, len(times))]

    return {
        'patient_id':        patient_id,
        'time_since_start':  time_since_start,
        'time_since_last':   time_since_last,
        'event_type':        [e['event_type'] for e in events],
        'seq_len':           len(events)
    }


def process_folder(folder: str, split: str):
    """
    Process all patients in a folder.
    split='train': non-sepsis only
    split='test': all patients, returns labels too
    """
    sequences = []
    labels    = []
    files     = sorted([f for f in os.listdir(folder) if f.endswith('.psv')])

    for fname in tqdm(files, desc=f"TPP {split} — {os.path.basename(folder)}"):
        fpath      = os.path.join(folder, fname)
        patient_id = fname.replace('.psv', '')
        df         = pd.read_csv(fpath, sep='|')

        has_sepsis = int(df['SepsisLabel'].max())

        if split == 'train' and has_sepsis == 1:
            continue

        seq = patient_to_tpp(df[FEATURES], patient_id)
        if seq is None:
            continue

        seq['label'] = has_sepsis
        sequences.append(seq)
        labels.append(has_sepsis)

    return sequences, labels


def save_json(sequences, path):
    with open(path, 'w') as f:
        json.dump(sequences, f)
    print(f"Saved {len(sequences)} sequences to {path}")


def main():
    os.makedirs(OUT_PATH, exist_ok=True)

    print("=== TPP Preprocessing ===")

    print("\nProcessing Set A — Train (non-sepsis only)...")
    train_seqs, train_labels = process_folder(SETA_PATH, 'train')
    print(f"  Train sequences: {len(train_seqs)}")
    print(f"  Avg seq length:  {np.mean([s['seq_len'] for s in train_seqs]):.1f} events")

    print("\nProcessing Set A — Test (all patients)...")
    testA_seqs, testA_labels = process_folder(SETA_PATH, 'test')
    print(f"  Test A sequences: {len(testA_seqs)}")
    print(f"  Sepsis patients:  {sum(testA_labels)}")

    print("\nProcessing Set B — Test (all patients)...")
    testB_seqs, testB_labels = process_folder(SETB_PATH, 'test')
    print(f"  Test B sequences: {len(testB_seqs)}")
    print(f"  Sepsis patients:  {sum(testB_labels)}")

    # Save
    save_json(train_seqs, os.path.join(OUT_PATH, 'train_setA.json'))
    save_json(testA_seqs, os.path.join(OUT_PATH, 'test_setA.json'))
    save_json(testB_seqs, os.path.join(OUT_PATH, 'test_setB.json'))

    # Summary stats
    print("\n=== Summary ===")
    all_lens = [s['seq_len'] for s in train_seqs]
    print(f"Train seq length — min: {min(all_lens)}, "
          f"max: {max(all_lens)}, "
          f"median: {int(np.median(all_lens))}, "
          f"mean: {np.mean(all_lens):.1f}")
    print(f"Num event types: {NUM_EVENT_TYPES}")
    print("\nTPP preprocessing complete.")


if __name__ == '__main__':
    main()