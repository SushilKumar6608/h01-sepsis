import pandas as pd
import numpy as np
import os

SETA_PATH = r"D:\Egna projekt\h01-sepsis\data\raw\training_setA\training"
FEATURES = ['HR','O2Sat','Resp','SBP','MAP','Temp','Glucose',
            'DBP','Creatinine','BUN','Potassium','WBC','Platelets','Hgb']

# Check: how many patients have ALL NaN for every feature?
all_nan_patients = []
files = sorted([f for f in os.listdir(SETA_PATH) if f.endswith('.psv')])[:500]

for fname in files:
    df = pd.read_csv(os.path.join(SETA_PATH, fname), sep='|')
    feature_data = df[FEATURES]
    if feature_data.isnull().all().all():
        all_nan_patients.append(fname)
    elif feature_data.isnull().values.sum() > 0:
        # Check if median() returns NaN for any column
        bad_cols = [c for c in FEATURES if df[c].isnull().all()]
        if bad_cols:
            print(f"{fname}: columns entirely NaN: {bad_cols}")

print(f"\nPatients with ALL features entirely NaN: {len(all_nan_patients)}")
if all_nan_patients:
    print(f"Examples: {all_nan_patients[:5]}")