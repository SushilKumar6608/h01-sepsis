"""
explain_lstm_ae.py
SHAP explainability for the LSTM Autoencoder.
Identifies which clinical features drive high anomaly scores.
"""

import os
import numpy as np
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_ae import LSTMAutoencoder

DATA_PATH  = r"D:\Egna projekt\h01-sepsis\data\processed"
MODEL_PATH = r"D:\Egna projekt\h01-sepsis\models"
OUT_PATH   = r"D:\Egna projekt\h01-sepsis\notebooks"

FEATURES = ['HR','O2Sat','Resp','SBP','MAP','Temp','Glucose',
            'DBP','Creatinine','BUN','Potassium','WBC','Platelets','Hgb']

DEVICE = torch.device('cpu')  # SHAP works best on CPU

def reconstruction_error_flat(X_flat):
    """
    Wrapper for SHAP: takes (N, 72*14) flat input,
    returns (N,) reconstruction error.
    """
    X = torch.tensor(
        X_flat.reshape(-1, 72, 14), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        err = model.reconstruction_error(X)
    return err.numpy()

def main():
    global model

    # Load model
    X_train = np.load(os.path.join(DATA_PATH, 'X_train_A.npy'))
    model = LSTMAutoencoder(
        n_features=14, seq_len=72).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, 'lstm_ae_best.pt'),
        map_location=DEVICE))
    model.eval()
    print("Model loaded.")

    # Load test data and scores
    X_test_A = np.load(os.path.join(DATA_PATH, 'X_test_A.npy'))
    y_test_A = np.load(os.path.join(DATA_PATH, 'y_test_A.npy'))
    scores_A = np.load(os.path.join(DATA_PATH, 'scores_lstm_A.npy'))

    # Select top 100 highest-scoring sepsis windows for explanation
    sepsis_idx   = np.where(y_test_A == 1)[0]
    top_idx      = sepsis_idx[np.argsort(scores_A[sepsis_idx])[-100:]]
    X_explain    = X_test_A[top_idx]   # (100, 72, 14)

    # Background: 200 random normal windows
    normal_idx   = np.where(y_test_A == 0)[0]
    bg_idx       = np.random.RandomState(42).choice(normal_idx, 200, replace=False)
    X_background = X_train[bg_idx[:200]] if len(bg_idx) < 200 else X_test_A[bg_idx]

    # Flatten for SHAP (72 timesteps * 14 features = 1008 dims)
    X_explain_flat = X_explain.reshape(100, -1)
    X_bg_flat      = X_background[:200].reshape(200, -1)

    print("Running SHAP KernelExplainer (this takes ~5 minutes)...")
    explainer   = shap.KernelExplainer(reconstruction_error_flat, X_bg_flat)
    shap_values = explainer.shap_values(X_explain_flat, nsamples=100)
    # shap_values: (100, 1008)

    # Aggregate SHAP by feature (sum over 72 timesteps, mean over 100 windows)
    shap_3d    = np.abs(shap_values).reshape(100, 72, 14)
    shap_feat  = shap_3d.mean(axis=(0, 1))   # (14,) — mean |SHAP| per feature

    # Bar plot
    sorted_idx = np.argsort(shap_feat)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        [FEATURES[i] for i in sorted_idx[::-1]],
        shap_feat[sorted_idx[::-1]],
        color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Mean |SHAP value| — contribution to anomaly score')
    ax.set_title('LSTM AE — Feature Importance for Sepsis Detection\n(Top 100 highest-scoring sepsis windows, Set A)')
    plt.tight_layout()
    out_file = os.path.join(OUT_PATH, 'shap_feature_importance.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"SHAP plot saved to {out_file}")

    # Print ranking
    print("\nFeature importance ranking:")
    for rank, i in enumerate(sorted_idx):
        print(f"  {rank+1:2d}. {FEATURES[i]:<12} {shap_feat[i]:.6f}")

if __name__ == '__main__':
    main()