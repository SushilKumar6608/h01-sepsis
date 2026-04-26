"""
evaluate_lstm_ae.py
Loads the trained LSTM AE and computes anomaly scores on test sets.
Reports AUC-ROC, AUC-PRC, and plots the score distributions.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from torch.utils.data import DataLoader, TensorDataset
from lstm_ae import LSTMAutoencoder

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH  = r"D:\Egna projekt\h01-sepsis\data\processed"
MODEL_PATH = r"D:\Egna projekt\h01-sepsis\models"
OUT_PATH   = r"D:\Egna projekt\h01-sepsis\notebooks"
BATCH_SIZE = 512
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_scores(model, X: np.ndarray) -> np.ndarray:
    """Run model on X in batches, return reconstruction error per window."""
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    scores  = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            err   = model.reconstruction_error(batch)
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)


def evaluate(scores: np.ndarray, labels: np.ndarray, name: str):
    """Print AUC-ROC and AUC-PRC for a test set."""
    auc_roc = roc_auc_score(labels, scores)
    auc_prc = average_precision_score(labels, scores)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  AUC-ROC:  {auc_roc:.4f}")
    print(f"  AUC-PRC:  {auc_prc:.4f}")
    print(f"  Positive windows (sepsis): {labels.sum()} / {len(labels)}")
    return auc_roc, auc_prc


def plot_results(scores_A, labels_A, scores_B, labels_B):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LSTM Autoencoder — Anomaly Detection Results', fontsize=14)

    for col, (scores, labels, name) in enumerate([
        (scores_A, labels_A, 'Set A'),
        (scores_B, labels_B, 'Set B')
    ]):
        # ROC curve
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_roc = roc_auc_score(labels, scores)
        axes[0, col].plot(fpr, tpr, color='steelblue', lw=2,
                          label=f'AUC-ROC = {auc_roc:.4f}')
        axes[0, col].plot([0,1],[0,1],'k--', lw=1)
        axes[0, col].set_title(f'{name} — ROC Curve')
        axes[0, col].set_xlabel('False Positive Rate')
        axes[0, col].set_ylabel('True Positive Rate')
        axes[0, col].legend()

        # Score distribution
        axes[1, col].hist(scores[labels==0], bins=80, alpha=0.6,
                          color='steelblue', label='Non-sepsis', density=True)
        axes[1, col].hist(scores[labels==1], bins=80, alpha=0.6,
                          color='tomato', label='Sepsis', density=True)
        axes[1, col].set_title(f'{name} — Anomaly Score Distribution')
        axes[1, col].set_xlabel('Reconstruction Error')
        axes[1, col].set_ylabel('Density')
        axes[1, col].legend()

    plt.tight_layout()
    out_file = os.path.join(OUT_PATH, 'lstm_ae_results.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out_file}")


def main():
    print(f"Device: {DEVICE}")

    # Load model
    X_train = np.load(os.path.join(DATA_PATH, 'X_train_A.npy'))
    model = LSTMAutoencoder(
        n_features=X_train.shape[2],
        seq_len=X_train.shape[1]
    ).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, 'lstm_ae_best.pt'),
        map_location=DEVICE))
    print("Model loaded.")

    # Load test sets
    X_test_A = np.load(os.path.join(DATA_PATH, 'X_test_A.npy'))
    y_test_A = np.load(os.path.join(DATA_PATH, 'y_test_A.npy'))
    X_test_B = np.load(os.path.join(DATA_PATH, 'X_test_B.npy'))
    y_test_B = np.load(os.path.join(DATA_PATH, 'y_test_B.npy'))

    # Compute scores
    print("Computing scores for Set A...")
    scores_A = compute_scores(model, X_test_A)
    print("Computing scores for Set B...")
    scores_B = compute_scores(model, X_test_B)

    # Evaluate
    evaluate(scores_A, y_test_A, 'Hospital A (Set A)')
    evaluate(scores_B, y_test_B, 'Hospital B (Set B)')

    # Save scores for later SHAP analysis
    np.save(os.path.join(DATA_PATH, 'scores_lstm_A.npy'), scores_A)
    np.save(os.path.join(DATA_PATH, 'scores_lstm_B.npy'), scores_B)
    print("\nScores saved.")

    # Plot
    plot_results(scores_A, y_test_A, scores_B, y_test_B)


if __name__ == '__main__':
    main()