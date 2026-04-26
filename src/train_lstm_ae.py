"""
train_lstm_ae.py
Trains the LSTM Autoencoder on non-sepsis windows from Set A.
Saves best model checkpoint to models/lstm_ae_best.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from lstm_ae import LSTMAutoencoder

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH  = r"D:\Egna projekt\h01-sepsis\data\processed"
MODEL_PATH = r"D:\Egna projekt\h01-sepsis\models"
BATCH_SIZE = 256
EPOCHS     = 50
LR         = 1e-3
HIDDEN_DIM = 64
N_LAYERS   = 2
DROPOUT    = 0.2
VAL_SPLIT  = 0.1        # 10% of train set for validation
PATIENCE   = 5          # early stopping patience
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    os.makedirs(MODEL_PATH, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Load data ──────────────────────────────────────────────────────────
    X_train = np.load(os.path.join(DATA_PATH, 'X_train_A.npy'))
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)

    # Train / val split
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train windows: {train_size} | Val windows: {val_size}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = LSTMAutoencoder(
        n_features=X_train.shape[2],
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        seq_len=X_train.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= train_size

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * len(batch)
        val_loss /= val_size

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping + save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_PATH, 'lstm_ae_best.pt'))
            print(f"           ↑ Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()