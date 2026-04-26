"""
lstm_ae.py
LSTM Autoencoder for unsupervised anomaly detection on ICU event sequences.
Trained on normal (non-sepsis) patients only.
Anomaly score = mean reconstruction error per window.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (hidden, _) = self.lstm(x)
        # hidden: (n_layers, batch, hidden_dim)
        return hidden[-1]  # take last layer: (batch, hidden_dim)


class Decoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_layers: int,
                 dropout: float, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, z):
        # z: (batch, hidden_dim) — repeat across time steps
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        # z_repeated: (batch, seq_len, hidden_dim)
        out, _ = self.lstm(z_repeated)
        # out: (batch, seq_len, hidden_dim)
        return self.output_layer(out)  # (batch, seq_len, n_features)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 72
    ):
        super().__init__()
        self.encoder = Encoder(n_features, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(n_features, hidden_dim, n_layers, dropout, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def reconstruction_error(self, x):
        """Per-window mean squared error — used as anomaly score."""
        with torch.no_grad():
            x_hat = self.forward(x)
            error = ((x - x_hat) ** 2).mean(dim=(1, 2))  # (batch,)
        return error