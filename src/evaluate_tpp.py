"""
evaluate_tpp.py
Evaluates trained SAHP on Set A and Set B.
Anomaly score = negative log-likelihood per event (length-normalized).
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

OUT_PATH  = r"D:\Egna projekt\h01-sepsis\notebooks"
DATA_PATH = r"D:\Egna projekt\h01-sepsis\data\processed"
TPP_DATA  = r"D:\Egna projekt\h01-sepsis\data\tpp\easytpp"


def load_labels(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return np.array([d['label'] for d in data])


def score_sequences(runner, split='dev'):
    """Compute per-sequence log-likelihood per event using SAHP internals."""
    loader = runner._data_loader.get_loader(split, shuffle=False)
    model  = runner.model
    model.eval()

    seq_loglikes = []

    with torch.no_grad():
        for batch in loader:
            time_seqs        = batch['time_seqs']
            time_delta_seqs  = batch['time_delta_seqs']
            type_seqs        = batch['type_seqs']
            seq_non_pad_mask = batch['seq_non_pad_mask']
            attention_mask   = batch['attention_mask']

            # Forward pass — encoder output
            enc_out = model.forward(
                time_seqs[:, :-1],
                time_delta_seqs[:, :-1],
                type_seqs[:, :-1],
                attention_mask[:, :-1, :-1]
            )

            # Intensity at event times
            cell_t = model.state_decay(
                encode_state=enc_out,
                duration_t=time_delta_seqs[:, 1:, None]
            )
            lambda_at_event = model.softplus(cell_t)

            # MC samples for integral
            sample_dtimes = model.make_dtime_loss_samples(time_delta_seqs[:, 1:])
            state_t_sample = model.compute_states_at_sample_times(
                encode_state=enc_out,
                sample_dtimes=sample_dtimes
            )
            lambda_t_sample = model.softplus(state_t_sample)

            # Mask for valid (non-pad) positions
            mask = seq_non_pad_mask[:, 1:]  # (batch, seq_len-1)

            # Event log-likelihood: log intensity at true event type
            type_idx = type_seqs[:, 1:].long().clamp(0, model.num_event_types - 1)
            lambda_true = lambda_at_event.gather(
                2, type_idx.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len-1)
            event_ll_per_seq = (torch.log(lambda_true + 1e-8) * mask).sum(dim=1)

            # Non-event log-likelihood (integral approximation)
            total_lambda_sample = lambda_t_sample.sum(dim=-1)  # (batch, seq_len-1, num_sample)
            dt = time_delta_seqs[:, 1:]
            integral_per_step = total_lambda_sample.mean(dim=-1) * dt
            non_event_ll_per_seq = (integral_per_step * mask).sum(dim=1)

            # Per-sequence loglike
            per_seq_ll = event_ll_per_seq - non_event_ll_per_seq  # (batch,)

            # Normalize by number of events to remove length bias
            n_events = mask.sum(dim=1).float().clamp(min=1)
            per_seq_ll_normalized = per_seq_ll / n_events

            seq_loglikes.extend(per_seq_ll_normalized.cpu().numpy().tolist())

    return np.array(seq_loglikes)


def main():
    CONFIG = r"D:\Egna projekt\h01-sepsis\configs\sahp_eval_A.yaml"

    print("Loading model...")
    config = Config.build_from_yaml_file(CONFIG, experiment_id='SAHP_eval_A')
    runner = Runner.build_from_config(config)

    print("Scoring Set A patients...")
    scores_A = score_sequences(runner, split='dev')
    labels_A = load_labels(os.path.join(TPP_DATA, 'dev.json'))

    print("Scoring Set B patients...")
    scores_B = score_sequences(runner, split='test')
    labels_B = load_labels(os.path.join(TPP_DATA, 'test.json'))

    print(f"\nSet A — sequences: {len(scores_A)}, labels: {len(labels_A)}")
    print(f"Set B — sequences: {len(scores_B)}, labels: {len(labels_B)}")

    # Anomaly score = negative loglike per event
    anom_A = -scores_A
    anom_B = -scores_B

    # Diagnostics
    for name, anom, scores, labels in [('Set A', anom_A, scores_A, labels_A),
                                        ('Set B', anom_B, scores_B, labels_B)]:
        print(f"\n{name} diagnostics:")
        print(f"  Score range: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        print(f"  Sepsis mean loglike/event:     {scores[labels==1].mean():.4f}")
        print(f"  Non-sepsis mean loglike/event: {scores[labels==0].mean():.4f}")

    # Evaluate
    for name, anom, labels in [('Set A', anom_A, labels_A),
                                ('Set B', anom_B, labels_B)]:
        auc_roc = roc_auc_score(labels, anom)
        auc_prc = average_precision_score(labels, anom)
        print(f"\n{'='*40}")
        print(f"  TPP SAHP — {name}")
        print(f"{'='*40}")
        print(f"  AUC-ROC:  {auc_roc:.4f}")
        print(f"  AUC-PRC:  {auc_prc:.4f}")
        print(f"  Patients: {len(labels)} | Sepsis: {int(labels.sum())}")

    # Save scores
    np.save(os.path.join(DATA_PATH, 'scores_tpp_A.npy'), anom_A)
    np.save(os.path.join(DATA_PATH, 'scores_tpp_B.npy'), anom_B)
    print("\nScores saved.")

    # Plot ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('SAHP TPP — Anomaly Detection Results', fontsize=13)
    for ax, (name, anom, labels) in zip(axes, [('Set A', anom_A, labels_A),
                                                ('Set B', anom_B, labels_B)]):
        fpr, tpr, _ = roc_curve(labels, anom)
        auc = roc_auc_score(labels, anom)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC-ROC = {auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(f'{name} — ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, 'tpp_results.png'), dpi=150)
    plt.close()
    print("Plot saved.")


if __name__ == '__main__':
    main()