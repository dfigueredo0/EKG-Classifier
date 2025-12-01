from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

def auroc_per_label(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # assume shape (n_samples, n_labels)
    n_labels = y_true.shape[1]
    per_label = []

    for i in range(n_labels):
        y_i = y_true[:, i]
        p_i = y_prob[:, i]

        # skip labels with only one class present
        if y_i.max() == y_i.min():
            per_label.append(np.nan)
        else:
            per_label.append(roc_auc_score(y_i, p_i))

    per_label = np.array(per_label, dtype=float)

    macro = np.nanmean(per_label)
    micro = roc_auc_score(y_true.ravel(), y_prob.ravel())

    return {
        "per_label": per_label,
        "macro": float(macro),
        "micro": float(micro),
    }

def f1_per_label(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_labels = y_true.shape[1]
    per_label = []

    for i in range(n_labels):
        y_i = y_true[:, i]
        yhat_i = y_pred[:, i]

        if y_i.max() == y_i.min():
            per_label.append(np.nan)
        else:
            per_label.append(f1_score(y_i, yhat_i))

    per_label = np.array(per_label, dtype=float)

    macro = np.nanmean(per_label)
    micro = f1_score(y_true.ravel(), y_pred.ravel())

    return {
        "per_label": per_label,
        "macro": float(macro),
        "micro": float(micro),
    }

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins [0,1]."""
    confidences = y_prob.ravel()
    labels = y_true.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)
