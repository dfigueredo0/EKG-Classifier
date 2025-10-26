from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

def auroc_per_label(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    out = []
    for i in range(y_true.shape[1]):
        try:
            out.append(roc_auc_score(y_true[:, i], y_score[:, i]))
        except Exception:
            pass
        
    return float(np.nanmean(out)) if out else float("nan")

def f1_per_label(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for i in range(y_true.shape[1]):
        p, r, f, _ = precision_recall_fscore_support(y_true[:, i], y_pred[:, i], zero_division=0, average="binary")
        out[i] = f
    out["macro"] = np.mean([v for k, v in out.items() if isinstance(k, int)])
    out["micro"] = f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    return out

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
