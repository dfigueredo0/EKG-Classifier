from __future__ import annotations

import torch

def confidence_gating(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Return accept mask [B] based on max probability per sample."""
    conf, _ = probs.max(dim=1)
    return conf >= threshold

def risk_coverage(scores: torch.Tensor, labels: torch.Tensor, coverages):
    """Compute risk (1 - F1 macro) at different coverage levels by thresholding max prob."""
    from sklearn.metrics import f1_score

    probs = scores
    max_conf, _ = probs.max(dim=1)
    sorted_idx = torch.argsort(max_conf, descending=True)
    risks = []
    for cov in coverages:
        k = int(max(1, cov * len(max_conf)))
        idx = sorted_idx[:k]
        y_true = labels[idx].cpu().numpy()
        y_pred = (probs[idx].cpu().numpy() >= 0.5).astype(int)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        risks.append(1 - f1_macro)
    return risks