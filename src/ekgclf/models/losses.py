from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsSmooth(nn.Module):
    def __init__(self, smoothing: float = 0.0, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="mean")

class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()