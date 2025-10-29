from __future__ import annotations

import torch
import torch.nn as nn

class MultiLabelHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, bias:bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F]
        returns: [B, C] logits
        """
        logits = self.fc(x)
        return logits