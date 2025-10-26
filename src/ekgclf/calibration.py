from __future__ import annotations

import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for multi-label logits."""

    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # T=1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_t)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 1000, lr: float = 1e-2):
        opt = torch.optim.LBFGS([self.log_t], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(torch.exp(self.log_t).item())
