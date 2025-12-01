from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None
    
def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class DomainAdversarialNN(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    
    def forward(self, x, lambd=1.0):
        return self.net(grad_reverse(x, lambd))
    
class DomainAdaptationModel(nn.Module):
    def __init__(self, body: nn.Module, cls_head: nn.Module, dom_head: nn.Module):
        super().__init__()
        self.body = body
        self.cls_head = cls_head
        self.dom_head = dom_head

    def forward(self, x: torch.Tensor, lambda_grl: float = 0.0):
        feat = self.body(x)                   # [B, 256]
        logits_y = self.cls_head(feat)        # [B, num_classes]

        rev_feat = grad_reverse(feat, lambda_grl)
        logits_d = self.dom_head(rev_feat)    # [B, 2]

        return logits_y, logits_d