from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from ekgclf.models.head_multilabel import MultiLabelHead
from ekgclf.models.resnet1d import ResNet1D

@dataclass
class InferenceResult:
    probs: np.ndarray  # [B, L]
    abstain: np.ndarray  # [B]
    lead_summary: Optional[List[float]] = None

class InferenceEngine:
    def __init__(self, ckpt_path: str = "checkpoints/best.pt", device: Optional[str] = None, threshold: float = 0.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.labels_map = ckpt["labels_map"]
        num_labels = len(self.labels_map)
        self.model = torch.nn.Sequential(ResNet1D(), MultiLabelHead(256, num_labels)).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.threshold = threshold

    def predict(self, signals: np.ndarray) -> InferenceResult:
        """
        signals: float32 array [B, T, C] at 500 Hz, z-scored, required lead order.
        """
        x = torch.from_numpy(signals).permute(0, 2, 1).float().to(self.device)  # [B,C,T]
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        conf = probs.max(axis=1)
        abstain = (conf < self.threshold).astype(bool)
        return InferenceResult(probs=probs, abstain=abstain)
