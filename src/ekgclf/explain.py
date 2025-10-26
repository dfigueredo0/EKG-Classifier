from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from ekgclf.settings import setup_logging

LOGGER = logging.getLogger("ekgclf.explain")

def _forward_probs(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return sigmoid probs [B, L] given inputs [B, C, T]."""
    logits = model(x)  # [B, L]
    return torch.sigmoid(logits)

@torch.no_grad()
def _to_device(*tensors: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in tensors)

def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,          # [B, C, T], requires_grad not needed by caller
    baseline: Optional[torch.Tensor] = None,  # [B, C, T] or None -> zeros
    steps: int = 64,
    internal_batch_size: Optional[int] = None,
    objective: str = "sum_probs",  # "sum_probs" | "max_prob" | "sum_logits"
) -> torch.Tensor:
    """
    Pure PyTorch Integrated Gradients.
    Returns attributions with same shape as inputs: [B, C, T].
    """
    assert steps >= 2, "steps must be >= 2"

    device = next(model.parameters()).device
    model.eval()

    x = inputs.to(device)
    b = baseline.to(device) if baseline is not None else torch.zeros_like(x, device=device)
    path = x - b

    # Riemann sums of gradients along the straight-line path
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    grads = torch.zeros_like(x)

    # We need grads, so enable autograd within the loop
    for alpha in alphas:
        x_alpha = (b + alpha * path).clone().detach().requires_grad_(True)

        probs_or_logits = _forward_probs(model, x_alpha) if objective in {"sum_probs", "max_prob"} else model(x_alpha)
        if objective == "sum_probs":
            scalar = probs_or_logits.sum()
        elif objective == "max_prob":
            # maximize confidence per sample; sum over batch for a scalar
            scalar = probs_or_logits.max(dim=1).values.sum()
        elif objective == "sum_logits":
            scalar = probs_or_logits.sum()
        else:
            raise ValueError(f"Unsupported objective: {objective}")

        model.zero_grad(set_to_none=True)
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        scalar.backward()

        grads = grads + (x_alpha.grad if x_alpha.grad is not None else torch.zeros_like(x_alpha))

    # Average gradient and scale by (input - baseline)
    grads = grads / float(steps)
    attributions = path * grads
    return attributions.detach()

@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig):
    CFG_ROOT = Path(get_original_cwd())
    setup_logging(CFG_ROOT / "configs" / "logging.yaml")

    # Load checkpoint & model
    ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
    labels_map = ckpt["labels_map"]
    num_labels = len(labels_map)

    from ekgclf.models.resnet1d import ResNet1D
    from ekgclf.models.head_multilabel import MultiLabelHead

    body = ResNet1D()
    head = MultiLabelHead(256, num_labels, bias_init=None)
    model = torch.nn.Sequential(body, head).eval()

    # Load a small test slice
    import json as _json
    from ekgclf.data.splitter import patient_level_split

    index = _json.load(open("data/processed/ptbxl_index.json", "r", encoding="utf-8"))
    pids = [m["patient_id"] for m in index]
    sp = patient_level_split(pids, 0.8, 0.1, 0.1, seed=1337)
    samples = [index[i] for i in sp.test[:5]]

    out_dir = Path("reports/explanations")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for s in samples:
        # [T, C] -> [1, C, T]
        x_np = np.load(s["npz"])["signals"].astype(np.float32)
        x = torch.from_numpy(x_np).permute(1, 0).unsqueeze(0)  # [1, C, T]
        baseline = torch.zeros_like(x)
        
        attributions = integrated_gradients(
            model=model,
            inputs=x,
            baseline=baseline,
            steps=64,
            objective="sum_probs",
        )  # [1, C, T]

        sal = attributions.abs().squeeze(0).cpu().numpy()  # [C, T]
        lead_importance = sal.mean(axis=1).tolist()
        time_importance = sal.mean(axis=0)
        payload = {
            "npz": s["npz"],
            "lead_importance": lead_importance,
            "time_importance_topk_idxs": np.argsort(time_importance)[-10:][::-1].tolist(),
        }
        with open(out_dir / (Path(s["npz"]).stem + "_ig.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    LOGGER.info("Wrote %d explanations to %s", len(samples), out_dir)

if __name__ == "__main__":
    main()