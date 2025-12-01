# ekgclf/diffusion/sample_diffusion.py

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ekgclf.models.diffusion.diffusion import SimpleUNet1D, SinusoidalTimeEmbedding

LOGGER = logging.getLogger("ekgclf.diffusion.sample")

def build_schedule_from_ckpt(ckpt: dict, device: torch.device):
    betas = torch.tensor(ckpt["betas"], dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

@torch.no_grad()
def p_sample_step(
    model: nn.Module,
    x_t: torch.Tensor,
    y: torch.Tensor,
    t_scalar: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    One reverse step x_t -> x_{t-1} using the DDPM update formula.
    """
    B = x_t.size(0)
    device = x_t.device

    t = torch.full((B,), t_scalar, device=device, dtype=torch.long)
    beta_t = betas[t_scalar]
    alpha_t = alphas[t_scalar]
    alpha_bar_t = alphas_cumprod[t_scalar]

    # predict noise
    eps_theta = model(x_t, y)  # (B, C)

    # DDPM mean
    # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
    coef1 = 1.0 / torch.sqrt(alpha_t)
    coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)

    mean = coef1 * (x_t - coef2 * eps_theta)

    if t_scalar == 0:
        return mean  # no noise at final step

    noise = torch.randn_like(x_t)
    sigma_t = torch.sqrt(beta_t)
    return mean + sigma_t * noise

@torch.no_grad()
def sample_label_segments(
    model: nn.Module,
    label_name: str,
    label_idx: int,
    labels_map: Dict[str, int],
    n_samples: int,
    batch_size: int,
    segment_len: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    out_dir: Path,
    in_channels: int,
):
    """
    Generate n_samples segments for a single label and save to out_dir.
    """
    device = next(model.parameters()).device
    T_steps = betas.shape[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Sampling %d segments for label '%s' into %s", n_samples, label_name, out_dir)

    num_labels = len(labels_map)
    done = 0
    sample_id = 0

    while done < n_samples:
        cur = min(batch_size, n_samples - done)
        # start from Gaussian noise
        x = torch.randn(cur, in_channels, segment_len, device=device)

        # fixed one-hot label for this batch
        y = torch.zeros(cur, num_labels, device=device)
        y[:, label_idx] = 1.0

        for t in reversed(range(T_steps)):
            x = p_sample_step(model, x, y, t, betas, alphas, alphas_cumprod)

        # x now approx samples from p(x0 | y)
        x_np = x.detach().cpu().numpy()  # (B, C, T)
        for b in range(cur):
            arr = x_np[b]  # [C, T]
            # sanitize label for filesystem-safe filename (keep folder as-is)
            safe_label = label_name.replace("/", "_").replace("\\", "_")
            fname = out_dir / f"{safe_label}_{sample_id:05d}.npy"
            np.save(fname, arr.astype(np.float32))
            sample_id += 1

        done += cur

    LOGGER.info("Finished sampling label '%s'. Total segments: %d", label_name, n_samples)

def _setup_device(device_str: str | None) -> torch.device:
    if device_str and device_str.lower() != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return dev
    return torch.device("cpu")

@hydra.main(version_base=None, config_path="../../../../configs", config_name="diffusion")
def main(cfg: DictConfig):
    CFG_ROOT = Path(get_original_cwd())

    from ekgclf.settings import setup_logging
    setup_logging(CFG_ROOT / "configs" / "logging.yaml")
    LOGGER.info("Diffusion sampling config:\n%s", OmegaConf.to_yaml(cfg))

    ckpt_path = Path(cfg.sampling.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    labels_map: Dict[str, int] = ckpt["labels_map"]
    rare_labels: List[str] = ckpt.get("rare_labels", [])
    num_steps_ckpt = ckpt["num_steps"]

    device = _setup_device(cfg.sampling.device)
    LOGGER.info("Using device: %s", device)

    in_channels = cfg.sampling.in_channels
    base_ch = cfg.sampling.base_channels
    segment_len = cfg.sampling.segment_len

    model = SimpleUNet1D(
        in_ch=in_channels,
        base_ch=base_ch,
        num_classes=len(labels_map),
    )
    
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    betas, alphas, alphas_cumprod = build_schedule_from_ckpt(ckpt, device=device)
    if num_steps_ckpt != betas.shape[0]:
        LOGGER.warning("num_steps in ckpt (%d) != len(betas) (%d)", num_steps_ckpt, betas.shape[0])
    LOGGER.info("Loaded diffusion schedule with %d steps.", betas.shape[0])

    # Which labels to sample?
    if cfg.sampling.labels:
        target_labels = [lab for lab in cfg.sampling.labels if lab in labels_map]
        if not target_labels:
            raise RuntimeError("None of cfg.sampling.labels are present in labels_map.")
    else:
        # default: use rare_labels saved in ckpt
        if not rare_labels:
            raise RuntimeError("Checkpoint has no rare_labels and cfg.sampling.labels is empty.")
        target_labels = [lab for lab in rare_labels if lab in labels_map]

    out_root = Path(cfg.sampling.out_root)

    n_per_label = cfg.sampling.samples_per_label
    batch_size = cfg.sampling.batch_size

    for lab in target_labels:
        idx = labels_map[lab]
        lab_dir = out_root / lab
        sample_label_segments(
            model=model,
            label_name=lab,
            label_idx=idx,
            labels_map=labels_map,
            n_samples=n_per_label,
            batch_size=batch_size,
            segment_len=segment_len,
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            out_dir=lab_dir,
            in_channels=in_channels,
        )

    LOGGER.info("All requested labels sampled.")

if __name__ == "__main__":
    main()
