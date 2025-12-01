from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

from ekgclf.settings import DataConfig, load_yaml
from ekgclf.data.splitter import patient_level_split
from ekgclf.models.diffusion.diffusion import SimpleUNet1D, SinusoidalTimeEmbedding

LOGGER = logging.getLogger("ekgclf.diffusion")


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor


def make_linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)


def q_sample(x0: torch.Tensor, t: torch.Tensor, sched: DiffusionSchedule, noise=None):
    """
    Diffusion forward process: q(x_t | x_0)
    x0: (B, C, T)
    t: (B,) with integer time steps
    """
    if noise is None:
        noise = torch.randn_like(x0)
    alphas_cumprod = sched.alphas_cumprod.to(x0.device)
    alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise, noise

class DiffusionEKGDataset(Dataset):
    """
    Yields random segments from records that contain one of the target labels.
    - Uses processed NPZ files from ptbxl_index.json / mitbih_index.json.
    - Returns (segment [C,T_seg], one_hot_label [num_labels]).
    """

    def __init__(
        self,
        index: List[dict],
        labels_map: Dict[str, int],
        target_labels: List[str],
        segment_len: int = 2048,
        max_segments_per_record: int = 4,
    ):
        self.index = []
        self.labels_map = labels_map
        self.segment_len = segment_len
        self.max_segments_per_record = max_segments_per_record

        target_set = set(target_labels)
        for meta in index:
            labs = [lab for lab in meta["labels"] if lab in labels_map]
            if not labs:
                continue
            if not (target_set & set(labs)):
                continue
            # keep only relevant labels
            meta = dict(meta)  # shallow copy
            meta["labels"] = labs
            self.index.append(meta)

        if not self.index:
            raise RuntimeError("No records contain the requested target labels!")

        LOGGER.info("Diffusion dataset: %d records match target labels %s", len(self.index), target_labels)

    def __len__(self):
        # logical length = records * max_segments_per_record
        return len(self.index) * self.max_segments_per_record

    def __getitem__(self, i: int):
        ridx = i // self.max_segments_per_record
        meta = self.index[ridx]
        npz = np.load(meta["npz"])
        sig = npz["signals"]  # [T, C]
        T_total, C = sig.shape
        # time-major → channel-first
        x = sig.T  # [C, T]

        if T_total < self.segment_len:
            # pad if too short
            pad = self.segment_len - T_total
            x = F.pad(torch.from_numpy(x), (0, pad))
        else:
            # random crop
            start = np.random.randint(0, T_total - self.segment_len + 1)
            x = torch.from_numpy(x[:, start : start + self.segment_len])

        # one-hot label: for now, pick first target label present
        y = torch.zeros(len(self.labels_map), dtype=torch.float32)
        for lab in meta["labels"]:
            if lab in self.labels_map:
                y[self.labels_map[lab]] = 1.0

        return x.float(), y

def _setup_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        dev = torch.device("cpu")
    return dev


@hydra.main(version_base=None, config_path="../../../../configs", config_name="diffusion")
def main(cfg: DictConfig):
    CFG_ROOT = Path(get_original_cwd())

    # Logging
    from ekgclf.settings import setup_logging
    setup_logging(CFG_ROOT / "configs" / "logging.yaml")
    LOGGER.info("Diffusion config:\n%s", OmegaConf.to_yaml(cfg))

    # Data config (for processed paths)
    data_cfg = DataConfig(**load_yaml(CFG_ROOT / "configs" / "data.yaml"))
    proc_dir = Path(data_cfg.paths.processed)
    ptb_idx_path = proc_dir / "ptbxl_index.json"
    mit_idx_path = proc_dir / "mitbih_index.json"

    if not ptb_idx_path.exists():
        raise FileNotFoundError(f"Missing processed PTB-XL index at {ptb_idx_path}")

    ptb_index = _json.load(open(ptb_idx_path, "r", encoding="utf-8"))
    for m in ptb_index:
        m.setdefault("source", "ptbxl")

    mit_index = []
    if mit_idx_path.exists():
        mit_index = _json.load(open(mit_idx_path, "r", encoding="utf-8"))
        for m in mit_index:
            m.setdefault("source", "mitbih")

    index = ptb_index + mit_index
    if len(index) == 0:
        raise RuntimeError("No samples found in processed indexes")

    # Build labels map from corpus (same as train.py)
    labels = sorted({l for m in index for l in m["labels"]})
    labels_map = {l: i for i, l in enumerate(labels)}
    LOGGER.info("Num labels (PTB-XL ∪ MIT-BIH): %d", len(labels))

    # Determine rare labels
    if cfg.diffusion.rare_labels:
        rare_labels = [lab for lab in cfg.diffusion.rare_labels if lab in labels_map]
        if not rare_labels:
            raise RuntimeError("None of cfg.diffusion.rare_labels are present in data.")
    else:
        # auto-select labels with frequency below threshold
        counts = np.zeros(len(labels_map), dtype=np.int64)
        for m in index:
            for lab in m["labels"]:
                if lab in labels_map:
                    counts[labels_map[lab]] += 1
        thr = cfg.diffusion.auto_max_count
        rare_labels = [lab for lab, idx in labels_map.items() if counts[idx] <= thr]
        LOGGER.info("Auto-selected rare labels (count ≤ %d): %s", thr, rare_labels)

    # patient-level split (reuse splitter for reproducibility)
    def pid_key(m):
        return f'{m.get("source","unk")}:{m["patient_id"]}'

    pids = [pid_key(m) for m in index]
    sp = patient_level_split(pids, train=0.8, val=0.1, test=0.1, seed=cfg.diffusion.seed)

    train_index = [index[i] for i in sp.train]
    val_index = [index[i] for i in sp.val]

    seg_len = cfg.diffusion.segment_len
    max_seg = cfg.diffusion.max_segments_per_record

    ds_train = DiffusionEKGDataset(
        train_index, labels_map, rare_labels, segment_len=seg_len, max_segments_per_record=max_seg
    )
    ds_val = DiffusionEKGDataset(
        val_index, labels_map, rare_labels, segment_len=seg_len, max_segments_per_record=max_seg
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.diffusion.batch_size,
        shuffle=True,
        num_workers=cfg.diffusion.num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.diffusion.batch_size,
        shuffle=False,
        num_workers=cfg.diffusion.num_workers,
        pin_memory=True,
    )

    # Model and diffusion schedule
    num_steps = cfg.diffusion.num_steps
    sched = make_linear_schedule(num_steps)
    model = SimpleUNet1D(
        cfg.diffusion.in_channels,
        cfg.diffusion.base_channels,
        len(labels_map),
    )

    device = _setup_device()
    model = model.to(device)
    sched = DiffusionSchedule(
        betas=sched.betas.to(device),
        alphas=sched.alphas.to(device),
        alphas_cumprod=sched.alphas_cumprod.to(device),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.diffusion.lr, weight_decay=cfg.diffusion.weight_decay)

    out_dir = Path(cfg.diffusion.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, cfg.diffusion.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for xb, yb in dl_train:
            xb = xb.to(device)  # [B, C, T]
            yb = yb.to(device)
            B = xb.size(0)

            t = torch.randint(0, num_steps, (B,), device=device, dtype=torch.long)
            x_t, noise = q_sample(xb, t, sched)
            pred_noise = model(x_t, yb)

            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.diffusion.grad_clip)
            opt.step()

            train_loss += float(loss.item()) * B
            n_train += B

        train_loss /= max(1, n_train)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                B = xb.size(0)

                t = torch.randint(0, num_steps, (B,), device=device, dtype=torch.long)
                x_t, noise = q_sample(xb, t, sched)
                pred_noise = model(x_t, yb)
                loss = F.mse_loss(pred_noise, noise)

                val_loss += float(loss.item()) * B
                n_val += B

        val_loss /= max(1, n_val)

        LOGGER.info(
            "Epoch %d/%d, train_loss=%.6f, val_loss=%.6f",
            epoch, cfg.diffusion.epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "labels_map": labels_map,
                "rare_labels": rare_labels,
                "num_steps": num_steps,
                "betas": sched.betas.detach().cpu().numpy().tolist(),
            }
            ckpt_path = out_dir / "diffusion_best.pt"
            torch.save(ckpt, ckpt_path)
            LOGGER.info("Saved best diffusion checkpoint to %s", ckpt_path)

    LOGGER.info("Done. Best val loss: %.6f", best_val_loss)

if __name__ == "__main__":
    main()