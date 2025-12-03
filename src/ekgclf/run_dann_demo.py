from __future__ import annotations

import argparse
import json as _json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from ekgclf.train_dann import EKGDomainDataset, build_feature_extractor
from ekgclf.models.head_multilabel import MultiLabelHead
from ekgclf.models.domain_adapt import DomainAdversarialNN, DomainAdaptationModel
from ekgclf.settings import load_yaml, DataConfig, ModelConfig


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def build_model_and_labels(
    ckpt_path: Path,
    model_cfg_path: Path,
) -> Tuple[DomainAdaptationModel, Dict[str, int]]:
    """Build the DANN model and load weights + labels_map from checkpoint."""
    # Load model config same as in train_dann.py
    model_cfg = ModelConfig(**load_yaml(model_cfg_path))

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels_map: Dict[str, int] = ckpt["labels_map"]
    num_labels = len(labels_map)

    # Build model body + heads (same layout as training)
    body = build_feature_extractor(model_cfg)
    cls_head = MultiLabelHead(in_features=256, num_classes=num_labels)
    dom_head = DomainAdversarialNN(in_features=256, hidden=128)
    model = DomainAdaptationModel(body, cls_head, dom_head)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, labels_map


def load_full_index(data_cfg_path: Path) -> Tuple[list, Dict[str, int]]:
    """Load combined PTB-XL + MIT-BIH index and build labels_map (like train_dann)."""
    data_cfg = DataConfig(**load_yaml(data_cfg_path))
    proc_dir = Path(data_cfg.paths.processed)

    ptb_idx_path = proc_dir / "ptbxl_index.json"
    mit_idx_path = proc_dir / "mitbih_index.json"

    if not ptb_idx_path.exists():
        raise FileNotFoundError(f"Missing processed index at {ptb_idx_path}")

    ptb_index = _json.load(open(ptb_idx_path, "r", encoding="utf-8"))
    for m in ptb_index:
        m.setdefault("source", "ptbxl")

    mit_index = []
    if mit_idx_path.exists():
        mit_index = _json.load(open(mit_idx_path, "r", encoding="utf-8"))
        for m in mit_index:
            m.setdefault("source", "mitbih")

    index = ptb_index + mit_index
    if not index:
        raise RuntimeError("No samples found in processed indexes")

    labels = sorted({l for m in index for l in m["labels"]})
    labels_map = {l: i for i, l in enumerate(labels)}
    return index, labels_map


def load_external_signal(path: Path) -> torch.Tensor:
    """
    Load an external .npz or .npy file and return a tensor shaped [1, C, T].

    Expected:
      - .npz with key 'signals' shaped [T, C], OR
      - .npy shaped [T, C].
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        npz = np.load(path)
        if "signals" not in npz:
            raise KeyError(f"{path}: expected key 'signals' in npz file")
        sig = npz["signals"]
    elif path.suffix == ".npy":
        sig = np.load(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (use .npz or .npy)")

    if sig.ndim != 2:
        raise ValueError(f"Expected 2D array [T, C], got shape {sig.shape}")

    # EKGDomainDataset does: torch.from_numpy(sig).float().transpose(0, 1)
    # i.e. [T, C] -> [C, T]
    x = torch.from_numpy(sig).float().transpose(0, 1)  # [C, T]
    return x.unsqueeze(0)  # [1, C, T]


# -------------------------------------------------------------------
# Demo modes
# -------------------------------------------------------------------

@torch.no_grad()
def run_demo_dataset(
    ckpt_path: Path,
    project_root: Path,
    idx: int,
) -> None:
    """Run demo on a sample from the processed dataset (PTB-XL + MIT-BIH)."""

    data_cfg_path = project_root / "configs" / "data.yaml"
    model_cfg_path = project_root / "configs" / "model.yaml"

    # Build model + labels from checkpoint
    model, ckpt_labels_map = build_model_and_labels(ckpt_path, model_cfg_path)
    inv_labels = {v: k for k, v in ckpt_labels_map.items()}

    # Load index & build dataset
    index, labels_map_from_data = load_full_index(data_cfg_path)

    # Optional sanity check: warn if labels differ
    if labels_map_from_data != ckpt_labels_map:
        print("[!] Warning: labels_map in data differs from checkpoint; using checkpoint map.")

    ds = EKGDomainDataset(index, ckpt_labels_map)

    if idx < 0 or idx >= len(ds):
        raise IndexError(f"Requested idx={idx}, but dataset has length {len(ds)}")

    x, y, d = ds[idx]  # x: [C, T], y: [num_labels], d: scalar domain
    x = x.unsqueeze(0)  # [1, C, T]

    logits_y, logits_d = model(x, lambda_grl=0.0)
    probs = torch.sigmoid(logits_y)[0]
    dom_probs = torch.softmax(logits_d, dim=-1)[0]

    print(f"\n=== DANN demo (dataset) idx={idx} ===")
    print(f"Domain label: {int(d.item())} -> {'PTB-XL' if int(d.item()) == 0 else 'MIT-BIH'}")
    print(f"Domain probabilities [ptbxl, mitbih]: {dom_probs.tolist()}")

    true_labels = [inv_labels[i] for i, v in enumerate(y.tolist()) if v > 0.5]
    print("True labels:", true_labels)

    topk_vals, topk_idx = torch.topk(probs, k=min(5, probs.numel()))
    print("\nPredicted top-5 labels:")
    for score, j in zip(topk_vals.tolist(), topk_idx.tolist()):
        print(f"  {inv_labels[j]} : {score:.4f}")
    print()


@torch.no_grad()
def run_demo_file(
    ckpt_path: Path,
    project_root: Path,
    file_path: Path,
) -> None:
    """Run demo on an external .npz/.npy file with EKG signals."""
    model_cfg_path = project_root / "configs" / "model.yaml"

    model, labels_map = build_model_and_labels(ckpt_path, model_cfg_path)
    inv_labels = {v: k for k, v in labels_map.items()}

    x = load_external_signal(file_path)  # [1, C, T]
    logits_y, logits_d = model(x, lambda_grl=0.0)

    probs = torch.sigmoid(logits_y)[0]
    dom_probs = torch.softmax(logits_d, dim=-1)[0]
    dom_pred = int(torch.argmax(dom_probs).item())

    print(f"\n=== DANN demo (external file) ===")
    print(f"File: {file_path}")
    print(f"Predicted domain: {dom_pred} -> {'PTB-XL' if dom_pred == 0 else 'MIT-BIH'}")
    print(f"Domain probabilities [ptbxl, mitbih]: {dom_probs.tolist()}")

    topk_vals, topk_idx = torch.topk(probs, k=min(5, probs.numel()))
    print("\nPredicted top-5 labels:")
    for score, j in zip(topk_vals.tolist(), topk_idx.tolist()):
        print(f"  {inv_labels[j]} : {score:.4f}")
    print("True labels: (not available for external file)\n")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/dann/best_dann.pt",
        help="Path to trained DANN checkpoint",
    )
    ap.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Path to repo root (where configs/ lives)",
    )

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--idx",
        type=int,
        help="Dataset index (combined PTB-XL + MIT-BIH)",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to external .npz/.npy file containing 'signals' [T, C]",
    )

    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    project_root = Path(args.project_root).resolve()

    if args.idx is not None:
        run_demo_dataset(ckpt_path, project_root, args.idx)
    else:
        run_demo_file(ckpt_path, project_root, Path(args.file).resolve())


if __name__ == "__main__":
    main()
