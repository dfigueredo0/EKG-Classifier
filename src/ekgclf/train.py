from __future__ import annotations

import json
import logging
import contextlib, time, importlib
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from ekgclf.metrics import auroc_per_label
from ekgclf.models.head_multilabel import MultiLabelHead
from ekgclf.models.losses import BCEWithLogitsSmooth, FocalLossMultiLabel
from ekgclf.models.resnet1d import ResNet1D
from ekgclf.settings import ModelConfig, TrainConfig, setup_logging
from ekgclf.tracking import log_dict, start_run

LOGGER = logging.getLogger("ekgclf.train")

class EKGDataset(Dataset):
    def __init__(self, index: list[dict], labels_map: dict[str, int]):
        self.index = index
        self.labels_map = labels_map
        self.num_labels = len(labels_map)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        meta = self.index[i]
        npz = np.load(meta["npz"])
        x = npz["signals"]  # [T, C]
        x = torch.from_numpy(x).float().transpose(0, 1)  # [C, T]
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        for lab in meta["labels"]:
            if lab in self.labels_map:
                y[self.labels_map[lab]] = 1.0
        return x, y

def _has_torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _has_directml() -> bool:
    return importlib.util.find_spec("torch_directml") is not None

def setup_accelerator(cfg):
    """
    Returns:
      device: torch.device or torch_directml device
      autocast_ctx: context manager for AMP autocast (no-op if backend missing)
      scaler: torch.amp.GradScaler or dummy (None) when AMP off
      notes: string describing backend
    """
    use_amp_flag = bool(getattr(cfg, "trainer", None) and getattr(cfg.trainer, "amp", False))

    if _has_torch_cuda():  # NVIDIA CUDA or AMD ROCm (HIP) — both surface as torch.cuda
        import torch
        device = torch.device("cuda")
        # Backend perf toggles (safe on ROCm; TF32 will be ignored on AMD)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        use_amp = bool(cfg.trainer.amp and torch.cuda.is_available())
        scaler = torch.amp.GradScaler(enabled=use_amp)
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
        notes = "torch.cuda (CUDA/ROCm)"
        
        return device, autocast_ctx, scaler, notes
    elif _has_directml():  # Windows fallback for AMD via DirectML
        import torch_directml
        device = torch_directml.device()
        scaler = None  # AMP not supported via torch-directml
        autocast_ctx =  contextlib.nullcontext()
        notes = "torch-directml (DirectML)"
        return device, autocast_ctx, scaler, notes
    else:
        import torch
        device = torch.device("cpu")
        scaler = None
        autocast_ctx = contextlib.nullcontext()
        notes = "CPU"
        return device, autocast_ctx, scaler, notes

def make_fast_loaders(ds_train, ds_val, train_bs, eval_bs, pin=True):
    """
    Builds high-throughput DataLoaders.
    """
    from torch.utils.data import DataLoader
    common = dict(
        num_workers=8,                 # tune 6–12 if you have spare cores
        pin_memory=pin,                # True iff GPU
        persistent_workers=True,       # keeps workers alive
        prefetch_factor=4,             # 2–6 depending on sample cost
        drop_last=False,
    )
    dl_train = DataLoader(ds_train, batch_size=train_bs, shuffle=True, **common)
    dl_val   = DataLoader(ds_val,   batch_size=eval_bs,  shuffle=False, **common)
    return dl_train, dl_val

def to_device(x, device):
    # Use non_blocking only when the destination is torch.cuda device
    import torch
    non_blocking = isinstance(device, torch.device) and device.type == "cuda"
    if hasattr(x, "to"):
        return x.to(device, non_blocking=non_blocking)
    return x

def build_model(cfg_m: ModelConfig, num_classes: int) -> Tuple[nn.Module, nn.Module]:
    body = ResNet1D(
        in_channels=cfg_m.model["in_channels"],
        base_channels=cfg_m.model["base_channels"],
        blocks=tuple(cfg_m.model["blocks"]),
        kernel_size=cfg_m.model["kernel_size"],
        stride=cfg_m.model["stride"],
        downsample=cfg_m.model["downsample"],
        dropout=cfg_m.model["dropout"],
    )
    head = MultiLabelHead(in_features=256, num_classes=num_classes)
    bias_init = cfg_m.head.get("bias_init", None)
    if bias_init is not None and head.fc.bias is not None:
        with torch.no_grad():
            head.fc.bias.fill_(float(bias_init))
    return body, head

def get_loss(loss_cfg, pos_weight: torch.Tensor | None):
    if loss_cfg.get("focal_gamma"):
        return FocalLossMultiLabel(float(loss_cfg.get("focal_gamma", None)), pos_weight)
    smoothing = loss_cfg.get("label_smoothing", loss_cfg.get("laplace_smoothing", 0.0))
    return BCEWithLogitsSmooth(smoothing, pos_weight)

def compute_class_weights(index: list[dict], labels_map: dict[str, int]) -> torch.Tensor:
    counts = np.zeros(len(labels_map), dtype=np.int64)
    for meta in index:
        for lab in meta["labels"]:
            if lab in labels_map:
                counts[labels_map[lab]] += 1
    total = max(1, len(index))
    pos_freq = counts / total
    # inverse freq with floor
    weights = 1.0 / np.clip(pos_freq, 1e-3, None)
    return torch.tensor(weights, dtype=torch.float32)

@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    CFG_ROOT = Path(get_original_cwd())
    
    # Logging
    setup_logging(CFG_ROOT / "configs" / "logging.yaml")
    LOGGER.info("Train config: %s", OmegaConf.to_yaml(cfg))

    # Load other configs
    from ekgclf.settings import load_yaml, DataConfig, ModelConfig
    data_cfg = DataConfig(**load_yaml(CFG_ROOT / "configs" / "data.yaml"))
    model_cfg = ModelConfig(**load_yaml(CFG_ROOT / "configs" / "model.yaml"))

    # Load processed index
    proc_dir = Path(data_cfg.paths.processed)
    index_path = proc_dir / "ptbxl_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing processed index at {index_path}")
    import json as _json

    index = _json.load(open(index_path, "r", encoding="utf-8"))

    # Build label map from corpus
    labels = sorted({l for m in index for l in m["labels"]})
    labels_map = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)
    LOGGER.info("Num labels: %d", num_labels)

    # Dataset split by patient ID
    from ekgclf.data.splitter import patient_level_split
    pids = [m["patient_id"] for m in index]
    sp = patient_level_split(pids, train=0.8, val=0.1, test=0.1, seed=cfg.trainer.seed)
    ds_train = EKGDataset([index[i] for i in sp.train], labels_map)
    ds_val = EKGDataset([index[i] for i in sp.val], labels_map)
    
    device, autocast_ctx, scaler, backend_notes = setup_accelerator(cfg)
    LOGGER.info("Accelerator backend: %s", backend_notes)

    train_bs = cfg.trainer.batch_size
    eval_bs  = getattr(getattr(cfg, "eval", None), "batch_size", 256)
    pin = isinstance(device, torch.device) and device.type == "cuda"
    dl_train, dl_val = make_fast_loaders(ds_train, ds_val, train_bs, eval_bs, pin=pin)

    body, head = build_model(model_cfg, num_labels)
    model = torch.nn.Sequential(body, head).to(device)

    pos_weight = None
    if cfg.class_weights.get("use_class_weights", False):
        pos_weight = compute_class_weights([index[i] for i in sp.train], labels_map).to(device)

    criterion = get_loss(cfg.loss, pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.wd)
    if cfg.trainer.lr_scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.trainer.epochs)
    else:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    best_score = -1.0
    ckpt_dir = Path(cfg.trainer.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    history_path = report_dir / "history.json"
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    # MLflow
    run = None
    if cfg.mlflow.enable:
        run = start_run(cfg.mlflow.experiment, cfg.mlflow.tracking_uri, run_name="train")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        log_dict("config_train", cfg_dict)

    for epoch in range(1, cfg.trainer.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = to_device(xb, device), to_device(yb, device)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(xb)
                loss = criterion(logits, yb)
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.trainer.grad_clip_norm:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.trainer.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip_norm)
                opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss_epoch = total_loss / max(1, len(ds_train))

        model.eval()
        ys_val, ps_val = [], []
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = to_device(xb, device), to_device(yb, device)
                with autocast_ctx:  # safe; no grads
                    logits = model(xb)
                    val_loss = criterion(logits, yb)
                val_loss_sum += float(val_loss.item()) * xb.size(0)
                ps_val.append(torch.sigmoid(logits).detach().cpu().numpy())
                ys_val.append(yb.detach().cpu().numpy())
                n_val += xb.size(0)

        y_true = np.concatenate(ys_val, axis=0)
        y_prob = np.concatenate(ps_val, axis=0)
        val_loss_epoch = val_loss_sum / max(1, n_val)

        y_pred = (y_prob >= 0.5).astype(int)
        val_acc = f1_score(y_true.ravel(), y_pred.ravel())  # micro-F1

        ys_tr, ps_tr = [], []
        with torch.no_grad():
            for xb, yb in dl_train:
                xb, yb = to_device(xb, device), to_device(yb, device)
                logits = model(xb)
                ps_tr.append(torch.sigmoid(logits).cpu().numpy())
                ys_tr.append(yb.cpu().numpy())
        y_true_tr = np.concatenate(ys_tr, axis=0)
        y_prob_tr = np.concatenate(ps_tr, axis=0)
        y_pred_tr = (y_prob_tr >= 0.5).astype(int)
        tr_acc = f1_score(y_true_tr.ravel(), y_pred_tr.ravel())

        macro = auroc_per_label(y_true, y_prob)
        LOGGER.info(
            "Epoch %d train_loss=%.4f val_loss=%.4f val_macro_auroc=%s val_microF1(acc)=%.4f",
            epoch, train_loss_epoch, val_loss_epoch,
            f"{macro:.4f}" if np.isfinite(macro) else "nan",
            val_acc
        )

        if run:
            import mlflow
            mlflow.log_metric("train/loss", train_loss_epoch, step=epoch)
            mlflow.log_metric("train/micro_f1", tr_acc, step=epoch)
            mlflow.log_metric("val/loss", val_loss_epoch, step=epoch)
            mlflow.log_metric("val/micro_f1", val_acc, step=epoch)
            mlflow.log_metric("val/macro_auroc", macro, step=epoch)
            mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=epoch)

        if macro > best_score:
            best_score = macro
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "labels_map": labels_map}, ckpt_path)
            LOGGER.info("Saved checkpoint to %s", ckpt_path)

        history["loss"].append(float(train_loss_epoch))
        history["val_loss"].append(float(val_loss_epoch))
        history["acc"].append(float(tr_acc))
        history["val_acc"].append(float(val_acc))
        history_path.write_text(json.dumps(history, indent=2))

        sched.step()

    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure(figsize=(7,4))
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / "loss.png", dpi=160); plt.close()

    # “Accuracy” proxy (micro-F1)
    plt.figure(figsize=(7,4))
    plt.plot(history["acc"], label="train micro-F1")
    plt.plot(history["val_acc"], label="val micro-F1")
    plt.title("Micro-F1")
    plt.xlabel("Epoch"); plt.ylabel("F1")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / "micro_f1.png", dpi=160); plt.close()

    if run:
        import mlflow
        mlflow.end_run()

if __name__ == "__main__":
    main()
