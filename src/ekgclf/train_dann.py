from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import hydra, torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from ekgclf.metrics import auroc_per_label
from ekgclf.models.head_multilabel import MultiLabelHead
from ekgclf.models.losses import BCEWithLogitsSmooth, FocalLossMultiLabel
from ekgclf.models.resnet1d import ResNet1D
from ekgclf.models.domain_adapt import DomainAdversarialNN, DomainAdaptationModel
from ekgclf.settings import ModelConfig, TrainConfig, setup_logging
from ekgclf.tracking import log_dict, start_run
from ekgclf.data.splitter import patient_level_split
from ekgclf.utils.data_utils import make_class_balanced_sampler

LOGGER = logging.getLogger("ekgclf.train_dann")

class EKGDomainDataset(Dataset):
    def __init__(self, index: List[Dict], labels_map: Dict[str, int]):
        self.index = index
        self.labels_map = labels_map
        self.num_labels = len(labels_map)
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self.index[i]
        npz = np.load(meta['npz'])
        x = npz['signals']
        x = torch.from_numpy(x).float().transpose(0, 1)
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        
        for lbl in meta['labels']:
            if lbl in self.labels_map:
                y[self.labels_map[lbl]] = 1.0
        
        source = meta.get('source', 'ptbxl')
        d = 0 if source == 'ptbxl' else 1
        d = torch.tensor(d, dtype=torch.long)
        
        return x, y, d
    
def _has_torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def setup_accelerator(cfg):
    use_amp_flag = bool(getattr(cfg, "trainer", None) and getattr(cfg.trainer, "amp", False))
    if _has_torch_cuda():
        device = torch.device("cuda")
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
        notes = "torch.cuda"
        return device, autocast_ctx, scaler, notes
    else:
        device = torch.device("cpu")
        scaler = None
        import contextlib
        autocast_ctx = contextlib.nullcontext()
        notes = "CPU"
        return device, autocast_ctx, scaler, notes

def to_device(x, device):
    import torch
    non_blocking = isinstance(device, torch.device) and device.type == "cuda"
    if hasattr(x, "to"):
        return x.to(device, non_blocking=non_blocking)
    return x

def build_feature_extractor(cfg_m: ModelConfig) -> ResNet1D:
    return ResNet1D(
        in_channels=cfg_m.model["in_channels"],
        base_channels=cfg_m.model["base_channels"],
        blocks=tuple(cfg_m.model["blocks"]),
        kernel_size=cfg_m.model["kernel_size"],
        stride=cfg_m.model["stride"],
        downsample=cfg_m.model["downsample"],
        dropout=cfg_m.model["dropout"],
    )

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
    weights = 1.0 / np.clip(pos_freq, 1e-3, None)
    return torch.tensor(weights, dtype=torch.float32)

def make_fast_loaders(ds_train, ds_val, train_bs, eval_bs, pin=True):
    common = dict(
        num_workers=8,
        pin_memory=pin,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
    )
    dl_train = DataLoader(ds_train, batch_size=train_bs, shuffle=True, **common)
    dl_val   = DataLoader(ds_val,   batch_size=eval_bs,  shuffle=False, **common)
    return dl_train, dl_val

@hydra.main(version_base=None, config_path="../../configs", config_name="train_dann")
def main(cfg: DictConfig):
    from ekgclf.settings import load_yaml, DataConfig, ModelConfig

    CFG_ROOT = Path(get_original_cwd())
    setup_logging(CFG_ROOT / "configs" / "logging.yaml")
    LOGGER.info("DANN Train config:\n%s", OmegaConf.to_yaml(cfg))

    data_cfg = DataConfig(**load_yaml(CFG_ROOT / "configs" / "data.yaml"))
    model_cfg = ModelConfig(**load_yaml(CFG_ROOT / "configs" / "model.yaml"))

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
    if len(index) == 0:
        raise RuntimeError("No samples found in processed indexes")

    labels = sorted({l for m in index for l in m["labels"]})
    labels_map = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)
    LOGGER.info("Num labels: %d", num_labels)

    # patient-level split
    def pid_key(m):
        return f'{m.get("source","unk")}:{m["patient_id"]}'

    pids = [pid_key(m) for m in index]
    from ekgclf.data.splitter import patient_level_split
    sp = patient_level_split(pids, train=0.8, val=0.1, test=0.1, seed=cfg.trainer.seed)

    ds_train = EKGDomainDataset([index[i] for i in sp.train], labels_map)
    ds_val   = EKGDomainDataset([index[i] for i in sp.val], labels_map)
    ds_test  = EKGDomainDataset([index[i] for i in sp.test], labels_map)
    dl_test  = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=2)
    
    device, autocast_ctx, scaler, backend_notes = setup_accelerator(cfg)
    LOGGER.info("Accelerator backend: %s", backend_notes)

    train_bs = cfg.trainer.batch_size
    eval_bs  = getattr(getattr(cfg, "eval", None), "batch_size", 256)
    pin = isinstance(device, torch.device) and device.type == "cuda"

    dl_train, dl_val = make_fast_loaders(ds_train, ds_val, train_bs, eval_bs, pin=pin)

    body = build_feature_extractor(model_cfg)
    cls_head = MultiLabelHead(in_features=256, num_classes=num_labels)
    dom_head = DomainAdversarialNN(in_features=256, hidden=128)

    model = DomainAdaptationModel(body, cls_head, dom_head).to(device)

    pos_weight = None
    if cfg.class_weights.get("use_class_weights", False):
        pos_weight = compute_class_weights([index[i] for i in sp.train], labels_map).to(device)

    criterion_cls = get_loss(cfg.loss, pos_weight)
    criterion_dom = nn.CrossEntropyLoss()

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
    history_path = report_dir / "history_dann.json"
    history = {"loss": [], "val_loss": [], "dom_loss": [], "val_dom_loss": []}

    run = None
    if cfg.mlflow.enable:
        run = start_run(cfg.mlflow.experiment, cfg.mlflow.tracking_uri, run_name="train_dann")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        log_dict("config_train_dann", cfg_dict)

    # lambda scheduling as in DANN paper
    def grl_lambda(epoch, total_epochs):
        p = float(epoch) / float(total_epochs)
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    for epoch in range(1, cfg.trainer.epochs + 1):
        model.train()
        total_loss = 0.0
        total_dom_loss = 0.0
        n_train = 0

        lam = grl_lambda(epoch, cfg.trainer.epochs)

        for xb, yb, db in dl_train:
            xb = to_device(xb, device)
            yb = to_device(yb, device)
            db = to_device(db, device)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits_y, logits_d = model(xb, lambda_grl=lam)
                loss_cls = criterion_cls(logits_y, yb)
                loss_dom = criterion_dom(logits_d, db)
                loss = loss_cls + cfg.dann.domain_loss_weight * loss_dom

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

            bs = xb.size(0)
            total_loss += float(loss_cls.item()) * bs
            total_dom_loss += float(loss_dom.item()) * bs
            n_train += bs

        train_loss_epoch = total_loss / max(1, n_train)
        train_dom_loss_epoch = total_dom_loss / max(1, n_train)

        model.eval()
        ys_val, ps_val = [], []
        val_loss_sum, val_dom_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb, db in dl_val:
                xb = to_device(xb, device)
                yb = to_device(yb, device)
                db = to_device(db, device)
                with autocast_ctx:
                    logits_y, logits_d = model(xb, lambda_grl=0.0)
                    val_loss = criterion_cls(logits_y, yb)
                    val_dom_loss = criterion_dom(logits_d, db)
                val_loss_sum += float(val_loss.item()) * xb.size(0)
                val_dom_sum += float(val_dom_loss.item()) * xb.size(0)
                ps_val.append(torch.sigmoid(logits_y).cpu().numpy())
                ys_val.append(yb.cpu().numpy())
                n_val += xb.size(0)

        y_true = np.concatenate(ys_val, axis=0)
        y_prob = np.concatenate(ps_val, axis=0)
        val_loss_epoch = val_loss_sum / max(1, n_val)
        val_dom_loss_epoch = val_dom_sum / max(1, n_val)

        macro = auroc_per_label(y_true, y_prob)
        macro_score = macro["macro"] if isinstance(macro, dict) else float(macro)

        LOGGER.info(
            "Epoch %d | lam=%.3f | train_loss=%.4f val_loss=%.4f dom_train=%.4f dom_val=%.4f macro_auroc=%.4f",
            epoch, lam, train_loss_epoch, val_loss_epoch,
            train_dom_loss_epoch, val_dom_loss_epoch, macro_score
        )

        if run:
            import mlflow
            mlflow.log_metric("train_dann/loss_cls", train_loss_epoch, step=epoch)
            mlflow.log_metric("train_dann/loss_dom", train_dom_loss_epoch, step=epoch)
            mlflow.log_metric("val_dann/loss_cls", val_loss_epoch, step=epoch)
            mlflow.log_metric("val_dann/loss_dom", val_dom_loss_epoch, step=epoch)
            mlflow.log_metric("val_dann/macro_auroc", macro_score, step=epoch)
            mlflow.log_metric("grl_lambda", lam, step=epoch)
            mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=epoch)

        if macro_score > best_score:
            best_score = macro_score
            ckpt_path = ckpt_dir / "best_dann.pt"
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "labels_map": labels_map},
                ckpt_path,
            )
            LOGGER.info("Saved DANN checkpoint to %s", ckpt_path)

        history["loss"].append(float(train_loss_epoch))
        history["val_loss"].append(float(val_loss_epoch))
        history["dom_loss"].append(float(train_dom_loss_epoch))
        history["val_dom_loss"].append(float(val_dom_loss_epoch))
        history_path.write_text(_json.dumps(history, indent=2))

        sched.step()
        
    plots_dir = report_dir / "dann_plots"
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