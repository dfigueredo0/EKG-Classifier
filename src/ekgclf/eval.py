from __future__ import annotations

import os
import json
import logging
from pathlib import Path

import hydra
import matplotlib as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from hydra.utils import get_original_cwd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


from ekgclf.calibration import TemperatureScaler
from ekgclf.metrics import auroc_per_label, expected_calibration_error, f1_per_label
from ekgclf.settings import setup_logging

LOGGER = logging.getLogger("ekgclf.eval")

class EvalDataset(Dataset):
    def __init__(self, index, labels_map):
        self.index = index
        self.labels_map = labels_map
        self.num_labels = len(labels_map)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        meta = self.index[i]
        x = np.load(meta["npz"])["signals"]
        x = torch.from_numpy(x).float().transpose(0, 1)
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        for lab in meta["labels"]:
            if lab in self.labels_map:
                y[self.labels_map[lab]] = 1.0
        return x, y

@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig):
    CFG_ROOT = Path(get_original_cwd())

    REPORTS_DIR = Path("reports")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    from ekgclf.settings import DataConfig, ModelConfig, load_yaml
    data_cfg = DataConfig(**load_yaml(CFG_ROOT / "configs" / "data.yaml"))
    model_cfg = ModelConfig(**load_yaml(CFG_ROOT / "configs" / "model.yaml"))

    CKPT_PATH = Path(cfg.trainer.checkpoint_dir) / "best.pt"
    IDX_PATH = Path(data_cfg.paths.processed) / "ptbxl_index.json"
    
    setup_logging(CFG_ROOT / "configs" / "logging.yaml")
    LOGGER.info("Eval config: %s", OmegaConf.to_yaml(cfg))

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    labels_map = ckpt["labels_map"]
    num_labels = len(labels_map)

    # Load index and build test set
    import json as _json

    index = _json.load(open(IDX_PATH, "r", encoding="utf-8"))
    # naive split like in train.py
    from ekgclf.data.splitter import patient_level_split

    pin = bool(torch.cuda.is_available())
    pids = [m["patient_id"] for m in index]
    sp = patient_level_split(pids, 0.8, 0.1, 0.1, seed=cfg.trainer.seed)
    ds = EvalDataset([index[i] for i in sp.test], labels_map)
    dl = DataLoader(ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    # Rebuild model and load weights
    from ekgclf.models.resnet1d import ResNet1D
    from ekgclf.models.head_multilabel import MultiLabelHead

    body = ResNet1D(in_channels=model_cfg.model["in_channels"], base_channels=model_cfg.model["base_channels"], blocks=tuple(model_cfg.model["blocks"]), kernel_size=model_cfg.model["kernel_size"], stride=model_cfg.model["stride"], 
                    downsample=model_cfg.model["downsample"], dropout=model_cfg.model["dropout"],)
    head = MultiLabelHead(256, num_labels)
    model = torch.nn.Sequential(body, head)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    # Debugging print(f"Missing keys: {missing}, \nUnexpected keys: {unexpected}")
    model.eval()

    ys, logits_acc = [], []
    with torch.no_grad():
        for xb, yb in dl:
            out = model(xb)
            ys.append(yb)
            logits_acc.append(out)
    y_true_t = torch.cat(ys, dim=0)
    logits_t = torch.cat(logits_acc, axis=0)

    # Calibration (temperature)
    if cfg.eval.calibration.method == "temperature":
        scaler = TemperatureScaler()
        scaler.fit(logits_t, y_true_t)
        scaler.eval()
        with torch.no_grad():
            logits_t = scaler(logits_t)
    
    probs = torch.sigmoid(logits_t).numpy()
    y_true = y_true_t.numpy()

    # Thresholding
    if cfg.eval.thresholds.mode == "fixed":
        vals = getattr(cfg.eval.thresholds, "fixed_values", [0.5])
        thr = float(vals[0])
        y_pred = (probs >= thr).astype(int)
    elif cfg.eval.thresholds.mode in ("per_label_opt_f1", "optimal"):
        y_pred = np.zeros_like(probs, dtype=int)
        for i in range(probs.shape[1]):
            best_f, best_t = -1, 0.5
            for t in np.linspace(0.05, 0.95, 19):
                p = (probs[:, i] >= t).astype(int)
                
                f = f1_score(y_true[:, i], p, zero_division=0)
                if f > best_f:
                    best_f, best_t = f, t
            y_pred[:, i] = (probs[:, i] >= best_t).astype(int)
    else:
        y_pred = (probs >= 0.5).astype(int)

    y_true_argmax = y_true.argmax(axis=1)
    y_pred_argmax = probs.argmax(axis=1)
    
    cm_raw = confusion_matrix(y_true_argmax, y_pred_argmax, labels=list(range(len(class_names))))
    # Row-normalized (each row sums to 1; handle divide-by-zero safely)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_raw, np.maximum(row_sums, 1), where=row_sums>0)

    with open(REPORTS_DIR / "confusion_matrix_raw.json", "w", encoding="utf-8") as f:
        json.dump({"labels": class_names, "matrix": cm_raw.tolist()}, f, indent=2)
    with open(REPORTS_DIR / "confusion_matrix_norm.json", "w", encoding="utf-8") as f:
        json.dump({"labels": class_names, "matrix": cm_norm.tolist()}, f, indent=2)
    
    # does not reflect multilabel co-occurrence; it’s a single-label proxy; for full multilabel views later, add per-class binary confusion matrices (one-vs-rest) and stack them in a grid.
    plots_dir = REPORTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm_norm, interpolation="nearest")  # normalized for readability
    plt.title("Confusion Matrix (argmax, row-normalized)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=160)
    plt.close()
        
    np.save(REPORTS_DIR / "y_true.npy", y_true)
    np.save(REPORTS_DIR / "y_prob.npy", probs)
    np.save(REPORTS_DIR / "y_pred.npy", y_pred)
    
    class_names = getattr(ds, "CLASS_NAMES", [str(i) for i in range(y_true.shape[1])])
    with open(REPORTS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    

    ece = expected_calibration_error(y_true, probs, n_bins=cfg.eval.calibration.ece_bins)

    out = {
        "macro_auroc": auroc_per_label(y_true, probs),
        "micro_f1": float(f1_score(y_true.ravel(), y_pred.ravel())),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ece": ece,
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    LOGGER.info("Eval: %s", out)

if __name__ == "__main__":
    main()
