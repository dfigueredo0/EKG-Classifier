# ekgclf/data/mitbih.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import wfdb

PTBXL_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LEAD_TO_IDX = {name: i for i, name in enumerate(PTBXL_ORDER)}

# Map MIT-BIH channel names to PTB-XL indices (arrhythmia DB typically has MLII and V1)
MITBIH_TO_12 = {
    "MLII": LEAD_TO_IDX["II"],
    "V1":   LEAD_TO_IDX["V1"],
}

@dataclass
class MitbihWindow:
    record_path: str      # wfdb prefix without extension
    patient_id: int       # if unknown, use record number
    start_s: float
    end_s: float
    label: str            # rhythm label (e.g., N, AFIB, AFL, …)

def _read_mitbih_record(prefix: Path) -> tuple[np.ndarray, int, list[str]]:
    rec = wfdb.rdrecord(str(prefix))
    fs = int(rec.fs)
    sig = rec.p_signal.T  # [C_src, T]
    lead_names = [str(n).split()[0] for n in rec.sig_name]
    return sig.astype(np.float32), fs, lead_names

def _lift_to_12(sig_src: np.ndarray, lead_names: Sequence[str], tgt_len: int) -> np.ndarray:
    """Place available signals (MLII/V1) into a 12-lead frame; zero-fill others."""
    x12 = np.zeros((12, tgt_len), dtype=np.float32)
    # ensure sig_src is [C_src, T]
    for i, nm in enumerate(lead_names):
        if nm in MITBIH_TO_12:
            j = MITBIH_TO_12[nm]
            x = sig_src[i]
            if x.shape[0] < tgt_len:
                pad = tgt_len - x.shape[0]
                x = np.pad(x, (0, pad))
            else:
                x = x[:tgt_len]
            x12[j] = x
    return x12  # [12, T]

def _load_index_from_csv(csv_path: Path) -> list[MitbihWindow]:
    """
    Expected CSV header:
      record,start_s,end_s,label,patient_id   (patient_id optional)
    record is the WFDB prefix relative to root, e.g., "100" or "records/100"
    """
    out: list[MitbihWindow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rec = row["record"].strip()
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            label = row["label"].strip()
            pid = int(row.get("patient_id", "".strip() or "-1")) if row.get("patient_id") else -1
            out.append(MitbihWindow(rec, pid, start_s, end_s, label))
    return out

def iter_mitbih(root: str | Path, split_csv: str | Path, target_fs: int) -> Iterator[dict]:
    """
    Yield per-window dicts, matching the shape your PTB-XL iterator returns:
      - patient_id: int
      - signals: np.ndarray [T, C] float32 (time-major to match the rest of your pipeline)
      - fs: int (target_fs)
      - leads: list[str] (PTB-XL order)
      - labels: list[str] (single rhythm string)
      - record_path: str
    NOTE: resampling, filtering, and z-scoring are applied downstream (make_dataset.py).
    """
    root = Path(root)
    items = _load_index_from_csv(Path(split_csv))
    for w in items:
        prefix = root / w.record_path
        sig_src, fs_src, lead_names = _read_mitbih_record(prefix)  # [C_src, T_src]
        # crop at source fs
        s0, s1 = int(round(w.start_s * fs_src)), int(round(w.end_s * fs_src))
        sig_src = sig_src[:, s0:s1]  # [C_src, T_crop]

        # convert to time-major for downstream transforms (T, C)
        # we'll lift to 12 leads AFTER resample in make_dataset
        # here just stash the minimal info
        yield {
            "patient_id": (w.patient_id if w.patient_id >= 0 else int(Path(w.record_path).name)),
            "signals": sig_src.T.astype(np.float32),  # [T_crop, C_src]
            "fs": fs_src,
            "leads": lead_names,
            "label_rhythm": w.label,  # keep exact string; mapped later if needed
            "record_path": str(prefix),
        }
