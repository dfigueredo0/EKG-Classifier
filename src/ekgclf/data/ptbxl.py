from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import wfdb

LOGGER = logging.getLogger("ekgclf.data.ptbxl")

PTBXL_REQUIRED_COLS = ["patient_id", "scp_codes"] 

def _load_ptbxl_metadata(root: Path) -> pd.DataFrame:
    """
    Load PTB-XL metadata and statements, safely parse scp_codes, and attach:
      - labels: list of fine-grained SCP codes per record
      - diagnostic_superclass: optional list of superclasses (e.g., NORM, MI, STTC, HYP, CD)
    """
    meta_csv = root / "ptbxl_database.csv"
    scp_csv = root / "scp_statements.csv"
    if not meta_csv.exists() or not scp_csv.exists():
        raise FileNotFoundError(
            f"PTB-XL metadata not found under {root}. See README for download instructions."
        )

    df_meta = pd.read_csv(meta_csv)  # includes patient_id, ecg_id, filename_hr/lr, scp_codes
    df_scp = pd.read_csv(scp_csv, index_col=0)

    # Basic column validation
    for col in PTBXL_REQUIRED_COLS:
        if col not in df_meta.columns:
            raise ValueError(f"PTB-XL missing required column: {col}")
    if ("filename_hr" not in df_meta.columns) and ("filename_lr" not in df_meta.columns):
        raise ValueError("PTB-XL metadata must include filename_hr or filename_lr.")

    # Safe parsing of scp_codes (avoid eval)
    def _parse_codes(s):
        if isinstance(s, str):
            return ast.literal_eval(s)
        return s

    df_meta = df_meta.copy()
    df_meta["scp_codes"] = df_meta["scp_codes"].apply(_parse_codes)
    df_meta["labels"] = df_meta["scp_codes"].apply(lambda d: list(d.keys()) if isinstance(d, dict) else [])

    # Optional diagnostic superclass projection
    if {"diagnostic", "diagnostic_class"}.issubset(df_scp.columns):
        diag_map = df_scp[df_scp["diagnostic"] == 1]["diagnostic_class"].to_dict()

        def _to_superclasses(d: dict) -> list[str]:
            if not isinstance(d, dict):
                return []
            return sorted({diag_map.get(code) for code in d.keys() if diag_map.get(code) is not None})

        df_meta["diagnostic_superclass"] = df_meta["scp_codes"].apply(_to_superclasses)
    else:
        df_meta["diagnostic_superclass"] = [[] for _ in range(len(df_meta))]

    return df_meta


def read_wfdb_record(path_prefix: Path) -> Tuple[np.ndarray, int, List[str]]:
    """
    Read a WFDB record via its prefix (no extension).
    Returns: (signals [T, C], fs, lead_names)
    """
    rec = wfdb.rdrecord(str(path_prefix))
    fs = int(rec.fs)
    sig = rec.p_signal  # [T, leads]
    leads = [str(n) for n in rec.sig_name]
    return sig.astype(np.float32), fs, leads


def iter_ptbxl(
    root: str | Path,
    use_scored_subset: bool = True,
    prefer_hr: bool = True,
) -> Iterator[dict]:
    """
    Record-by-record iterator over PTB-XL.

    Yields dicts with keys:
      - ecg_id (int)
      - patient_id (int)
      - signals: np.ndarray [T, C] float32 (raw from WFDB; downstream transforms handle resample/filter/zscore)
      - fs: int
      - leads: list[str]
      - labels: list[str] fine-grained SCP codes
      - labels_super: list[str] diagnostic superclasses (if available)
      - record_path: str path prefix used for WFDB
    """
    root = Path(root)
    meta = _load_ptbxl_metadata(root)

    # Optional scored subset (consistent with PhysioNet's strat_fold presence)
    if use_scored_subset and "strat_fold" in meta.columns:
        meta = meta[meta["strat_fold"].notna()]

    fn_col = None
    if prefer_hr and "filename_hr" in meta.columns:
        fn_col = "filename_hr"
    elif "filename_lr" in meta.columns:
        fn_col = "filename_lr"
    else:
        raise ValueError("No filename_hr or filename_lr column available in metadata.")

    for _, row in meta.iterrows():
        rec_path = root / row[fn_col]
        try:
            sig, fs, leads = read_wfdb_record(rec_path)
        except Exception as e:
            LOGGER.warning("Skipping %s due to %s", rec_path, e)
            continue

        yield {
            "ecg_id": int(row["ecg_id"]) if "ecg_id" in row else -1,
            "patient_id": int(row["patient_id"]),
            "signals": sig,  # [T, C]
            "fs": fs,
            "leads": leads,
            "labels": row["labels"],
            "labels_super": row.get("diagnostic_superclass", []),
            "record_path": str(rec_path),
        }