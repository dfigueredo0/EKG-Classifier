from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import wfdb

LOGGER = logging.getLogger("ekgclf.data.ptbxl")

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


def iter_mitbih(root: str | Path):
    """Two-lead rhythm dataset; kept for rhythm generalization."""
    root = Path(root)
    for dat in sorted(root.glob("*.dat")):
        base = dat.stem
        path = root / base
        try:
            sig, fs, leads = read_wfdb_record(path)
        except Exception as e:
            LOGGER.warning("Skipping %s due to %s", path, e)
            continue
        yield {
            "patient_id": base,
            "signals": sig,
            "fs": fs,
            "leads": leads,
            "labels": [],  # rhythm labels can be mapped later
            "record_path": str(path),
        }

