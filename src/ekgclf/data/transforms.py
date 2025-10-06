import logging
from typing import Tuple

import numpy as np
from scipy import signal

LOGGER = logging.getLogger("ekgclf.data.transforms")


def resample(sig: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return sig
    t, c = sig.shape
    tgt_len = int(round(t * (target_fs / orig_fs)))
    out = signal.resample(sig, tgt_len, axis=0)
    return out.astype(np.float32)


def bandpass(sig: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, sig, axis=0).astype(np.float32)


def zscore(sig: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0, keepdims=True)
    norm = (sig - mean) / (std + eps)
    return norm.astype(np.float32), mean.squeeze(0), std.squeeze(0)


def _canon_lead(name: str) -> str:
    """Canonicalize lead names to match ['I','II','III','aVR','aVL','aVF','V1'..'V6']."""
    # strip non-alnum and upper-case
    s = "".join(ch for ch in name if ch.isalnum()).upper()
    # Map common synonyms
    if s == "AVR": return "aVR"
    if s == "AVL": return "aVL"
    if s == "AVF": return "aVF"
    if s in {"I","II","III"}: return s
    if s.startswith("V") and s[1:].isdigit():
        return "V" + str(int(s[1:]))  # normalize V01 -> V1
    # Fall back to original for visibility
    return name

def _derive_avr(sig: np.ndarray, leads: list[str]) -> np.ndarray:
    """Compute aVR ≈ -(I + II)/2 when available. sig: [T,C]."""
    idx = {l: i for i, l in enumerate(leads)}
    if "I" in idx and "II" in idx:
        avr = - (sig[:, idx["I"]] + sig[:, idx["II"]]) / 2.0
        return avr.astype(np.float32)
    raise ValueError("Cannot derive aVR without leads I and II")

def ensure_leads(
    sig: np.ndarray,
    leads: list[str],
    required: list[str],
    allow_derive_avr: bool = True,
) -> np.ndarray:
    """
    Reorder/augment leads to match required order.
    - Normalizes present and required lead names.
    - Optionally derives aVR from I and II if missing.
    Fails if any other required lead is missing.
    """
    # Canonicalize
    canon_present = [_canon_lead(l) for l in leads]
    canon_required = [_canon_lead(l) for l in required]
    present_idx = {l: i for i, l in enumerate(canon_present)}

    # Derive aVR if requested and missing
    if "aVR" in canon_required and "aVR" not in present_idx and allow_derive_avr:
        try:
            avr = _derive_avr(sig, {l: i for i, l in zip(canon_present, canon_present)})
        except Exception:
            # try with original names too
            orig_map = {l: i for i, l in enumerate(leads)}
            if "I" in orig_map and "II" in orig_map:
                avr = - (sig[:, orig_map["I"]] + sig[:, orig_map["II"]]) / 2.0
            else:
                avr = None
        if avr is not None:
            # Append derived aVR
            sig = np.concatenate([sig, avr[:, None]], axis=1)
            canon_present.append("aVR")
            present_idx = {l: i for i, l in enumerate(canon_present)}

    # Build index list; error if missing
    idxs = []
    missing = []
    for r in canon_required:
        i = present_idx.get(r)
        if i is None:
            missing.append(r)
        else:
            idxs.append(i)

    if missing:
        LOGGER.debug("Available leads (canonicalized): %s", canon_present)
        raise ValueError(f"Missing required lead(s): {missing}")

    return sig[:, idxs]