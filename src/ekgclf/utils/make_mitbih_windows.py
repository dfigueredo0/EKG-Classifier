#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, logging
from pathlib import Path
from typing import List, Tuple

import wfdb
import numpy as np

LOG = logging.getLogger("make_mitbih_windows")

# Rhythm labels in MIT-BIH come in aux_note entries like "(N", "(AFIB", "(AFL", "(B", "(T", "(SBR", etc.)
# We'll map them to a compact set. Extend as you see fit.
RHYTHM_MAP = {
    "N": "N", "AFIB": "AFIB", "AFL": "AFL", "SVTA": "SVTA", "VT": "VT", "B": "BBB",
    "SBR": "SBR", "T": "T", "IVR": "IVR", "VFL": "VFL",
}
# If an aux_note starts with "(" we strip it and stop at first non-alnum.
def _canon_rhythm(aux: str) -> str | None:
    if not aux: 
        return None
    s = aux.strip()
    if s.startswith("("): s = s[1:]
    s = "".join(ch for ch in s if ch.isalnum())
    return RHYTHM_MAP.get(s, None)

def _record_ids(root: Path) -> List[str]:
    # Any .hea present is considered a record
    return sorted([p.stem for p in root.glob("*.hea")])

def _rhythm_intervals(rec_id: str, fs: float, root: Path) -> List[Tuple[int,int,str]]:
    """Return [(start_sample, end_sample, rhythm)] across the record."""
    # Read beat/rhythm annotations (extension 'atr')
    try:
        ann = wfdb.rdann(str(root / rec_id), "atr")
    except Exception as e:
        LOG.warning("No annotations for %s: %s", rec_id, e)
        return []
    # Build change points based on aux_note rhythm change markers
    changes = []
    for samp, aux in zip(ann.sample, ann.aux_note):
        r = _canon_rhythm(aux)
        if r: changes.append((int(samp), r))
    if not changes:
        return []
    # Close intervals with record length
    sig = wfdb.rdrecord(str(root / rec_id))
    total = int(sig.sig_len)
    intervals = []
    for i, (s, r) in enumerate(changes):
        e = changes[i+1][0] if i+1 < len(changes) else total
        if e > s:
            intervals.append((s, e, r))
    return intervals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="Folder with MIT-BIH records (.dat/.hea)")
    ap.add_argument("--out", type=Path, required=True, help="CSV output path")
    ap.add_argument("--win", type=float, default=10.0, help="Window (sec)")
    ap.add_argument("--stride", type=float, default=5.0, help="Stride (sec)")
    ap.add_argument("--default_rhythm", type=str, default="N", help="Fallback rhythm if none present")
    ap.add_argument("--log", type=str, default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    root: Path = args.raw
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    ids = _record_ids(root)
    if not ids:
        raise FileNotFoundError(f"No .hea files found under {root}. Did you point to the correct folder?")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record","start_s","end_s","label","patient_id"])  # patient_id = record id for MIT-BIH
        for rec in ids:
            try:
                r = wfdb.rdrecord(str(root / rec))
            except Exception as e:
                LOG.warning("Skip %s: cannot read record (%s)", rec, e)
                continue
            fs = float(r.fs)
            dur_s = r.sig_len / fs
            win_n = int(round(args.win * fs))
            stride_n = int(round(args.stride * fs))

            intervals = _rhythm_intervals(rec, fs, root)
            if not intervals:
                # no rhythm markers → treat entire record as default
                intervals = [(0, int(r.sig_len), args.default_rhythm)]
            # Create windows constrained to intervals
            for s0, s1, lab in intervals:
                t = s0
                while t + win_n <= s1:
                    w.writerow([rec, f"{t/fs:.3f}", f"{(t+win_n)/fs:.3f}", lab, rec])
                    t += stride_n

    LOG.info("Wrote %s", out)

if __name__ == "__main__":
    main()
