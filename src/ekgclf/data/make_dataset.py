from __future__ import annotations

import argparse, json, yaml, logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from ekgclf.data.ptbxl import iter_ptbxl
from ekgclf.data.mitbih import iter_mitbih, PTBXL_ORDER
from ekgclf.data.transforms import bandpass, ensure_leads, resample, zscore

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("ekgclf.data.make_dataset")

def process_record(rec: dict, cfg: dict) -> dict:
    sig = rec["signals"]
    fs = rec["fs"]
    required = cfg["signals"]["required_leads"]
    allow_derive = bool(cfg["signals"].get("allow_derive_avr", True))
    # reorder
    sig = ensure_leads(sig, rec["leads"], required, allow_derive_avr=allow_derive)
    # resample
    sig = resample(sig, fs, cfg["signals"]["resample_hz"])
    # filter
    bp = cfg["signals"]["bandpass"]
    sig = bandpass(sig, cfg["signals"]["resample_hz"], bp["low"], bp["high"], bp["order"])
    # zscore
    if cfg["signals"]["zscore"]:
        sig, mean, std = zscore(sig)
    else:
        mean = std = None
    return {
        "patient_id": rec["patient_id"],
        "signals": sig.astype(np.float32),  # [T, C]
        "labels": rec["labels"],
        "path": rec["record_path"],
        "mean": mean,
        "std": std,
    }

def _lift_mitbih_to_required(rec: dict, cfg: dict) -> dict:
    """
    MIT-BIH records provide 1-2 channels; we place them into a 12-lead frame (PTB-XL order),
    zero-fill missing, and update rec['leads'] so ensure_leads works unmodified.
    """
    required = cfg["signals"]["required_leads"]
    # Build a 12-lead frame and install any present channels (II/V1)
    present = [nm for nm in rec["leads"]]
    # signals are [T, Csrc]; transpose to [Csrc, T] to place; then back to [T, 12]
    x = rec["signals"].T  # [Csrc, T]
    T = x.shape[1]
    x12 = np.zeros((len(PTBXL_ORDER), T), dtype=np.float32)
    name_to_row = {nm: i for i, nm in enumerate(rec["leads"])}
    # map known channels
    if "II" in name_to_row:
        x12[PTBXL_ORDER.index("II")] = x[name_to_row["II"]]
    if "MLII" in name_to_row and "II" not in name_to_row:
        x12[PTBXL_ORDER.index("II")] = x[name_to_row["MLII"]]
    if "V1" in name_to_row:
        x12[PTBXL_ORDER.index("V1")] = x[name_to_row["V1"]]
    # time-major back
    rec["signals"] = x12.T  # [T, 12]
    rec["leads"] = PTBXL_ORDER[:]  # advertise a full set so ensure_leads() can slice/reorder
    # carry labels as list (aligns with PTB-XL)
    rec["labels"] = [rec.pop("label_rhythm")]
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, required=True, help="Path to raw dataset root")
    ap.add_argument("--out", type=str, required=True, help="Output directory for processed NPZ")
    ap.add_argument("--dataset", choices=["ptbxl", "mitbih"], required=True)
    ap.add_argument("--config", type=str, required=True, help="Path to data.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ptbxl":
        it = iter_ptbxl(args.raw, use_scored_subset=cfg["ptbxl"]["use_scored_subset"])
    elif args.dataset == "mitbih":
        # Expect cfg["mitbih"]["split_csv"] built offline or via a helper script
        it = iter_mitbih(args.raw, cfg["mitbih"]["split_csv"], target_fs=cfg["signals"]["resample_hz"])
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    index = []
    for i, rec in enumerate(it):
        try:
            if args.dataset == "mitbih":
                rec = _lift_mitbih_to_required(rec, cfg)
            proc = process_record(rec, cfg)
        except Exception as e:
            LOGGER.warning("Skip %s due to %s", rec.get("record_path"), e)
            continue
        
        npz_path = out_dir / f"{args.dataset}_{i:07d}.npz"
        np.savez_compressed(npz_path, signals=proc["signals"])
        index.append(
            {
                "id": i,
                "npz": str(npz_path),
                "patient_id": proc["patient_id"],
                "labels": proc["labels"],
                "path": proc["path"],
            }
        )
        if (i + 1) % 100 == 0:
            LOGGER.info("Processed %d records", i + 1)

    with open(out_dir / f"{args.dataset}_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    LOGGER.info("Done. Wrote %d examples to %s", len(index), out_dir)


if __name__ == "__main__":
    main()