from __future__ import annotations

import argparse

def main():
    ap = argparse.ArgumentParser(description="Class-conditioned 1-D diffusion (stub)")
    ap.add_argument("--mode", choices=["train", "sample"], required=True)
    ap.add_argument("--class", dest="klass", type=str, help="Target class label for conditioning")
    args = ap.parse_args()
    if args.mode == "train":
        print("TODO: implement 1-D diffusion training for rare classes.")
    else:
        print(f"TODO: generate samples conditioned on {args.klass!r}.")

if __name__ == "__main__":
    main()
