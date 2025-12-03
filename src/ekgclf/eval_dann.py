# src/ekgclf/eval_dann.py

import csv
import torch

from ekgclf.config import load_dann_config
from ekgclf.train_dann import build_model
from ekgclf.data import get_dataloaders


@torch.no_grad()
def main():
    cfg = load_dann_config("configs/dann.yaml")
    _, _, dl_test = get_dataloaders(cfg)

    model = build_model(cfg)
    ckpt = torch.load("checkpoints/dann/best.pt", map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    rows = []
    for i, (x, y, d) in enumerate(dl_test):
        # Assuming batch_size=1 here; otherwise loop over batch
        x = x
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        trues = torch.argmax(y, dim=-1)

        for j in range(x.size(0)):
            idx = i * dl_test.batch_size + j
            rows.append({
                "idx": idx,
                "domain": int(d[j]),
                "true": int(trues[j]),
                "pred": int(preds[j]),
                "conf": float(probs[j, preds[j]].item()),
            })

    with open("dann_test_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
