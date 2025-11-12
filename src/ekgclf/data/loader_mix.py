import json
from pathlib import Path
from typing import List, Dict
import numpy as np 

def load_indicies(*json_paths: str) -> List[Dict]:
    items = []
    for p in json_paths:
        with open(p, 'r', encoding="utf-8") as f:
            items.extend(json.load(f))
    return items

def load_npz_items(item: Dict) -> np.ndarray:
    d = np.load(item["npz"])
    return d["signals"].astype(np.float32)

def make_fast_loaders(ds_train, ds_val, train_bs, eval_bs, pin=True):
    """
    Builds high-throughput DataLoaders.
    """
    from torch.utils.data import DataLoader
    common = dict(
        num_workers=8,                 # tune 6–12 if you have spare cores
        pin_memory=pin,                # True iff GPU
        persistent_workers=True,       # keeps workers alive
        prefetch_factor=4,             # 2–6 depending on sample cost
        drop_last=False,
    )
    dl_train = DataLoader(ds_train, batch_size=train_bs, shuffle=True, **common)
    dl_val   = DataLoader(ds_val,   batch_size=eval_bs,  shuffle=False, **common)
    return dl_train, dl_val