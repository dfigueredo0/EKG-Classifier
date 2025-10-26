from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]


def patient_level_split(patient_ids: Iterable, train: float, val: float, test: float, seed: int) -> SplitIndices:
    """Deterministic patient-level split; prevents leakage."""
    ids = np.array(sorted(set(patient_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_test = n - n_train - n_val
    train_ids = set(ids[:n_train].tolist())
    val_ids = set(ids[n_train : n_train + n_val].tolist())
    test_ids = set(ids[n_train + n_val :].tolist())
    # Map back to indices
    pid_list = list(patient_ids)
    idx_train, idx_val, idx_test = [], [], []
    for i, p in enumerate(pid_list):
        if p in train_ids:
            idx_train.append(i)
        elif p in val_ids:
            idx_val.append(i)
        elif p in test_ids:
            idx_test.append(i)
    return SplitIndices(idx_train, idx_val, idx_test)