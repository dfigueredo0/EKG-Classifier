# src/ekgclf/data_utils.py
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def make_class_balanced_sampler(dataset):
    """
    Compute per-sample weights = sum of inverse label frequencies for each sample.
    dataset[i] must return (x, multihot_y)
    """
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y.numpy())
    labels = np.stack(labels)   # [N, C]

    freq = labels.sum(axis=0) + 1e-6
    inv = 1.0 / freq            # [C]

    weights = (labels * inv).sum(axis=1)  # [N]
    weights = torch.tensor(weights, dtype=torch.float32)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
