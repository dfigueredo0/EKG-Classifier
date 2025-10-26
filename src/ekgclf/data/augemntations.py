from __future__ import annotations

import numpy as np


def jitter(x: np.ndarray, std: float = 0.01) -> np.ndarray:
    return x + np.random.normal(0, std, size=x.shape).astype(np.float32)


def scale(x: np.ndarray, min_s: float = 0.9, max_s: float = 1.1) -> np.ndarray:
    s = np.random.uniform(min_s, max_s)
    return (x * s).astype(np.float32)


def time_warp(x: np.ndarray, max_ratio: float = 0.05) -> np.ndarray:
    t, c = x.shape
    max_shift = int(t * max_ratio)
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return x
    out = np.zeros_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
    else:
        out[:shift] = x[-shift:]
    return out