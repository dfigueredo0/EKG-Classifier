from __future__ import annotations

import numpy as np
import torch
import random
class ECGAugment:
    def __init__(self,
                 amp_scale_range=(0.9, 1.1),
                 noise_std=0.01,
                 time_shift_frac=0.05,
                 p_amp=0.5,
                 p_noise=0.5,
                 p_shift=0.5):
        self.amp_scale_range = amp_scale_range
        self.noise_std = noise_std
        self.time_shift_frac = time_shift_frac
        self.p_amp = p_amp
        self.p_noise = p_noise
        self.p_shift = p_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, T)
        if random.random() < self.p_amp:
            scale = random.uniform(*self.amp_scale_range)
            x = x * scale

        if random.random() < self.p_noise:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        if random.random() < self.p_shift:
            shift = int(self.time_shift_frac * x.shape[-1])
            k = random.randint(-shift, shift)
            x = torch.roll(x, shifts=k, dims=-1)

        return x

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