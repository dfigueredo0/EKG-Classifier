import numpy as np
from ekgclf.data.transforms import resample, zscore, bandpass

def test_resample_identity():
    x = np.random.randn(1000, 12).astype(np.float32)
    y = resample(x, 500, 500)
    assert y.shape == x.shape

def test_zscore():
    x = np.random.randn(1000, 12).astype(np.float32) * 2 + 3
    y, m, s = zscore(x)
    assert np.allclose(y.mean(0), 0, atol=1e-1)
    assert np.allclose(y.std(0), 1, atol=1e-1)

def test_bandpass_shapes():
    x = np.random.randn(1000, 12).astype(np.float32)
    y = bandpass(x, 500, 0.5, 40.0, 4)
    assert y.shape == x.shape
