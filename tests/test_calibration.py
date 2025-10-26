import torch
from ekgclf.calibration import TemperatureScaler

def test_temperature_fit_runs():
    logits = torch.randn(64, 3)
    labels = (torch.rand(64, 3) > 0.7).float()
    t = TemperatureScaler()
    T = t.fit(logits, labels, max_iter=50)
    assert T > 0
