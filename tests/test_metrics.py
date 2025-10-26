import numpy as np
from ekgclf.metrics import expected_calibration_error, auroc_per_label, f1_per_label

def test_ece_basic():
    y = np.array([[1,0],[0,1],[1,0],[0,1]])
    p = np.array([[0.9,0.1],[0.1,0.9],[0.8,0.2],[0.2,0.8]])
    ece = expected_calibration_error(y,p,n_bins=10)
    assert 0 <= ece <= 0.1

def test_auroc_f1_shapes():
    y = np.array([[1,0,1],[0,1,0]])
    p = np.array([[0.9,0.2,0.8],[0.1,0.7,0.3]])
    au = auroc_per_label(y,p)
    f1 = f1_per_label(y,(p>=0.5).astype(int))
    assert "macro" in au and "macro" in f1 and "micro" in f1
