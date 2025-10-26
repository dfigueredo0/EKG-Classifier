import json
import numpy as np
from pathlib import Path
from ekgclf.inference import InferenceEngine

def test_inference_engine_smoke(tmp_path: Path):
    # Create fake ckpt
    labels_map = {"X": 0, "Y": 1}
    import torch
    from ekgclf.models.resnet1d import ResNet1D
    from ekgclf.models.head_multilabel import MultiLabelHead
    model = torch.nn.Sequential(ResNet1D(), MultiLabelHead(256, len(labels_map)))
    ckpt = {"epoch": 0, "model": model.state_dict(), "labels_map": labels_map}
    ckpt_path = tmp_path / "best.pt"
    torch.save(ckpt, ckpt_path)
    eng = InferenceEngine(str(ckpt_path))
    x = np.random.randn(2, 1000, 12).astype(np.float32)
    out = eng.predict(x)
    assert out.probs.shape == (2,2)
