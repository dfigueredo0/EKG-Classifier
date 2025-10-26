import torch
from ekgclf.models.resnet1d import ResNet1D
from ekgclf.models.head_multilabel import MultiLabelHead

def test_forward_shapes():
    body = ResNet1D(in_channels=12)
    head = MultiLabelHead(256, 5)
    model = torch.nn.Sequential(body, head)
    x = torch.randn(2, 12, 1000)
    y = model(x)
    assert y.shape == (2,5)
