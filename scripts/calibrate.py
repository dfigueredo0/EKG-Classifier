# scripts/calibrate.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from ekgclf.model import build_model
from ekgclf.data import build_val_dataset
from ekgclf.metrics import compute_nll  # or implement here

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

def find_temperature(model, val_loader, device):
    model.eval()
    scaled_model = ModelWithTemperature(model).to(device)

    optimizer = torch.optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=50)

    def _eval():
        loss_total = 0.0
        n = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = scaled_model(x)
            loss = nn.BCEWithLogitsLoss()(logits, y)
            loss_total += loss.item() * x.size(0)
            n += x.size(0)
        return loss_total / n

    def closure():
        optimizer.zero_grad()
        loss = 0.0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = scaled_model(x)
            l = nn.BCEWithLogitsLoss()(logits, y)
            l.backward()
            loss += l.item()
        return loss

    optimizer.step(closure)
    return scaled_model.temperature.detach().item()
