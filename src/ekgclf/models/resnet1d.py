from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, kernel_size: int = 3, dropout: float = 0.0, downsample: nn.Module | None = None) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        if downsample is not None:
            self.shortcut = downsample
        else:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes * self.expansion:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(planes * self.expansion),
                )

    def _ensure_channels(self, x: torch.Tensor) -> None:
        in_channel = x.size(1)
        out_channel = self.planes * self.expansion
        
        proj = (self.stride != 1) or (in_channel != out_channel)
        
        if isinstance(self.shortcut, nn.Sequential) and proj:
            conv = None
            for m in self.shortcut.modules():
                if isinstance(m, nn.Conv1d):
                    conv = m
                    break
            if conv and conv.in_channels == in_channel and conv.out_channels == out_channel and conv.stride[0] == self.stride:
                return
        if not proj:
            self.shortcut = nn.Identity()
            return
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=self.stride, bias=False),
            nn.BatchNorm1d(out_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_channels(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, base_channels=64, blocks=[2, 2, 2, 2], kernel_size=7, stride=2, downsample=True, dropout=0.1, global_pool="avg", feat_dim=256) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.in_planes = base_channels
        self.layer1 = self._make_layer(base_channels, blocks[0], kernel_size, 1, dropout, downsample)
        self.layer2 = self._make_layer(base_channels * 2, blocks[1], kernel_size, 2, dropout, downsample)
        self.layer3 = self._make_layer(base_channels * 4, blocks[2], kernel_size, 2, dropout, downsample)
        self.layer4 = self._make_layer(base_channels * 8, blocks[3], kernel_size, 2, dropout, downsample)
        self.global_pool = global_pool
        self.feat = nn.Linear(base_channels * 8, feat_dim)
        
    def _make_layer(self, planes, blocks, kernel_size, stride, dropout, downsample_flag):
        projection = None
        if stride != 1 or self.in_planes != planes:
            down_layers = []
            if downsample_flag:
                down_layers.append(nn.Conv1d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False))
                down_layers.append(nn.BatchNorm1d(planes))
            projection = nn.Sequential(*down_layers) if down_layers else None

        layers = [BasicBlock1D(self.in_planes, planes, stride=stride, downsample=projection, kernel_size=kernel_size, dropout=dropout)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_planes, planes, kernel_size=kernel_size, dropout=dropout, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):  # x: [B, C, T]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.global_pool == "avg":
            x = torch.mean(x, dim=-1)
        elif self.global_pool == "max":
            x = torch.amax(x, dim=-1)
        else:
            raise ValueError("Unsupported global pool")
        x = self.feat(x)
        x = F.relu(x, inplace=True)
        return x  # [B, feat_dim]