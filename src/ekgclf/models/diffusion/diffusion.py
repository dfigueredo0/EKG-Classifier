import torch
import torch.nn as nn
import numpy as np

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard DDPM-style sinusoidal time embedding for t in [0, T-1]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int64
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(
                np.log(1.0), np.log(10000.0), half, device=t.device
            )
        )
        args = t[:, None].float() / freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # (B, dim)

class SimpleUNet1D(nn.Module):
    def __init__(self, in_ch=12, base_ch=32, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        cond_ch = base_ch if num_classes is not None else 0

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_ch + cond_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool1d(2)  # /2

        self.enc2 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool1d(2)  # /4 total

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder: two upsampling stages to get back to original length
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)  # /2 total
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 2, 3, padding=1),  # cat(u1, e2) -> 4*base_ch
            nn.ReLU(),
            nn.Conv1d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=2, stride=2)      # back to original T
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch, 3, padding=1),  # cat(u2, e1) -> 2*base_ch
            nn.ReLU(),
            nn.Conv1d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv1d(base_ch, in_ch, 1)

        if num_classes is not None:
            self.class_emb = nn.Linear(num_classes, cond_ch)

    def forward(self, x, y=None):
        # x: (B, C, T); y: (B, num_classes) one-hot labels
        if self.num_classes is not None and y is not None:
            emb = self.class_emb(y)  # (B, cond_ch)
            emb = emb[..., None].expand(-1, -1, x.size(-1))
            x = torch.cat([x, emb], dim=1)

        # Encoder
        e1 = self.enc1(x)    # [B, base_ch, T]
        p1 = self.pool1(e1)  # [B, base_ch, T/2]

        e2 = self.enc2(p1)   # [B, 2*base_ch, T/2]
        p2 = self.pool2(e2)  # [B, 2*base_ch, T/4]

        # Bottleneck
        b = self.bottleneck(p2)  # [B, 2*base_ch, T/4]

        # Decoder
        u1 = self.up1(b)         # [B, 2*base_ch, T/2]
        u1 = torch.cat([u1, e2], dim=1)  # [B, 4*base_ch, T/2]
        d1 = self.dec1(u1)       # [B, 2*base_ch, T/2]

        u2 = self.up2(d1)        # [B, base_ch, T]
        u2 = torch.cat([u2, e1], dim=1)  # [B, 2*base_ch, T]
        d2 = self.dec2(u2)       # [B, base_ch, T]

        out = self.out(d2)       # [B, in_ch, T]
        return out