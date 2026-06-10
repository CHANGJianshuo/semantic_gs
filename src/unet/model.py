"""Minimal U-Net for 2D semantic segmentation.

Classic Ronneberger encoder-decoder with skip connections, following the
milesial/Pytorch-UNet bilinear variant that the AAE5303 UNet demo references.
Self-contained single file -- no external segmentation libraries.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Bilinear upsample -> concat skip -> DoubleConv.

    up_in_ch  channels of the feature coming up from below
    skip_ch   channels of the skip-connection feature
    out_ch    output channels
    """

    def __init__(self, up_in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(up_in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Classic U-Net (bilinear-upsampling variant).  base=64 -> ~17M params."""

    def __init__(self, n_channels: int = 3, n_classes: int = 8, base: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.inc = DoubleConv(n_channels, c1)        # -> c1
        self.down1 = Down(c1, c2)                    # -> c2
        self.down2 = Down(c2, c3)                    # -> c3
        self.down3 = Down(c3, c4)                    # -> c4
        self.down4 = Down(c4, c4)                    # bottleneck -> c4

        self.up1 = Up(c4, c4, c3)                    # (c4 up + c4 skip) -> c3
        self.up2 = Up(c3, c3, c2)                    # (c3 up + c3 skip) -> c2
        self.up3 = Up(c2, c2, c1)                    # (c2 up + c2 skip) -> c1
        self.up4 = Up(c1, c1, c1)                    # (c1 up + c1 skip) -> c1
        self.outc = nn.Conv2d(c1, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              n_classes: int, ignore_index: int = 255, eps: float = 1e-6) -> torch.Tensor:
    """Multi-class soft Dice loss (mean over classes, ignoring `ignore_index`)."""
    valid = (target != ignore_index)
    tgt = target.clone()
    tgt[~valid] = 0
    target_onehot = F.one_hot(tgt, n_classes).permute(0, 3, 1, 2).float()
    target_onehot = target_onehot * valid.unsqueeze(1).float()
    probs = F.softmax(logits, dim=1) * valid.unsqueeze(1).float()
    dims = (0, 2, 3)
    inter = (probs * target_onehot).sum(dims)
    denom = probs.sum(dims) + target_onehot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()
