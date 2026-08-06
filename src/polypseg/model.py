"""A deliberately small U-Net for CPU-friendly binary segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNetSmall(nn.Module):
    """Three encoder downsamples keep the model small while preserving U-Net skip links."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        channels = base_channels
        self.enc1 = ConvBlock(3, channels)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.enc3 = ConvBlock(channels * 2, channels * 4)
        self.bottleneck = ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(channels * 2, channels)
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    @staticmethod
    def _decode(x: torch.Tensor, skip: torch.Tensor, up: nn.Module, block: nn.Module) -> torch.Tensor:
        x = up(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return block(torch.cat([x, skip], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self._decode(bottleneck, enc3, self.up3, self.dec3)
        dec2 = self._decode(dec3, enc2, self.up2, self.dec2)
        dec1 = self._decode(dec2, enc1, self.up1, self.dec1)
        return self.head(dec1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(dim=(1, 2, 3)) + eps
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (numerator / denominator).mean()
