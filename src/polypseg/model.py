"""A deliberately small U-Net for CPU-friendly binary segmentation."""

from __future__ import annotations

import torch
import torchvision.models as tv_models
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


class UNetResNet18(nn.Module):
    """U-Net decoder with a compact ImageNet-pretrained ResNet18 encoder."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            backbone = tv_models.resnet18(weights=weights)
        except (OSError, RuntimeError) as exc:
            if not pretrained:
                raise
            raise RuntimeError(
                "ResNet18 pretrained weights are unavailable. Use --no-pretrained for an offline run."
            ) from exc
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(96, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def set_encoder_trainable(self, full: bool = False) -> None:
        """Freeze early features; optionally unfreeze the complete encoder."""
        encoder = (self.stem, self.layer1, self.layer2, self.layer3, self.layer4)
        for module in encoder:
            for parameter in module.parameters():
                parameter.requires_grad = full
        if not full:
            for parameter in self.layer4.parameters():
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        stem = self.stem(x)
        skip1 = self.layer1(self.pool(stem))
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        x = self.layer4(skip3)
        x = self.dec4(torch.cat([self.up4(x), skip3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), skip2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), skip1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), stem], dim=1))
        return self.head(torch.nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False))


def build_model(name: str, *, pretrained: bool = True) -> nn.Module:
    if name == "unet":
        return UNetSmall(base_channels=16)
    if name == "resnet18_unet":
        return UNetResNet18(pretrained=pretrained)
    raise ValueError(f"Unknown segmentation model: {name}")


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(dim=(1, 2, 3)) + eps
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (numerator / denominator).mean()
