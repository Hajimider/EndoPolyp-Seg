import torch

from polypseg.model import UNetResNet18, UNetSmall, dice_loss


def test_unet_preserves_spatial_shape() -> None:
    model = UNetSmall(base_channels=4)
    output = model(torch.randn(2, 3, 65, 67))
    assert output.shape == (2, 1, 65, 67)


def test_dice_loss_is_near_zero_for_confident_correct_prediction() -> None:
    targets = torch.ones(1, 1, 8, 8)
    assert dice_loss(torch.full_like(targets, 12.0), targets).item() < 0.001


def test_resnet18_unet_preserves_spatial_shape_without_download() -> None:
    model = UNetResNet18(pretrained=False).eval()
    output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)
