import torch

from polypseg.model import UNetSmall, dice_loss


def test_unet_preserves_spatial_shape() -> None:
    model = UNetSmall(base_channels=4)
    output = model(torch.randn(2, 3, 65, 67))
    assert output.shape == (2, 1, 65, 67)


def test_dice_loss_is_near_zero_for_confident_correct_prediction() -> None:
    targets = torch.ones(1, 1, 8, 8)
    assert dice_loss(torch.full_like(targets, 12.0), targets).item() < 0.001
