import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(encoder="efficientnet-b0", pretrained=True):
    weights = "imagenet" if pretrained else None
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation=None,  # raw logits; sigmoid applied in loss/predict
    )
    return model


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        )
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
