import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(encoder="efficientnet-b4", pretrained=True):
    weights = "imagenet" if pretrained else None
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation=None,  # raw logits; sigmoid applied in loss/predict
    )
    return model


class DiceFocalLoss(nn.Module):
    def __init__(self, focal_weight=0.5, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice = smp.losses.DiceLoss(mode="binary", smooth=1.0)
        self.focal = smp.losses.FocalLoss(mode="binary", alpha=alpha, gamma=gamma)

    def forward(self, logits, targets):
        return (1 - self.focal_weight) * self.dice(logits, targets) + \
               self.focal_weight * self.focal(logits, targets)
