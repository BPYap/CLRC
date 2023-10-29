import torch

from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder


class BaseModel(torch.nn.Module):
    args_schema = {
        "backbone": (str, "resnet-50", f"Choose from {AVAILABLE_BACKBONES}."),
        "imagenet_init": (bool, False, "Initialize with built-in weights (if available) pretrained on ILSVRC-2012."),
        "backbone_weights": (str, None, "Path to pretrained backbone weights."),
        "freeze_backbone": (bool, False, "Freeze the backbone weights."),
        "momentum_encoder": (bool, False, "Initialize encoder from momentum encoder.")
    }

    def __init__(self, backbone, imagenet_init, backbone_weights, freeze_backbone, momentum_encoder):
        super().__init__()
        self.encoder = get_encoder(
            backbone=backbone,
            imagenet_init=imagenet_init,
            pretrain_path=backbone_weights,
            freeze_weights=freeze_backbone,
            momentum=momentum_encoder
        )
