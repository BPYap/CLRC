import torch

from clrc.model.encoder.base_encoder import BaseEncoder
from clrc.model.encoder.resnet_encoder import ResNetEncoder, ResNetEncoderMini

AVAILABLE_BACKBONES = [
    'resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152',
    'resnet-50-mini'
]


def get_encoder(backbone, imagenet_init, pretrain_path, freeze_weights, momentum=False):
    if backbone.startswith("resnet"):
        if backbone.endswith("mini"):
            encoder = ResNetEncoderMini(backbone, imagenet_init)
        else:
            encoder = ResNetEncoder(backbone, imagenet_init)
    else:
        raise ValueError(f"Unknown backbone: '{backbone}'.")

    if pretrain_path:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        encoder_state_dict = {}
        prefix = "encoder" if not momentum else "target_encoder"
        for k, v in state_dict.items():
            if k.startswith(f"{prefix}."):
                new_key = k.replace(f"{prefix}.", "", 1)
                encoder_state_dict[new_key] = v
        print(encoder.load_state_dict(encoder_state_dict, strict=True))

    if freeze_weights:
        encoder.freeze()

    return encoder
