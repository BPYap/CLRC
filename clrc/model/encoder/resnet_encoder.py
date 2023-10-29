import torch.nn as nn
import torchvision

from clrc.model.encoder.base_encoder import BaseEncoder

BACBONE_CONSTRUCTORS = {
    'resnet-18': torchvision.models.resnet.resnet18,
    'resnet-34': torchvision.models.resnet.resnet34,
    'resnet-50': torchvision.models.resnet.resnet50,
    'resnet-101': torchvision.models.resnet.resnet101,
    'resnet-152': torchvision.models.resnet.resnet152
}

RESNET_OUT_CHANNELS = {
    'resnet-18': 512,
    'resnet-34': 512,
    'resnet-50': 2048,
    'resnet-101': 2048,
    'resnet-152': 2048
}


class ResNetEncoder(BaseEncoder):
    def __init__(self, backbone, imagenet_init):
        super().__init__(RESNET_OUT_CHANNELS[backbone])

        model = BACBONE_CONSTRUCTORS[backbone](pretrained=imagenet_init)
        layers = list(model.children())[:-2]
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, avg_pool=True):
        feature_maps = self.layers(x)

        return self.avg_pool(feature_maps).flatten(1) if avg_pool else feature_maps


class ResNetEncoderMini(ResNetEncoder):
    def __init__(self, backbone, imagenet_init):
        super().__init__(backbone.replace("-mini", ""), imagenet_init)
        self.layers[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        del self.layers[3]  # remove first max pooling layer
