import numpy as np
import torchvision.transforms as transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT = transforms.Compose([
    transforms.ToTensor()
])


class TrainTransform(transforms.Compose):
    def __init__(self, crop_size, rrc_scale=(0.4, 1.), mean=IMAGENET_MEAN, std=IMAGENET_STD):
        transforms_ = [
            transforms.RandomResizedCrop(size=crop_size, scale=rrc_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        super().__init__(transforms_)


class ValTransform(transforms.Compose):
    def __init__(self, crop_size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        transforms_ = [
            transforms.Resize(int(2 ** (np.ceil(np.log2(crop_size))))),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        super().__init__(transforms_)
