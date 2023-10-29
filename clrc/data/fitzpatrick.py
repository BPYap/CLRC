import os
from typing import Optional

import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from clrc.data.base_data import BaseDataModule
from clrc.transforms import ValTransform

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TrainTransform(transforms.Compose):
    def __init__(self, crop_size, mean=MEAN, std=STD):
        transforms_ = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomCrop(size=(crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        super().__init__(transforms_)


# === Dataset === #
class FitzpatrickDataset(data.Dataset):
    CLASS_MAPPINGS = {
        "non-neoplastic": 0,
        "benign": 0,
        "malignant": 1
    }

    def __init__(self, image_dir, label_csv, num_views, transform):
        """
        Args:
            image_dir (string): Root directory of images.
            label_csv (string): Path to the csv file with class annotations.
            num_views (int): Number of views to generate from each image.
        """
        self.image_dir = image_dir
        self.df = pd.read_csv(label_csv)
        self.df["fitzpatrick_scale"] -= 1  # change to zero-based indexing
        self.num_views = num_views
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], label_tensor
        sample = self.df.iloc[index]
        sample_id = sample['md5hash']
        image_path = os.path.join(self.image_dir, f"{sample_id}.jpg")
        label = sample['three_partition_label']
        label = self.CLASS_MAPPINGS[label]
        labels = [label, sample['fitzpatrick_scale']]
        image = Image.open(image_path)
        if self.num_views == 1:
            return self.transform(image), labels
        else:
            return [self.transform(image) for _ in range(self.num_views)], labels


class FitzpatrickDatasetLabeled(FitzpatrickDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v]
        image_tensors, (label, _) = super().__getitem__(index)

        return image_tensors, label


class FitzpatrickDatasetUnlabeled(FitzpatrickDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v]
        image_tensors, _ = super().__getitem__(index)

        return image_tensors


class FitzpatrickDatasetAttribute(FitzpatrickDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], [[label_tensor_1_1, ...], ..., [label_tensor_g_1, ...]]
        image_tensors, labels = super().__getitem__(index)

        row = self.df.iloc[index]
        attributes = []
        for skin_type in range(6):
            one_hot = [0, 0, 0]  # benign, malignant, non-group
            if row['fitzpatrick_scale'] == skin_type:
                one_hot[labels[0]] = 1
            else:
                one_hot[-1] = 1
            attributes.append(one_hot)

        return image_tensors, attributes


# === PL DataModule === #
class FitzpatrickDataModule(BaseDataModule):
    name = "Fitzpatrick17k"
    source = "https://github.com/mattgroh/fitzpatrick17k"
    args_schema = {
        **BaseDataModule.args_schema,
        "data_dir": (str, None, "Data directory."),
        "image_size": (int, 224, "Resize the shorter side of each image to the specified size."),
        "num_views": (int, 1, "Number of views to generate from each image.")
    }

    def __init__(self, data_dir, image_size, num_views, dataset_cls=FitzpatrickDataset, **kwargs):
        """
        Directory structure of `data_dir`:
            `data_dir`
                |- data
                    |-finalfitz17k-256
                |- fitzpatrick17k.csv
        """
        super().__init__(pin_memory=False, **kwargs)
        hparams = {
            "image_size": image_size,
            "num_views": num_views
        }
        self.save_hyperparameters(hparams)

        self.data_dir = data_dir
        self.image_size = image_size
        self.num_views = num_views
        self.dataset_cls = dataset_cls

        self.image_dir = os.path.join(self.data_dir, "data", "finalfitz17k-256")
        self.csv = os.path.join(self.data_dir, "fitzpatrick17k.csv")

        self.train_csv = os.path.join(self.data_dir, "train.csv")
        self.val_csv = os.path.join(self.data_dir, "val.csv")
        self.test_csv = os.path.join(self.data_dir, "test.csv")

    def prepare_data(self):
        if not os.path.exists(self.train_csv):
            df = pd.read_csv(self.csv)
            # drop unknown skin type
            df = df.drop(df[df.fitzpatrick_scale == -1].index)

            train, val_test = train_test_split(df, test_size=0.2, random_state=0)
            val, test = train_test_split(val_test, test_size=0.5, random_state=0)

            train.to_csv(self.train_csv, index=False)
            val.to_csv(self.val_csv, index=False)
            test.to_csv(self.test_csv, index=False)

    def setup(self, stage: Optional[str] = None):
        dataset = self.dataset_cls
        self.dataset_train = dataset(
            image_dir=self.image_dir,
            label_csv=self.train_csv,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, mean=MEAN, std=STD)
        )
        self.dataset_val = dataset(
            image_dir=self.image_dir,
            label_csv=self.val_csv,
            num_views=1,
            transform=ValTransform(crop_size=self.image_size, mean=MEAN, std=STD)
        )
        self.dataset_test = dataset(
            image_dir=self.image_dir,
            label_csv=self.test_csv,
            num_views=1,
            transform=ValTransform(crop_size=self.image_size, mean=MEAN, std=STD)
        )


class FitzpatrickDataModuleLabeled(FitzpatrickDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=FitzpatrickDatasetLabeled, **kwargs)


class FitzpatrickDataModuleUnlabeled(FitzpatrickDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=FitzpatrickDatasetUnlabeled, **kwargs)


class FitzpatrickDataModuleAttribute(FitzpatrickDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=FitzpatrickDatasetAttribute, **kwargs)
