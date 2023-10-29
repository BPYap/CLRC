import json
import os
from typing import Optional

import pandas as pd
import torch.utils.data as data
from PIL import Image

from clrc.data.base_data import BaseDataModule
from clrc.transforms import TrainTransform, ValTransform

MEAN = (0.4772, 0.4405, 0.4100)
STD = (0.2960, 0.2876, 0.2935)


# === Dataset === #
class WIDERDataset(data.Dataset):
    def __init__(self, image_dir, label_json, num_views, transform):
        """
        Args:
            image_dir (string): Root directory of images.
            label_json (string): Path to the json file with class annotations.
            num_views (int): Number of views to generate from each image.
        """
        self.image_dir = image_dir
        with open(label_json, 'r') as f:
            self.data = json.load(f)['images']
        self.num_views = num_views
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], label_tensor
        sample = self.data[index]

        image_path = os.path.join(self.image_dir, sample['file_name'])
        label = sample['scene_id']

        image = Image.open(image_path)
        if self.num_views == 1:
            return self.transform(image), label
        else:
            return [self.transform(image) for _ in range(self.num_views)], label


class WIDERDatasetUnlabeled(WIDERDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v]
        image_tensors, _ = super().__getitem__(index)

        return image_tensors


class WIDERDatasetCluster(data.Dataset):
    def __init__(self, image_dir, label_csv, num_views, cluster_name, transform):
        self.image_dir = image_dir
        self.df = pd.read_csv(label_csv)
        self.num_views = num_views
        self.cluster_name = cluster_name
        self.transform = transform

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], label_tensor
        row = self.df.iloc[index]

        image_path = os.path.join(self.image_dir, row['path'])
        cluster = int(row[self.cluster_name])

        image = Image.open(image_path)
        if self.num_views == 1:
            return self.transform(image), cluster
        else:
            return [self.transform(image) for _ in range(self.num_views)], cluster


class WIDERDatasetAttribute(WIDERDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grouped_attributes = dict()
        for index in range(len(self.data)):
            sample = self.data[index]
            grouped = [[0] for _ in range(14)]
            # merge bounding box attributes
            for t in sample['targets']:
                attributes = t['attribute']
                for i in range(14):
                    if attributes[i] == 1:
                        grouped[i] = [1]
            self.grouped_attributes[index] = grouped

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], [[label_tensor_1_1, ...], ..., [label_tensor_g_1, ...]]
        image_tensors, _ = super().__getitem__(index)
        group_attributes = self.grouped_attributes[index]

        return image_tensors, group_attributes


class WIDERDatasetCCLK(WIDERDatasetAttribute):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], [label_tensor_1, ..., label_tensor_k]
        image_tensors, group_attributes = super().__getitem__(index)

        return image_tensors, [attr for group in group_attributes for attr in group]


# === PL DataModule === #
class WIDERDataModule(BaseDataModule):
    name = "WIDER Attribute"
    source = "http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html"
    args_schema = {
        **BaseDataModule.args_schema,
        "data_dir": (str, None, "Data directory."),
        "image_size": (int, 224, "Resize the shorter side of each image to the specified size."),
        "num_views": (int, 1, "Number of views to generate from each image.")
    }

    def __init__(self, data_dir, image_size, num_views, dataset_cls=WIDERDataset, **kwargs):
        """
        Directory structure of `data_dir`:
            `data_dir`
                |- Image
                |- wider_attribute_annotation
                    |- wider_attribute_test.json
                    |- wider_attribute_trainval.json
        """
        super().__init__(**kwargs)
        hparams = {
            "image_size": image_size,
            "num_views": num_views
        }
        self.save_hyperparameters(hparams)

        self.data_dir = data_dir
        self.image_size = image_size
        self.num_views = num_views
        self.dataset_cls = dataset_cls

        self.image_dir = os.path.join(self.data_dir, "Image")
        self.train_json = os.path.join(self.data_dir, "wider_attribute_annotation", "wider_attribute_trainval.json")
        self.val_json = os.path.join(self.data_dir, "wider_attribute_annotation", "wider_attribute_test.json")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = self.dataset_cls
        self.dataset_train = dataset(
            image_dir=self.image_dir,
            label_json=self.train_json,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, rrc_scale=(0.3, 1.), mean=MEAN, std=STD)
        )
        self.dataset_val = dataset(
            image_dir=self.image_dir,
            label_json=self.val_json,
            num_views=1,
            transform=ValTransform(crop_size=self.image_size, mean=MEAN, std=STD)
        )


class WIDERDataModuleUnlabeled(WIDERDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=WIDERDatasetUnlabeled, **kwargs)

    def setup(self, stage: Optional[str] = None):
        super().setup()
        del self.dataset_val


class WIDERDataModuleCluster(BaseDataModule):
    args_schema = {
        **BaseDataModule.args_schema,
        "data_dir": (str, None, "Data directory."),
        "image_size": (int, 224, "Resize the shorter side of each image to the specified size."),
        "num_views": (int, 1, "Number of views to generate from each image."),
        "cluster_name": (str, "label_gran_7", "Name of the clustering scheme.")
    }

    def __init__(self, data_dir, image_size, num_views, cluster_name, **kwargs):
        """ Cluster labels obtained from https://github.com/Crazy-Jack/Cl-InfoNCE/tree/main/data_processing."""
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_views = num_views
        self.cluster_name = cluster_name
        self.image_dir = os.path.join(data_dir, "Image")
        self.train_csv = os.path.join(data_dir, "rank_H", "meta_data_train.csv")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = WIDERDatasetCluster(
            image_dir=self.image_dir,
            label_csv=self.train_csv,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, rrc_scale=(0.3, 1.), mean=MEAN, std=STD),
            cluster_name=self.cluster_name
        )


class WIDERDataModuleAttribute(WIDERDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=WIDERDatasetAttribute, **kwargs)


class WIDERDataModuleCCLK(WIDERDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=WIDERDatasetCCLK, **kwargs)
