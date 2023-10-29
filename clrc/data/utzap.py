import os
from collections import defaultdict
from typing import Optional

import pandas as pd
import torch.utils.data as data
from PIL import Image

from clrc.data.base_data import BaseDataModule
from clrc.transforms import TrainTransform, ValTransform

MEAN = (0.8342, 0.8142, 0.8081)
STD = (0.2804, 0.3014, 0.3072)


# === Dataset === #
class UTZapposDataset(data.Dataset):
    def __init__(self, image_dir, label_csv, num_views, transform):
        """
        Args:
            image_dir (string): Root directory of images.
            label_csv (string): Path to the csv file with class annotations.
            num_views (int): Number of views to generate from each image.
        """
        self.image_dir = image_dir
        self.df = pd.read_csv(label_csv)
        self.num_views = num_views
        self.transform = transform

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], label_tensor
        sample = self.df.iloc[index]

        image_path = os.path.join(self.image_dir, sample['path'])
        label = sample['class']
        image_path = image_path.replace("%2E", ".")
        image = Image.open(image_path)
        if self.num_views == 1:
            return self.transform(image), label
        else:
            return [self.transform(image) for _ in range(self.num_views)], label


class UTZapposDatasetUnlabeled(UTZapposDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v]
        image_tensors, _ = super().__getitem__(index)

        return image_tensors


class UTZapposDatasetCluster(data.Dataset):
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
        image_path = image_path.replace("%2E", ".")
        image = Image.open(image_path)
        if self.num_views == 1:
            return self.transform(image), cluster
        else:
            return [self.transform(image) for _ in range(self.num_views)], cluster


class UTZapposDatasetCCLK(UTZapposDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], [label_tensor_1, ..., label_tensor_k]
        image_tensors, _ = super().__getitem__(index)

        row = self.df.iloc[index]
        attributes = []
        for key in self.df.keys()[3:]:
            attributes.append(row[key])

        return image_tensors, attributes


class UTZapposDatasetAttribute(UTZapposDataset):
    def __init__(self, meta_csv, **kwargs):
        super().__init__(**kwargs)
        self.meta_df = pd.read_csv(meta_csv)
        self.grouped_attributes = defaultdict(list)
        for attribute_name in self.meta_df.keys()[26:]:
            group = attribute_name.split(".")[0]
            self.grouped_attributes[group].append(attribute_name)
        self.cid_index = {row["CID"]: i for i, row in self.meta_df.iterrows()}

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], [[label_tensor_1_1, ...], ..., [label_tensor_g_1, ...]]
        image_tensors, _ = super().__getitem__(index)

        row = self.df.iloc[index]
        path = row['path']
        cid = "-".join(path.split("/")[-1].split(".")[:-1])
        meta_row = self.meta_df.iloc[self.cid_index[cid]]
        attributes = []
        for group in self.grouped_attributes.values():
            grouped_attributes = []
            for name in group:
                grouped_attributes.append(meta_row[name])
            attributes.append(grouped_attributes)

        return image_tensors, attributes


# === PL DataModule === #
class UTZapposDataModule(BaseDataModule):
    name = "UT Zappos50K"
    source = "https://vision.cs.utexas.edu/projects/finegrained/utzap50k/"
    args_schema = {
        **BaseDataModule.args_schema,
        "data_dir": (str, None, "Data directory."),
        "image_size": (int, 32, "Resize the shorter side of each image to the specified size."),
        "num_views": (int, 1, "Number of views to generate from each image.")
    }

    def __init__(self, data_dir, image_size, num_views, dataset_cls=UTZapposDataset, **kwargs):
        """
        Directory structure of `data_dir`:
            `data_dir`
                |- rank_H
                    |- meta_data_train.csv
                    |- meta_data_val.csv
                |- ut-zap50k-data
                    |- meta-data-bin.csv
                |- ut-zap50k-images-square
                    |- Boots
                    |- Sandals
                    |- Shoes
                    |- Slippers
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

        self.image_dir = os.path.join(self.data_dir, "ut-zap50k-images-square")
        self.train_csv = os.path.join(self.data_dir, "rank_H", "meta_data_train.csv")
        self.val_csv = os.path.join(self.data_dir, "rank_H", "meta_data_val.csv")

    def prepare_data(self):
        pass

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


class UTZapposDataModuleUnlabeled(UTZapposDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=UTZapposDatasetUnlabeled, **kwargs)

    def setup(self, stage: Optional[str] = None):
        super().setup()
        del self.dataset_val


class UTZapposDataModuleCluster(BaseDataModule):
    args_schema = {
        **BaseDataModule.args_schema,
        "data_dir": (str, None, "Data directory."),
        "image_size": (int, 224, "Resize the shorter side of each image to the specified size."),
        "num_views": (int, 1, "Number of views to generate from each image."),
        "cluster_name": (str, "label_gran_10", "Name of the clustering scheme.")
    }

    def __init__(self, data_dir, image_size, num_views, cluster_name, **kwargs):
        """ Cluster labels obtained from https://github.com/Crazy-Jack/Cl-InfoNCE/tree/main/data_processing."""
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_views = num_views
        self.cluster_name = cluster_name
        self.image_dir = os.path.join(data_dir, "ut-zap50k-images-square")
        self.train_csv = os.path.join(data_dir, "rank_H", "meta_data_train.csv")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = UTZapposDatasetCluster(
            image_dir=self.image_dir,
            label_csv=self.train_csv,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, mean=MEAN, std=STD),
            cluster_name=self.cluster_name
        )


class UTZapposDataModuleCCLK(UTZapposDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=UTZapposDatasetCCLK, **kwargs)
        self.train_csv = os.path.join(self.data_dir, "cclk", "meta_data_bin_train.csv")


class UTZapposDataModuleAttribute(UTZapposDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=UTZapposDatasetAttribute, **kwargs)
        self.meta_csv = os.path.join(self.data_dir, "ut-zap50k-data", "meta-data-bin.csv")

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = UTZapposDatasetAttribute(
            image_dir=self.image_dir,
            label_csv=self.train_csv,
            meta_csv=self.meta_csv,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, mean=MEAN, std=STD)
        )
        self.dataset_val = UTZapposDatasetAttribute(
            image_dir=self.image_dir,
            label_csv=self.val_csv,
            meta_csv=self.meta_csv,
            num_views=1,
            transform=ValTransform(crop_size=self.image_size, mean=MEAN, std=STD)
        )
