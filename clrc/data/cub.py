import csv
import os
from collections import defaultdict
from typing import Optional

import pandas as pd
import torch.utils.data as data
from PIL import Image

from clrc.data.base_data import BaseDataModule
from clrc.transforms import TrainTransform, ValTransform

MEAN = (0.4863, 0.4999, 0.4312)
STD = (0.2070, 0.2018, 0.2428)


# === Dataset === #
class CUBDataset(data.Dataset):
    def __init__(self, image_dir, label_csv, num_views, transform):
        """
        Args:
            image_dir (string): Directory of images.
            label_csv (string): Path to the csv file with class annotations.
            num_views (int): Number of views to generate from each image.
        """
        self.image_dir = image_dir
        self.df = pd.read_csv(label_csv)
        self.num_views = num_views
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], label_tensor
        row = self.df.iloc[index]

        image_path = os.path.join(self.image_dir, row['path'])
        label = int(row['class']) - 1  # make the class label starts from 0 instead of 1

        image = Image.open(image_path).convert('RGB')
        if self.num_views == 1:
            return self.transform(image), label
        else:
            return [self.transform(image) for _ in range(self.num_views)], label


class CUBDatasetUnlabeled(CUBDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v]
        image_tensors, _ = super().__getitem__(index)

        return image_tensors


class CUBDatasetCluster(CUBDataset):
    def __init__(self, cluster_name, **kwargs):
        super().__init__(**kwargs)
        self.cluster_name = cluster_name

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], label_tensor
        image_tensors, _ = super().__getitem__(index)

        row = self.df.iloc[index]
        cluster = row[self.cluster_name]

        return image_tensors, cluster


class CUBDatasetCCLK(CUBDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], [label_tensor_1, ..., label_tensor_k]
        image_tensors, _ = super().__getitem__(index)

        row = self.df.iloc[index]
        attributes = []
        for key in self.df.keys()[4:]:
            attributes.append(row[key])

        return image_tensors, attributes


class CUBDatasetAttribute(CUBDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.grouped_attributes = defaultdict(list)
        for attribute_name in self.df.keys()[2:]:
            group = attribute_name.split("::")[0]
            self.grouped_attributes[group].append(attribute_name)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], [[label_tensor_1_1, ...], ..., [label_tensor_g_1, ...]]
        image_tensors, _ = super().__getitem__(index)

        row = self.df.iloc[index]
        attributes = []
        for group in self.grouped_attributes.values():
            grouped_attributes = []
            for name in group:
                grouped_attributes.append(row[name])
            attributes.append(grouped_attributes)

        return image_tensors, attributes


# === PL DataModule === #
class CUBDataModule(BaseDataModule):
    name = "CUB-200-2011"
    source = "https://www.vision.caltech.edu/datasets/cub_200_2011/"
    args_schema = {
        **BaseDataModule.args_schema,
        "data_dir": (str, None, "Data directory."),
        "image_size": (int, 224, "Resize the shorter side of each image to the specified size."),
        "num_views": (int, 1, "Number of views to generate from each image.")
    }

    def __init__(self, data_dir, image_size, num_views, dataset_cls=CUBDataset, **kwargs):
        """
        Directory structure of `data_dir`:
            `data_dir`
                |- images
                |- attributes
                    |- attributes.txt
                    |- image_attribute_labels.txt
                |- image_class_labels.txt
                |- images.txt
                |- train_test_split.txt
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

        self.image_dir = os.path.join(self.data_dir, "images")
        self.train_csv = os.path.join(self.data_dir, "train.csv")
        self.val_csv = os.path.join(self.data_dir, "val.csv")

    def prepare_data(self):
        if os.path.exists(self.train_csv) and os.path.exists(self.val_csv):
            return

        # extract image classes
        classes = dict()
        with open(os.path.join(self.data_dir, "image_class_labels.txt"), 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                id_, class_ = int(tokens[0]), int(tokens[1])
                classes[id_] = class_

        # extract image paths
        paths = dict()
        with open(os.path.join(self.data_dir, "images.txt"), 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                id_, path = int(tokens[0]), tokens[1]
                paths[id_] = path

        # extract attributes
        attribute_names = []
        with open(os.path.join(self.data_dir, "attributes", "attributes.txt"), 'r') as f:
            for line in f.readlines():
                attribute_names.append(line.split()[-1])
        attributes = defaultdict(dict)
        with open(os.path.join(self.data_dir, "attributes", "image_attribute_labels.txt"), 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                id_, attr_id, label = int(tokens[0]), int(tokens[1]), int(tokens[2])
                attributes[id_][attribute_names[attr_id - 1]] = label

        # extract train-text split
        train_ids = []
        val_ids = []
        with open(os.path.join(self.data_dir, "train_test_split.txt"), 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                id_, is_train = int(tokens[0]), int(tokens[1])
                if is_train:
                    train_ids.append(id_)
                else:
                    val_ids.append(id_)

        # write to csv
        for filename, ids in [(self.train_csv, train_ids), (self.val_csv, val_ids)]:
            with open(filename, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["path", "class"] + attribute_names)
                for id_ in ids:
                    path = paths[id_]
                    class_ = classes[id_]
                    attribs = [attributes[id_][name] for name in attribute_names]
                    csv_writer.writerow([path, class_] + attribs)

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


class CUBDataModuleUnlabeled(CUBDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=CUBDatasetUnlabeled, **kwargs)

    def setup(self, stage: Optional[str] = None):
        super().setup()
        del self.dataset_val


class CUBDataModuleCluster(CUBDataModule):
    args_schema = {
        **CUBDataModule.args_schema,
        "cluster_name": (str, "label_gran_10", "Name of the clustering scheme.")
    }

    def __init__(self, cluster_name, **kwargs):
        """ Cluster labels obtained from https://github.com/Crazy-Jack/Cl-InfoNCE/tree/main/data_processing."""
        super().__init__(dataset_cls=CUBDatasetCluster, **kwargs)
        self.cluster_name = cluster_name
        self.train_csv = os.path.join(self.data_dir, "rank_H", "meta_data_train.csv")

    def setup(self, stage: Optional[str] = None):
        dataset = self.dataset_cls
        self.dataset_train = dataset(
            image_dir=self.image_dir,
            label_csv=self.train_csv,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, mean=MEAN, std=STD),
            cluster_name=self.cluster_name
        )


class CUBDataModuleCCLK(CUBDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=CUBDatasetCCLK, **kwargs)
        self.train_csv = os.path.join(self.data_dir, "cclk", "meta_data_bin_train.csv")


class CUBDataModuleAttribute(CUBDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=CUBDatasetAttribute, **kwargs)
