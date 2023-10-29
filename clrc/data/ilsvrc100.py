import csv
import os
from typing import Optional

import pandas as pd
import torch.utils.data as data
from PIL import Image

from clrc.data.base_data import BaseDataModule
from clrc.transforms import TrainTransform, ValTransform

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# === Dataset === #
class ILSVRC100Dataset(data.Dataset):
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

        image_path = os.path.join(self.image_dir, row['filename'])
        label = int(row['class'])

        image = Image.open(image_path).convert('RGB')
        if self.num_views == 1:
            return self.transform(image), label
        else:
            return [self.transform(image) for _ in range(self.num_views)], label


class ILSVRC100DatasetUnlabeled(ILSVRC100Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v]
        image_tensors, _ = super().__getitem__(index)

        return image_tensors


class ILSVRC100DatasetCluster(ILSVRC100Dataset):
    def __init__(self, cluster_name, **kwargs):
        super().__init__(**kwargs)
        self.cluster_name = cluster_name

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], label_tensor
        row = self.df.iloc[index]
        image_path = os.path.join(self.image_dir, row['path'])
        image = Image.open(image_path).convert('RGB')
        if self.num_views == 1:
            image_tensors = self.transform(image)
        else:
            image_tensors = [self.transform(image) for _ in range(self.num_views)]
        cluster = row[self.cluster_name]

        return image_tensors, cluster


class ILSVRC100DatasetAttribute(ILSVRC100Dataset):
    def __init__(self, group_csv, **kwargs):
        super().__init__(**kwargs)

        df = pd.read_csv(group_csv)
        level_1_labels = []
        level_2_labels = []
        for attribute_name in df.keys()[2:]:
            level_1, level_2 = attribute_name.split("-")
            if level_1 not in level_1_labels:
                level_1_labels.append(level_1)
            if level_2 not in level_2_labels:
                level_2_labels.append(level_2)

        self.wind_attributes = dict()
        self.class_to_wind = dict()
        for _, row in df.iterrows():
            wnid = row["wnid"]
            level_1_onehot = [0] * (len(level_1_labels) + 1)
            level_2_onehot = [0] * (len(level_2_labels) + 1)
            for attribute_name in df.keys()[2:]:
                if row[attribute_name] == 1:
                    level_1, level_2 = attribute_name.split("-")
                    level_1_onehot[level_1_labels.index(level_1)] = 1
                    level_2_onehot[level_2_labels.index(level_2)] = 1
                    break
            else:
                level_1_onehot[-1] = 1
                level_2_onehot[-1] = 1
            self.wind_attributes[wnid] = [level_1_onehot, level_2_onehot]
            self.class_to_wind[row['class']] = wnid

    def __getitem__(self, index):
        # return format: [image_tensor_1, ..., image_tensor_v], [[label_tensor_1_1, ...], ..., [label_tensor_g_1, ...]]
        image_tensors, _ = super().__getitem__(index)
        row = self.df.iloc[index]
        wnid = self.class_to_wind[row["class"]]

        return image_tensors, self.wind_attributes[wnid]


# === PL DataModule === #
class ILSVRC100DataModule(BaseDataModule):
    name = "ILSVRC-2012-100"
    source = "https://image-net.org/download.php"
    args_schema = {
        **BaseDataModule.args_schema,
        "data_dir": (str, None, "Data directory."),
        "image_size": (int, 224, "Resize the shorter side of each image to the specified size."),
        "num_views": (int, 1, "Number of views to generate from each image.")
    }

    def __init__(self, data_dir, image_size, num_views, dataset_cls=ILSVRC100Dataset, **kwargs):
        """
        Directory structure of `data_dir`:
            `data_dir`
                |- train
                    |- n01496331
                    |- ...
                |- val
                |- val.csv
                |- wordnet_groups.csv
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

        self.train_image_dir = os.path.join(self.data_dir, "train")
        self.val_image_dir = os.path.join(self.data_dir, "val")

        self.train_csv = os.path.join(self.data_dir, "train.csv")
        self.val_csv = os.path.join(self.data_dir, "val.csv")

        self.group_csv = os.path.join(self.data_dir, "wordnet_groups.csv")

    def prepare_data(self):
        if os.path.exists(self.train_csv):
            return

        wnid_to_class = dict()
        for _, row in pd.read_csv(self.group_csv).iterrows():
            wnid_to_class[row["wnid"]] = row["class"]

        with open(self.train_csv, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["filename", "class"])
            for wnid in os.listdir(self.train_image_dir):
                for img in os.listdir(os.path.join(self.train_image_dir, wnid)):
                    filename = f"{wnid}/{img}"
                    csv_writer.writerow([filename, wnid_to_class[wnid]])

    def setup(self, stage: Optional[str] = None):
        dataset = self.dataset_cls
        self.dataset_train = dataset(
            image_dir=self.train_image_dir,
            label_csv=self.train_csv,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, mean=MEAN, std=STD)
        )
        self.dataset_val = dataset(
            image_dir=self.val_image_dir,
            label_csv=self.val_csv,
            num_views=1,
            transform=ValTransform(crop_size=self.image_size, mean=MEAN, std=STD)
        )


class ILSVRC100DataModuleUnlabeled(ILSVRC100DataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=ILSVRC100DatasetUnlabeled, **kwargs)

    def setup(self, stage: Optional[str] = None):
        super().setup()
        del self.dataset_val


class ILSVRC100DataModuleCluster(ILSVRC100DataModule):
    args_schema = {
        **ILSVRC100DataModule.args_schema,
        "cluster_name": (str, "label_gran_level_4", "Name of the clustering scheme.")
    }

    def __init__(self, cluster_name, **kwargs):
        """ Cluster labels obtained from https://github.com/Crazy-Jack/Cl-InfoNCE/tree/main/data_processing."""
        super().__init__(dataset_cls=ILSVRC100DatasetCluster, **kwargs)
        self.cluster_name = cluster_name
        self.train_csv = os.path.join(self.data_dir, "meta_data_train.csv")

    def setup(self, stage: Optional[str] = None):
        dataset = self.dataset_cls
        self.dataset_train = dataset(
            image_dir=os.path.join(self.data_dir, "train"),
            label_csv=self.train_csv,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, mean=MEAN, std=STD),
            cluster_name=self.cluster_name
        )


class ILSVRC100DataModuleAttribute(ILSVRC100DataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_cls=None, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = ILSVRC100DatasetAttribute(
            image_dir=self.train_image_dir,
            label_csv=self.train_csv,
            num_views=self.num_views,
            transform=TrainTransform(crop_size=self.image_size, mean=MEAN, std=STD),
            group_csv=self.group_csv
        )
        self.dataset_val = ILSVRC100DatasetAttribute(
            image_dir=self.val_image_dir,
            label_csv=self.val_csv,
            num_views=1,
            transform=ValTransform(crop_size=self.image_size, mean=MEAN, std=STD),
            group_csv=self.group_csv
        )
