import csv
import os
from typing import Optional

import torch
from torch.nn import functional as F
from torchmetrics import Accuracy, F1, AUROC

from clrc.model.base_module import BaseModule
from clrc.model.downstream.base_model import BaseModel


class Classifier(BaseModule):
    args_schema = {
        **BaseModel.args_schema,
        **BaseModule.args_schema,
        "output_dim": (int, 2, "Output dimensions (number of classes)."),
        "metrics": (str, "top-1,top-5", "Evaluation metrics. Choose from {'top-1', 'top-5', 'f1', 'auc-roc'}."),
        "log_predictions": (bool, False, "Log test predictions to a .csv file")
    }

    def __init__(self, output_dim, metrics, log_predictions, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if k in BaseModule.args_schema})
        hparams = {
            "backbone": kwargs['backbone'],
            "imagenet_init": kwargs['imagenet_init'],
            "backbone_weights": kwargs['backbone_weights'],
            "freeze_backbone": kwargs['freeze_backbone']
        }
        self.save_hyperparameters(hparams)

        self.model = BaseModel(**{k: v for k, v in kwargs.items() if k in BaseModel.args_schema})
        self.output = torch.nn.Linear(self.model.encoder.feature_dim, output_dim)

        self.metrics_names = metrics.split(",")
        self.metrics = torch.nn.ModuleDict()
        for prefix in ["train", "val", "test"]:
            for metrics_name in self.metrics_names:
                if metrics_name == 'top-1':
                    self.metrics[f"{prefix}_top-1"] = Accuracy(num_classes=output_dim, average='micro')
                elif metrics_name == 'top-5':
                    self.metrics[f"{prefix}_top-5"] = Accuracy(num_classes=output_dim, top_k=5, average='micro')
                elif metrics_name == 'f1':
                    self.metrics[f"{prefix}_f1"] = F1(num_classes=output_dim, average='macro')
                elif metrics_name == 'auc-roc':
                    self.metrics[f"{prefix}_auc-roc"] = AUROC(num_classes=output_dim, average='macro')
                else:
                    raise ValueError(f"Unkown evaluation metrics: '{metrics_name}'.")

        self.log_predictions = log_predictions
        self.predictions = []

    def forward(self, x):
        logits = self.output(x)

        return logits

    def on_train_start(self):
        initial_metrics = {f"val/{name}": 0 for name in self.metrics_names}
        self.register_log_metrics(initial_metrics)

    def update_metrics(self, prefix, y_hat, y):
        for name in self.metrics_names:
            if name == 'auc-roc':
                self.metrics[f"{prefix}_auc-roc"].update(F.softmax(y_hat, dim=1), y)
            else:
                self.metrics[f"{prefix}_{name}"].update(y_hat, y)

        if prefix == "test" and self.log_predictions:
            for pred, gt in zip(F.softmax(y_hat, dim=1), y):
                pred = [round(p.item(), 2) for p in pred]
                self.predictions.append((pred, gt.item()))

    def _shared_step(self, batch, prefix):
        x, y = batch

        features = self.model.encoder(x)
        y_hat = self(features)
        loss = F.cross_entropy(y_hat, y)

        self.log(f"{prefix}/loss", loss)
        self.update_metrics(prefix, y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def _shared_on_epoch_end(self, prefix):
        for name in self.metrics_names:
            key = f"{prefix}_{name}"
            score = self.metrics[key].compute()
            self.metrics[key].reset()
            self.log(f"{prefix}/{name}", score, rank_zero_only=True)

    def on_train_epoch_end(self, unused: Optional = None):
        self._shared_on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end("val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end("test")

        if self.log_predictions:
            csv_path = os.path.join(
                self.trainer.logger.log_dir,
                "predictions.csv"
            )
            with open(csv_path, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["y_hat", "y"])
                csv_writer.writerows(self.predictions)
