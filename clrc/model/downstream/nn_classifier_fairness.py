from typing import Optional

import torch
from torch.nn import functional as F
from torchmetrics import Accuracy, AUROC

from clrc.model.base_module import BaseModule
from clrc.model.downstream.base_model import BaseModel


class ClassifierFairness(BaseModule):
    args_schema = {
        **BaseModel.args_schema,
        **BaseModule.args_schema,
        "num_groups": (int, 2, "Number of sensitive groups."),
    }

    def __init__(self, num_groups, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if k in BaseModule.args_schema})
        hparams = {
            "backbone": kwargs['backbone'],
            "imagenet_init": kwargs['imagenet_init'],
            "backbone_weights": kwargs['backbone_weights'],
            "freeze_backbone": kwargs['freeze_backbone']
        }
        self.save_hyperparameters(hparams)

        self.model = BaseModel(**{k: v for k, v in kwargs.items() if k in BaseModel.args_schema})
        self.output = torch.nn.Linear(self.model.encoder.feature_dim, 1)

        self.metrics = torch.nn.ModuleDict()
        self.num_groups = num_groups
        for prefix in ["train", "val", "test"]:
            for name in ["top-1", "auc-roc"]:
                for group in range(num_groups):
                    if name == "top-1":
                        metric = Accuracy()
                    else:
                        metric = AUROC(pos_label=1)
                    self.metrics[f"{prefix}_{group}_{name}"] = metric

            self.metrics[f"{prefix}_overall_top-1"] = Accuracy()
            self.metrics[f"{prefix}_overall_auc-roc"] = AUROC(pos_label=1)

    def forward(self, x):
        logits = self.output(x)

        return logits.view(-1)

    def on_train_start(self):
        initial_metrics = {}
        for name in ["top-1", "auc-roc"]:
            for group in range(self.num_groups):
                initial_metrics[f"val/{group}_{name}"] = 0
        initial_metrics[f"val/overall_top-1"] = 0
        initial_metrics[f"val/overall_auc-roc"] = 0
        initial_metrics[f"val/worst-case_auc-roc"] = 0
        initial_metrics[f"val/auc-roc_gap"] = 0
        self.register_log_metrics(initial_metrics)

    def update_metrics(self, prefix, y_hat, y, groups):
        for group in range(self.num_groups):
            indices = torch.nonzero(groups == group, as_tuple=True)[-1]
            if len(indices) > 0:
                preds = y_hat[indices]
                gts = y[indices]
                self.metrics[f"{prefix}_{group}_top-1"].update(preds, gts)
                self.metrics[f"{prefix}_{group}_auc-roc"].update(torch.sigmoid(preds), gts)

        self.metrics[f"{prefix}_overall_top-1"].update(y_hat, y)
        self.metrics[f"{prefix}_overall_auc-roc"].update(torch.sigmoid(y_hat), y)

    def _shared_step(self, batch, prefix):
        x, (y, groups) = batch

        features = self.model.encoder(x)
        y_hat = self(features)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())

        self.log(f"{prefix}/loss", loss)
        self.update_metrics(prefix, y_hat, y, groups)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def _shared_on_epoch_end(self, prefix):
        worst_case_auc = float('inf')
        best_case_auc = float('-inf')
        for name in ["top-1", "auc-roc"]:
            for group in range(self.num_groups):
                score = self.metrics[f"{prefix}_{group}_{name}"].compute()
                self.metrics[f"{prefix}_{group}_{name}"].reset()

                if name == "auc-roc":
                    worst_case_auc = min(worst_case_auc, score)
                    best_case_auc = max(best_case_auc, score)

                self.log(f"{prefix}/{group}_{name}", score, rank_zero_only=True)

        top_1 = self.metrics[f"{prefix}_overall_top-1"].compute()
        self.metrics[f"{prefix}_overall_top-1"].reset()
        auc = self.metrics[f"{prefix}_overall_auc-roc"].compute()
        self.metrics[f"{prefix}_overall_auc-roc"].reset()

        self.log(f"{prefix}/overall_top-1", top_1, rank_zero_only=True)
        self.log(f"{prefix}/overall_auc-roc", auc, rank_zero_only=True)

        self.log(f"{prefix}/worst-case_auc-roc", worst_case_auc, rank_zero_only=True)
        self.log(f"{prefix}/auc-roc_gap", best_case_auc - worst_case_auc, rank_zero_only=True)

    def on_train_epoch_end(self, unused: Optional = None):
        self._shared_on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end("val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end("test")
