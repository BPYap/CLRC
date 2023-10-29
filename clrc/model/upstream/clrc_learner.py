import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from SupContrast.losses import SupConLoss
from clrc.model.base_module import BaseModule
from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder
from clrc.utils import calculate_train_steps


class CLRCLearner(BaseModule):
    args_schema = {
        **BaseModule.args_schema,
        "backbone": (
            str,
            "resnet-50",
            f"Choose from {AVAILABLE_BACKBONES}."
        ),
        "num_groups": (
            int,
            -1,
            "Number of attribute groups."
        ),
        "mlp_hidden_dim": (
            int,
            1024,
            "Hidden dimensions of the MLP."
        ),
        "group_dim": (
            int,
            64,
            "Dimensions of each grouped feature vector."
        ),
        "temperature": (
            float,
            0.07,
            "Temperature parameter for the contrastive loss function."
        ),
        "temp_per_group": (
            str,
            "",
            "Temperature scaling for each group."
        ),
        "base_ema": (
            float,
            0.996,
            "Exponential moving average parameter for the target network."
        )
    }

    def __init__(
            self, backbone, num_groups, mlp_hidden_dim, group_dim,
            temperature, temp_per_group, base_ema, **kwargs
    ):
        super().__init__(**kwargs)
        hparams = {
            "backbone": backbone,
            "num_groups": num_groups,
            "mlp_hidden_dim": mlp_hidden_dim,
            "group_dim": group_dim,
            "temperature": temperature,
            "temp_per_group": temp_per_group,
            "base_ema": base_ema
        }
        self.save_hyperparameters(hparams)

        self.encoder = get_encoder(
            backbone,
            imagenet_init=False,
            pretrain_path=None,
            freeze_weights=False
        )
        self.linear_1 = torch.nn.Linear(
            self.encoder.feature_dim,
            mlp_hidden_dim,
            bias=True
        )
        self.bn_1 = torch.nn.BatchNorm1d(mlp_hidden_dim)
        self.linear_2 = torch.nn.Linear(
            mlp_hidden_dim,
            num_groups * group_dim,
            bias=True
        )
        self.attribute_projectors = torch.nn.ModuleList([
            torch.nn.Linear(
                group_dim,
                group_dim // 2,
                bias=True
            ) for _ in range(num_groups)
        ])

        self.target_encoder = deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)

        self.temperature = temperature
        self.temp_scalings = [1] * num_groups if temp_per_group == "" else \
            [float(f) for f in temp_per_group.split(",")]
        normalization_factor = sum(self.temp_scalings)
        self.temp_scalings = [s / normalization_factor for s in self.temp_scalings]
        self.base_ema = 0.996
        self.total_steps = ...
        self.num_groups = num_groups

        self.supcon_loss = SupConLoss(
            temperature=temperature,
            base_temperature=temperature
        )

    @staticmethod
    def _collapse_attributes(grouped_attributes):
        labels = torch.vstack(grouped_attributes).T
        batch_size = labels.shape[0]
        temp_1 = labels.reshape(batch_size, 1, -1)
        temp_2 = labels.reshape(1, batch_size, -1).tile(batch_size, 1, 1)
        mask = ((temp_1 & temp_2).sum(dim=2) > 0).int()

        return mask

    def _cl_loss(self, projs, mask, group_id):
        batch_size = projs.shape[0]
        num_pos = mask.sum()
        if num_pos <= batch_size or num_pos == np.product(mask.shape):
            return 0
        projs = projs.reshape(batch_size, 1, -1)

        return self.temp_scalings[group_id] * self.supcon_loss(projs, mask=mask)

    def _reconstruction_loss(self, group_feats, targets):
        reconstructions = torch.relu_(
            self.bn_1(
                torch.nn.functional.linear(
                    torch.cat(group_feats, dim=1) - self.linear_2.bias,
                    self.linear_2.weight.T
                )
            )
        )
        reconstructions = torch.nn.functional.linear(
            reconstructions - self.linear_1.bias,
            self.linear_1.weight.T
        )

        return F.mse_loss(torch.sigmoid(reconstructions), torch.sigmoid(targets))

    def on_train_start(self):
        self.register_log_metrics({"train/loss": np.nan})

        _, self.total_steps = calculate_train_steps(self)

    def training_step(self, batch, batch_idx):
        x, y = batch
        v1, v2 = x
        num_attribute_group = len(y)

        features = self.encoder(torch.cat([v1, v2]))
        w = torch.relu_(self.bn_1(self.linear_1(features)))
        group_feats = torch.chunk(
            self.linear_2(w), num_attribute_group, dim=1
        )
        group_projs = [
            F.normalize(proj(feats))
            for feats, proj in zip(group_feats, self.attribute_projectors)
        ]

        # split into two views
        group_feats = [torch.chunk(group, 2) for group in group_feats]
        group_feats_v1 = [chunks[0] for chunks in group_feats]
        group_feats_v2 = [chunks[1] for chunks in group_feats]
        group_projs = [torch.chunk(group, 2) for group in group_projs]
        group_projs_v1 = [chunks[0] for chunks in group_projs]
        group_projs_v2 = [chunks[1] for chunks in group_projs]

        gcl_loss = 0
        for group_id, group in enumerate(y):
            mask = self._collapse_attributes(group).to(self.device)
            gcl_loss += self._cl_loss(
                group_projs_v1[group_id], mask, group_id
            )
            gcl_loss += self._cl_loss(
                group_projs_v2[group_id], mask, group_id
            )
        gcl_loss *= 0.5

        with torch.no_grad():
            targets = self.target_encoder(torch.cat([v1, v2]))
            targets_v1, targets_v2 = torch.chunk(targets, 2)
        reconstruction_loss = 0.5 * (
                self._reconstruction_loss(group_feats_v1, targets_v2) +
                self._reconstruction_loss(group_feats_v2, targets_v1)
        )

        loss = gcl_loss + reconstruction_loss

        self.log("tran/gcl_loss", gcl_loss)
        self.log("train/reconstruction_loss", reconstruction_loss)
        self.log("train/loss", loss)

        return loss

    def _get_ema(self):
        current_step = self.global_step

        return 1 - (1 - self.base_ema) * \
               (math.cos((current_step * math.pi) / self.total_steps) + 1) / 2

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        ema = self._get_ema()
        self.log("ema", ema, rank_zero_only=True)

        # update feature extractor of target network
        parameters = zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        )
        for online_param, target_param in parameters:
            target_param.data = ema * target_param.data + \
                                (1 - ema) * online_param.data
