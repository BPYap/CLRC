import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from clrc.model.base_module import BaseModule
from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder
from clrc.model.mlp import MLP
from clrc.utils import calculate_train_steps


class BYOLLearner(BaseModule):
    args_schema = {
        **BaseModule.args_schema,
        "backbone": (str, "resnet-50", f"Choose from {AVAILABLE_BACKBONES}."),
        "imagenet_init": (bool, False, "Initialize with weights (if available) pretrained on ILSVRC-2012."),
        "mlp_layers": (int, 2, "Number of layers in the MLP projector."),
        "mlp_hidden_dim": (int, 128, "Hidden dimensions of the MLP projector."),
        "mlp_output_dim": (int, 256, "Output dimensions of the MLP projector."),
        "base_ema": (float, 0.996, "Exponential moving average parameter for the target network.")
    }

    def __init__(self, backbone, imagenet_init, mlp_layers, mlp_hidden_dim, mlp_output_dim, base_ema, **kwargs):
        super().__init__(**kwargs)
        hparams = {
            "backbone": backbone,
            "imagenet_init": imagenet_init,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_output_dim": mlp_output_dim,
            "base_ema": base_ema
        }
        self.save_hyperparameters(hparams)

        # online network
        self.encoder = get_encoder(backbone, imagenet_init, pretrain_path=None, freeze_weights=False)
        self.projector = MLP(mlp_layers, self.encoder.feature_dim, mlp_hidden_dim, mlp_output_dim, include_bn=True)
        self.predictor = MLP(mlp_layers, mlp_output_dim, mlp_hidden_dim, mlp_output_dim, include_bn=True)

        # target network
        self.target_encoder = deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        self.target_projector = deepcopy(self.projector)
        self.target_projector.requires_grad_(False)

        self.base_ema = base_ema

        self.total_steps = ...

    def forward_online_network(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        predictions = self.predictor(projections)

        return predictions

    def forward_target_network(self, x):
        with torch.no_grad():
            features = self.target_encoder(x)
            targets = self.target_projector(features)

        return targets

    def on_train_start(self):
        self.register_log_metrics({"train/loss": np.nan})
        _, self.total_steps = calculate_train_steps(self)

    def training_step(self, batch, batch_idx):
        x = batch

        v1, v2 = x
        v1_targets, v2_targets = torch.chunk(self.forward_target_network(torch.cat([v1, v2])), 2)
        v1_preds, v2_preds = torch.chunk(self.forward_online_network(torch.cat([v1, v2])), 2)

        loss_1 = F.mse_loss(F.normalize(v2_preds), F.normalize(v1_targets))
        loss_2 = F.mse_loss(F.normalize(v1_preds), F.normalize(v2_targets))
        loss = loss_1 + loss_2

        self.log("train/loss", loss)

        return loss

    def _get_ema(self):
        current_step = self.global_step

        return 1 - (1 - self.base_ema) * (math.cos((current_step * math.pi) / self.total_steps) + 1) / 2

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        ema = self._get_ema()
        self.log("ema", ema, rank_zero_only=True)

        # update feature extractor of target network
        for online_param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = ema * target_param.data + (1 - ema) * online_param.data

        # update projector of target network
        for online_param, target_param in zip(self.projector.parameters(), self.target_projector.parameters()):
            target_param.data = ema * target_param.data + (1 - ema) * online_param.data
