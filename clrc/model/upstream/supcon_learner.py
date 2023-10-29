import numpy as np
import torch
import torch.nn.functional as F

from SupContrast.losses import SupConLoss
from clrc.model.base_module import BaseModule
from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder
from clrc.model.mlp import MLP
from clrc.utils import gather_tensors


class SupConLearner(BaseModule):
    args_schema = {
        **BaseModule.args_schema,
        "backbone": (str, "resnet-50", f"Choose from {AVAILABLE_BACKBONES}."),
        "mlp_layers": (int, 2, "Number of layers in the MLP projector."),
        "mlp_hidden_dim": (int, 128, "Hidden dimensions of the MLP projector."),
        "mlp_output_dim": (int, 256, "Output dimensions of the MLP projector."),
        "temperature": (float, 0.07, "Temperature parameter for the loss function.")
    }

    def __init__(self, backbone, mlp_layers, mlp_hidden_dim, mlp_output_dim, temperature, **kwargs):
        super().__init__(**kwargs)
        hparams = {
            "backbone": backbone,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_output_dim": mlp_output_dim,
            "temperature": temperature
        }
        self.save_hyperparameters(hparams)

        self.encoder = get_encoder(backbone, imagenet_init=False, pretrain_path=None, freeze_weights=False)
        self.projector = MLP(mlp_layers, self.encoder.feature_dim, mlp_hidden_dim, mlp_output_dim)

        self.supcon_loss = SupConLoss(temperature=temperature, base_temperature=temperature)

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)

        return F.normalize(projections)

    def on_train_start(self):
        self.register_log_metrics({"train/loss": np.nan})

    def training_step(self, batch, batch_idx):
        x, y = batch

        batch_size = x[0].shape[0]
        num_views = len(x)

        projections = self(torch.cat(x))
        projections = torch.cat(torch.chunk(projections, num_views), dim=1).reshape(batch_size, num_views, -1)
        projections = gather_tensors(projections)
        y = gather_tensors(y)
        loss = self.supcon_loss(projections, labels=y)

        self.log("train/loss", loss)

        return loss
