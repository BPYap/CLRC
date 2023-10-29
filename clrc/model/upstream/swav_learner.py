import numpy as np
import torch
import torch.nn.functional as F

from clrc.model.base_module import BaseModule
from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder
from clrc.model.mlp import MLP
from clrc.utils import calculate_train_steps


class SwAVLearner(BaseModule):
    args_schema = {
        **BaseModule.args_schema,
        "backbone": (str, "resnet-50", f"Choose from {AVAILABLE_BACKBONES}."),
        "imagenet_init": (bool, False, "Initialize with weights (if available) pretrained on ILSVRC-2012."),
        "mlp_layers": (int, 2, "Number of layers in the MLP projector."),
        "mlp_hidden_dim": (int, 2048, "Hidden dimensions of the MLP projector."),
        "mlp_output_dim": (int, 128, "Output dimensions of the MLP projector."),
        "num_prototypes": (int, 3000, "Number of prototype vectors"),
        "temperature": (float, 0.1, "Temperature parameter for score scaling."),
    }

    def __init__(self, backbone, imagenet_init, mlp_layers, mlp_hidden_dim, mlp_output_dim,
                 num_prototypes, temperature, **kwargs):
        super().__init__(**kwargs)
        hparams = {
            "backbone": backbone,
            "imagenet_init": imagenet_init,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_output_dim": mlp_output_dim,
            "num_prototypes": num_prototypes,
            "temperature": temperature
        }
        self.save_hyperparameters(hparams)

        self.encoder = get_encoder(backbone, imagenet_init, pretrain_path=None, freeze_weights=False)
        self.projector = MLP(mlp_layers, self.encoder.feature_dim, mlp_hidden_dim, mlp_output_dim)
        self.prototypes = torch.nn.Linear(mlp_output_dim, num_prototypes, bias=False)

        self.temperature = temperature

        self.steps_per_epoch = ...

    def on_train_start(self):
        self.register_log_metrics({"train/loss": np.nan})
        self.steps_per_epoch, _ = calculate_train_steps(self)

    def forward(self, x):
        features = self.encoder(x)
        projections = F.normalize(self.projector(features))

        return projections

    @staticmethod
    def _sinkhorn(scores, eps=0.05, niters=3):
        Q = torch.exp(scores / eps).T
        Q /= Q.sum()
        K, B = Q.shape
        device = scores.device
        r = torch.ones(K).to(device) / K
        c = torch.ones(B).to(device) / B
        for _ in range(niters):
            u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).T

    def training_step(self, batch, batch_idx):
        if self.global_step < self.steps_per_epoch:
            self.prototypes.requires_grad_(False)
        else:
            self.prototypes.requires_grad_(True)

        x = batch
        x_t, x_s = x

        z = self(torch.cat([x_t, x_s]))

        scores = self.prototypes(z)
        score_t, score_s = torch.chunk(scores, 2)

        # compute assignments
        with torch.no_grad():
            q_t = self._sinkhorn(score_t)
            q_s = self._sinkhorn(score_s)

        # convert scores to probabilities
        p_t = F.softmax(score_t / self.temperature, dim=-1)
        p_s = F.softmax(score_s / self.temperature, dim=-1)

        # swap prediction problem
        loss = - 0.5 * (q_t * torch.log(p_s) + q_s * torch.log(p_t)).mean()
        self.log("train/loss", loss)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # normalize prototypes
        if self.global_step >= self.steps_per_epoch:
            with torch.no_grad():
                weight_matrix = self.prototypes.weight.data
                self.prototypes.weight.data = F.normalize(weight_matrix, dim=-1)
