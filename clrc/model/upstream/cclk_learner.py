# Codes adapted from https://github.com/Crazy-Jack/CCLK-release
import numpy as np
import torch

from clrc.model.base_module import BaseModule
from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w1 = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(input_dim)
        self.relu = torch.nn.ReLU()
        self.w2 = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(x)))))


class CCLKLearner(BaseModule):
    args_schema = {
        **BaseModule.args_schema,
        "backbone": (
            str,
            "resnet-50",
            f"Choose from {AVAILABLE_BACKBONES}."
        ),
        "mlp_output_dim": (
            int,
            128,
            "Output dimensions of the MLP projector."
        ),
        "temperature": (
            float,
            0.5,
            "Temperature parameter for the contrastive loss function."
        )
    }

    def __init__(
            self, backbone, mlp_output_dim,
            temperature, **kwargs
    ):
        super().__init__(**kwargs)
        hparams = {
            "backbone": backbone,
            "mlp_output_dim": mlp_output_dim,
            "temperature": temperature
        }
        self.save_hyperparameters(hparams)

        self.encoder = get_encoder(
            backbone,
            imagenet_init=False,
            pretrain_path=None,
            freeze_weights=False
        )
        self.projector = MLP(
            self.encoder.feature_dim,
            mlp_output_dim,
        )

        self.temperature = temperature
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)

    def on_train_start(self):
        self.register_log_metrics({"train/loss": np.nan})

    def _compute_scores(self, v1, v2):
        z1, z2 = torch.chunk(self.projector(torch.cat([v1, v2])), 2)

        temp = self.temperature
        sim11 = self.cos_sim(z1.unsqueeze(-2), z1.unsqueeze(-3)) / temp
        sim22 = self.cos_sim(z2.unsqueeze(-2), z2.unsqueeze(-3)) / temp
        sim12 = self.cos_sim(z1.unsqueeze(-2), z2.unsqueeze(-3)) / temp

        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)

        return raw_scores

    @staticmethod
    def _compute_weight(K_Z, lambda_=0.01, clip_threshold=1e-6):
        n = K_Z.shape[0]
        device = K_Z.device
        with torch.no_grad():
            K_Z = K_Z.detach()
            inverse = torch.inverse(K_Z + lambda_ * torch.eye(n).to(device))

        weight = torch.matmul(inverse, K_Z)
        weight[range(n), range(n)] = 0.
        weight[weight < clip_threshold] = 0.

        return weight

    def _compute_loss(self, scores, conditions):
        n = int(scores.shape[0] / 2)
        Kxy = torch.exp(scores[:n, :n])
        Kxx = torch.exp(scores[:n, n:])
        Kyy = torch.exp(scores[n:, :n])
        Kyx = torch.exp(scores[n:, n:])

        distance = self.cos_sim(
            conditions.unsqueeze(-2), conditions.unsqueeze(-3)
        )
        n = distance.shape[0]
        distance[range(n), range(n)] = 1.

        weight = self._compute_weight(distance)
        Mxy = torch.matmul(Kxy, weight)
        Mxx = torch.matmul(Kxx, weight)
        Myx = torch.matmul(Kyx, weight)
        Myy = torch.matmul(Kyy, weight)

        pos = torch.clamp(
            torch.diagonal(Mxx, 0) + torch.diagonal(Mxy, 0),
            1e-7,
            1e+20
        )
        deno = torch.clamp(
            pos + Kxy.sum(1) + Kxx.sum(1),
            1e-7,
            1e+20
        )
        log_negatives = torch.log(deno)
        loss_x = - (torch.log(pos) - log_negatives).mean()

        pos = torch.clamp(
            torch.diagonal(Myy, 0) + torch.diagonal(Myx, 0),
            1e-7,
            1e+20
        )
        deno = torch.clamp(
            pos + Kyx.sum(1) + Kyy.sum(1),
            1e-7,
            1e+20
        )
        log_negatives = torch.log(deno)
        loss_y = - (torch.log(pos) - log_negatives).mean()

        return (loss_x + loss_y) / 2

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.cat(x)
        y = torch.vstack(y).T.float()

        features = self.encoder(x)
        v1, v2 = torch.chunk(features, 2)
        raw_scores = self._compute_scores(v1, v2)
        loss = self._compute_loss(raw_scores, y)
        self.log("train/loss", loss)

        return loss
