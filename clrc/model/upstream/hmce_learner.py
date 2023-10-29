import math

import numpy as np
import torch
import torch.nn.functional as F

from clrc.model.base_module import BaseModule
from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder
from clrc.model.mlp import MLP


class SupConLoss(torch.nn.Module):
    """A modified version of `SupContrast.losses`"""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, higher_level_losses=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        losses = loss.view(anchor_count, batch_size)
        if higher_level_losses is not None:
            losses = torch.max(losses, higher_level_losses)
        loss = losses.sum() / torch.nonzero(mask.sum(1)).shape[0]

        return losses, loss


class HMCELearner(BaseModule):
    args_schema = {
        **BaseModule.args_schema,
        "backbone": (
            str,
            "resnet-50",
            f"Choose from {AVAILABLE_BACKBONES}."
        ),
        "mlp_layers": (
            int,
            2,
            "Number of layers in the MLP projector."
        ),
        "mlp_hidden_dim": (
            int,
            128,
            "Hidden dimensions of the MLP projector."
        ),
        "mlp_output_dim": (
            int,
            256,
            "Output dimensions of the MLP projector."
        ),
        "temperature": (
            float,
            0.07,
            "Temperature parameter for the contrastive loss function."
        )
    }

    def __init__(
            self, backbone, mlp_layers, mlp_hidden_dim, mlp_output_dim,
            temperature, **kwargs
    ):
        super().__init__(**kwargs)
        hparams = {
            "backbone": backbone,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_output_dim": mlp_output_dim,
            "temperature": temperature,
        }
        self.save_hyperparameters(hparams)

        self.encoder = get_encoder(
            backbone,
            imagenet_init=False,
            pretrain_path=None,
            freeze_weights=False
        )
        self.projector = MLP(
            mlp_layers, self.encoder.feature_dim, mlp_hidden_dim, mlp_output_dim
        )

        self.supcon_loss = SupConLoss(
            temperature=temperature, base_temperature=temperature
        )

    def on_train_start(self):
        self.register_log_metrics({"train/loss": np.nan})

    @staticmethod
    def _collapse_labels(labels):
        one_hots = torch.vstack(labels).T.float()

        return torch.nonzero(one_hots, as_tuple=True)[1]

    def training_step(self, batch, batch_idx):
        x, y = batch
        v1, v2 = x

        features = self.encoder(torch.cat([v1, v2]))
        projections = F.normalize(self.projector(features))
        projections = torch.cat(
            torch.chunk(projections, 2), dim=1
        ).reshape(v1.shape[0], 2, -1)

        level_1_labels = self._collapse_labels(y[0])
        level_2_labels = self._collapse_labels(y[1])
        level_2_losses, level_2_loss = self.supcon_loss(
            projections, labels=level_2_labels
        )
        _, level_1_loss = self.supcon_loss(
            projections, labels=level_1_labels,
            higher_level_losses=level_2_losses
        )
        loss = 0.5 * (math.exp(0.5) * level_1_loss + math.exp(1) * level_2_loss)

        self.log("train/level_1_loss", level_1_loss)
        self.log("train/level_2_loss", level_2_loss)
        self.log("train/loss", loss)

        return loss
