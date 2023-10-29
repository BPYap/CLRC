import numpy as np
import torch
import torch.nn.functional as F

from clrc.model.base_module import BaseModule
from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder


class CELearner(BaseModule):
    args_schema = {
        **BaseModule.args_schema,
        "backbone": (str, "resnet-50", f"Choose from {AVAILABLE_BACKBONES}."),
        "attribute_dim": (int, 312, "Dimensions of the attribute vector."),
    }

    def __init__(self, backbone, attribute_dim, **kwargs):
        super().__init__(**kwargs)
        hparams = {
            "backbone": backbone,
            "attribute_dim": attribute_dim
        }
        self.save_hyperparameters(hparams)

        self.encoder = get_encoder(backbone, imagenet_init=False, pretrain_path=None, freeze_weights=False)
        self.classifier = torch.nn.Linear(self.encoder.feature_dim, attribute_dim)

    def forward(self, x1, x2):
        pass

    def on_train_start(self):
        self.register_log_metrics({"train/loss": np.nan})

    def training_step(self, batch, batch_idx):
        x, y = batch
        assert isinstance(x, torch.Tensor)
        y = torch.vstack([tensor for group in y for tensor in group]).T.float()

        features = self.encoder(x)
        y_hat = self.classifier(features)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        self.log("train/loss", loss)

        return loss
