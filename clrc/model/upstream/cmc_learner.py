import numpy as np
import torch
import torch.nn.functional as F

from SupContrast.losses import SupConLoss
from clrc.model.base_module import BaseModule
from clrc.model.encoder import AVAILABLE_BACKBONES, get_encoder
from clrc.model.mlp import MLP
from clrc.utils import gather_tensors


class CMCLearner(BaseModule):
    args_schema = {
        **BaseModule.args_schema,
        "backbone": (str, "resnet-50", f"Choose from {AVAILABLE_BACKBONES}."),
        "attribute_dim": (int, 312, "Dimensions of the attribute vector."),
        "attribute_mlp_layers": (int, 2, "Number of layers in the MLP projector for the attribute vectors."),
        "attribute_hidden_dim": (int, 256, "Hidden dimensions of the MLP projector for the attribute vectors"),
        "mlp_layers": (int, 2, "Number of layers in the MLP projector for the image feature vectors."),
        "mlp_hidden_dim": (int, 2048, "Hidden dimensions of the MLP projector for the image feature vectors."),
        "mlp_output_dim": (int, 128, "Output dimensions of the MLP projectors."),
        "temperature": (float, 0.07, "Temperature parameter for the loss function.")
    }

    def __init__(self, backbone, attribute_dim, attribute_mlp_layers, attribute_hidden_dim, mlp_layers,
                 mlp_hidden_dim, mlp_output_dim, temperature, **kwargs):
        super().__init__(**kwargs)
        hparams = {
            "backbone": backbone,
            "attribute_dim": attribute_dim,
            "attribute_mlp_layers": attribute_mlp_layers,
            "attribute_hidden_dim": attribute_hidden_dim,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_output_dim": mlp_output_dim,
            "temperature": temperature
        }
        self.save_hyperparameters(hparams)

        self.encoder = get_encoder(backbone, imagenet_init=False, pretrain_path=None, freeze_weights=False)
        self.attribute_projector = MLP(attribute_mlp_layers, attribute_dim, attribute_hidden_dim, mlp_output_dim)
        self.projector = MLP(mlp_layers, self.encoder.feature_dim, mlp_hidden_dim, mlp_output_dim)

        self.supcon_loss = SupConLoss(temperature=temperature, base_temperature=temperature)

    def forward(self, x1, x2):
        image_features = self.encoder(x1)
        image_projections = self.projector(image_features)
        attribute_projections = self.attribute_projector(x2)

        return F.normalize(image_projections), F.normalize(attribute_projections)

    def on_train_start(self):
        self.register_log_metrics({"train/loss": np.nan})

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        assert isinstance(x1, torch.Tensor)
        x2 = torch.vstack([tensor for group in x2 for tensor in group]).T.float()

        batch_size = x1.shape[0]
        num_views = 2

        image_projections, attribute_projections = self(x1, x2)
        projections = torch.cat([image_projections, attribute_projections], dim=1).reshape(batch_size, num_views, -1)
        projections = gather_tensors(projections)
        loss = self.supcon_loss(projections)

        self.log("train/loss", loss)

        return loss
