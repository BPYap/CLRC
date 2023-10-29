import pytorch_lightning as pl


class BaseEncoder(pl.LightningModule):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        raise NotImplementedError
