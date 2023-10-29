import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, include_bn=False):
        super().__init__()
        if num_layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            if include_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if include_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
