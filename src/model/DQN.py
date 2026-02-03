import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class DQNNet(nn.Module):
    def __init__(
        self,
        nS: int,
        nA: int,
        hidden: int = 128,
        layernorm: bool = False,
        weightnorm: bool = False,
    ):
        super().__init__()

        def linear(in_dim, out_dim):
            layer = nn.Linear(in_dim, out_dim)
            if weightnorm:
                layer = weight_norm(layer)
            return layer

        def block(in_dim, out_dim):
            layers = [linear(in_dim, out_dim)]
            if layernorm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            return layers

        self.net = nn.Sequential(
            *block(nS, hidden),
            *block(hidden, hidden),
            linear(hidden, nA),
        )

    def forward(self, x):
        return self.net(x)
