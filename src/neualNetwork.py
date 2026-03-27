import torch
import torch.nn as nn
import numpy as np


def _make_conv(in_channels):
    """3-layer CNN: extracts spatial features from 84x84 frames."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # 84→20
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20→9
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9→7
        nn.ReLU(),
    )


def _conv_out_size(conv, input_shape):
    with torch.no_grad():
        o = conv(torch.zeros(1, *input_shape))
    return int(np.prod(o.size()))


class ActorNet(nn.Module):
    """Policy network: outputs logits over actions (which action to take)."""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = _make_conv(input_shape[0])
        self.fc = nn.Sequential(
            nn.Linear(_conv_out_size(self.conv, input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),  # raw logits, Categorical handles softmax
        )

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), start_dim=1))


class CriticNet(nn.Module):
    """Value network: estimates how good the current state is (scalar)."""
    def __init__(self, input_shape):
        super().__init__()
        self.conv = _make_conv(input_shape[0])
        self.fc = nn.Sequential(
            nn.Linear(_conv_out_size(self.conv, input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), start_dim=1)).squeeze(-1)  # (batch,)
