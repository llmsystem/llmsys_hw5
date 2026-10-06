"""Small deterministic fixtures: no download, token, or prior-homework solution."""

import torch
from torch import nn


def model():
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def batches(device="cpu"):
    generator = torch.Generator().manual_seed(987)
    return [
        (
            torch.randn(8, 4, generator=generator).to(device),
            torch.randn(8, 2, generator=generator).to(device),
        )
        for _ in range(3)
    ]
