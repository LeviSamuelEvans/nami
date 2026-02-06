from .base import VectorField
import torch
from torch import nn

class VelocityField(VectorField):  # inherit from VectorField instead of nn.Module
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, dim),
        )

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, x, t, c=None):
        t_exp = t.unsqueeze(-1).expand(*x.shape[:-1], 1)
        return self.net(torch.cat([x, t_exp], dim=-1))