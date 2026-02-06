from __future__ import annotations

import torch
from torch import nn


class VectorField(nn.Module):
    @property
    def event_ndim(self) -> int:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def call_and_divergence(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        *,
        estimator=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
