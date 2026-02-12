from __future__ import annotations

import torch

from .base import ProbabilityPath


class LinearPath(ProbabilityPath):
    def sample_xt(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # broadcast t to the same shape as x_target to allow for implicit broadcasting
        t = t.reshape(t.shape + (1,) * (x_target.ndim - t.ndim))
        return (1.0 - t) * x_target + t * x_source

    def target_ut(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        _ = t
        return x_source - x_target
