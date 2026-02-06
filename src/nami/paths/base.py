from __future__ import annotations

import torch


class ProbabilityPath:
    def sample_xt(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def target_ut(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
