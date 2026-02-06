from __future__ import annotations

import torch


class DivergenceEstimator:
    def __call__(
        self, field, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None
    ) -> torch.Tensor:
        raise NotImplementedError
