from __future__ import annotations

import torch


class NoiseSchedule:
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        s = self.sigma(t)
        return (a * a) / (s * s)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
