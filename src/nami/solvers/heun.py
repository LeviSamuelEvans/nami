from __future__ import annotations

import torch


class Heun:
    requires_steps = True
    supports_rsample = True
    is_sde = False

    def __init__(self, steps: int = 32):
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        self.steps = int(steps)

    def integrate(
        self,
        f,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        dt = (t1 - t0) / steps
        x = x0
        t = t0

        for _ in range(steps):
            k1 = f(x, t)
            k2 = f(x + dt * k1, t + dt)
            x = x + (dt / 2.0) * (k1 + k2)
            t = t + dt

        return x

    def integrate_augmented(
        self,
        f_aug,
        x0: torch.Tensor,
        logp0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        dt = (t1 - t0) / steps
        x = x0
        logp = logp0
        t = t0

        for _ in range(steps):
            k1, l1 = f_aug(x, t)
            k2, l2 = f_aug(x + dt * k1, t + dt)

            x = x + (dt / 2.0) * (k1 + k2)
            logp = logp + (dt / 2.0) * (l1 + l2)
            t = t + dt

        return x, logp
