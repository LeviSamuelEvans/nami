from __future__ import annotations

import math

import torch


class EulerMaruyama:
    requires_steps = True
    is_sde = True

    def __init__(self, steps: int = 64):
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        self.steps = int(steps)

    def integrate(
        self,
        drift,
        diffusion,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(steps or self.steps)
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        dt = (t1 - t0) / steps
        sqrt_dt = math.sqrt(abs(dt))
        x = x0
        t = t0

        for _ in range(steps):
            g = diffusion(t)
            noise = torch.randn_like(x)
            x = x + drift(x, t) * dt + g * sqrt_dt * noise
            t = t + dt

        return x
