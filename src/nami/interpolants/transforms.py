from __future__ import annotations

import torch
from torch import nn

from .gamma import GammaSchedule

# Based on https://github.com/malbergo/stochastic-interpolants/tree/main [https://arxiv.org/abs/2303.08797 Albergo et al.]


def _expand_time_like(
    scale: torch.Tensor, target: torch.Tensor, event_ndim: int | None
) -> torch.Tensor:
    if event_ndim is None:
        while scale.ndim < target.ndim:
            scale = scale.unsqueeze(-1)
        return scale

    lead_ndim = target.ndim - event_ndim
    if scale.ndim > lead_ndim:
        return scale

    n_prepend = lead_ndim - scale.ndim
    shape = (1,) * n_prepend + tuple(scale.shape) + (1,) * event_ndim
    return scale.reshape(shape)


class ScoreFromNoise(nn.Module):
    """Convert a noise-prediction model eta(x, t) into a score model s(x, t)."""

    def __init__(
        self, eta_model: nn.Module, gamma_schedule: GammaSchedule, eps: float = 1e-12
    ):
        super().__init__()
        self.eta_model = eta_model
        self.gamma_schedule = gamma_schedule
        self.eps = float(eps)

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.eta_model, "event_ndim", None)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        eta_val = self.eta_model(x, t, c)
        gamma_val = torch.clamp(self.gamma_schedule.gamma(t), min=self.eps)
        gamma_val = _expand_time_like(gamma_val, eta_val, self.event_ndim)
        return eta_val / gamma_val


class DriftFromVelocityScore(nn.Module):
    """Combine velocity and score into drift b(x, t) = v + gamma*gamma_dot*s."""

    def __init__(
        self,
        velocity_model: nn.Module,
        score_model: nn.Module,
        gamma_schedule: GammaSchedule,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.score_model = score_model
        self.gamma_schedule = gamma_schedule

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.velocity_model, "event_ndim", None)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        v_val = self.velocity_model(x, t, c)
        s_val = self.score_model(x, t, c)
        gg_val = self.gamma_schedule.gamma_gamma_dot(t)
        gg_val = _expand_time_like(gg_val, s_val, self.event_ndim)
        return v_val + gg_val * s_val


class MirrorVelocityFromScore(nn.Module):
    """Create mirror-flow velocity v_mirror(x, t) = gamma*gamma_dot*s(x, t)."""

    def __init__(self, score_model: nn.Module, gamma_schedule: GammaSchedule):
        super().__init__()
        self.score_model = score_model
        self.gamma_schedule = gamma_schedule

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.score_model, "event_ndim", None)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        s_val = self.score_model(x, t, c)
        gg_val = self.gamma_schedule.gamma_gamma_dot(t)
        gg_val = _expand_time_like(gg_val, s_val, self.event_ndim)
        return gg_val * s_val
