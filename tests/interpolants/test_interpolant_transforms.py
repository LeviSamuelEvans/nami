from __future__ import annotations

import torch
from torch import nn

from nami.interpolants.gamma import BrownianGamma, ScaledBrownianGamma, ZeroGamma
from nami.interpolants.transforms import (
    DriftFromVelocityScore,
    MirrorVelocityFromScore,
    ScoreFromNoise,
)


def _expand_like_time(
    scale: torch.Tensor, target: torch.Tensor, event_ndim: int = 1
) -> torch.Tensor:
    lead_ndim = target.ndim - event_ndim
    n_prepend = lead_ndim - scale.ndim
    shape = (1,) * n_prepend + tuple(scale.shape) + (1,) * event_ndim
    return scale.reshape(shape)


class ScoreModel(nn.Module):
    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, x, t, c=None):
        _ = t
        if c is None:
            return 2.0 * x
        return 2.0 * x + c


class EtaModel(nn.Module):
    def __init__(self, score_model: nn.Module, gamma_schedule):
        super().__init__()
        self.score_model = score_model
        self.gamma_schedule = gamma_schedule

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.score_model, "event_ndim", None)

    def forward(self, x, t, c=None):
        score = self.score_model(x, t, c)
        gamma = _expand_like_time(self.gamma_schedule.gamma(t), score)
        return gamma * score


class ConstantModel(nn.Module):
    def __init__(self, constant: float):
        super().__init__()
        self.constant = float(constant)
        self.last_c = None

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, x, t, c=None):
        _ = t
        self.last_c = c
        return torch.full_like(x, self.constant)


class TestScoreFromNoise:
    def test_recovers_score(self):
        score_model = ScoreModel()
        gamma = BrownianGamma()
        eta_model = EtaModel(score_model, gamma)
        wrapper = ScoreFromNoise(eta_model, gamma)

        x = torch.randn(12, 3)
        t = torch.linspace(0.1, 0.9, 12)
        c = torch.randn(12, 1)

        expected = score_model(x, t, c)
        actual = wrapper(x, t, c)

        assert wrapper.event_ndim == 1
        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_endpoint_safe_division(self):
        class OnesEta(nn.Module):
            @property
            def event_ndim(self) -> int:
                return 1

            def forward(self, x, t, c=None):
                _ = t, c
                return torch.ones_like(x)

        wrapper = ScoreFromNoise(OnesEta(), BrownianGamma(), eps=1e-6)
        x = torch.randn(4, 2)
        t = torch.zeros(4)
        out = wrapper(x, t)

        assert torch.isfinite(out).all()


class TestDriftFromVelocityScore:
    def test_zero_gamma_reduces_to_velocity(self):
        velocity = ConstantModel(2.0)
        score = ConstantModel(3.0)
        wrapper = DriftFromVelocityScore(velocity, score, ZeroGamma())

        x = torch.randn(6, 4)
        t = torch.rand(6)
        c = torch.randn(6, 1)
        out = wrapper(x, t, c)

        assert wrapper.event_ndim == 1
        assert torch.allclose(out, torch.full_like(x, 2.0))
        assert velocity.last_c is c
        assert score.last_c is c

    def test_formula_with_broadcasting(self):
        velocity = ConstantModel(2.0)
        score = ConstantModel(3.0)
        gamma = ScaledBrownianGamma(scale=1.5)
        wrapper = DriftFromVelocityScore(velocity, score, gamma)

        x = torch.randn(5, 7, 2)
        t = torch.linspace(0.1, 0.9, 7)
        out = wrapper(x, t)

        gg = _expand_like_time(gamma.gamma_gamma_dot(t), x)
        expected = torch.full_like(x, 2.0) + gg * 3.0
        assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)


class TestMirrorVelocityFromScore:
    def test_formula(self):
        score = ConstantModel(4.0)
        gamma = BrownianGamma()
        wrapper = MirrorVelocityFromScore(score, gamma)

        x = torch.randn(3, 5, 2)
        t = torch.linspace(0.1, 0.9, 5)
        c = torch.randn(5, 1)
        out = wrapper(x, t, c)

        gg = _expand_like_time(gamma.gamma_gamma_dot(t), x)
        expected = gg * 4.0
        assert wrapper.event_ndim == 1
        assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)
        assert score.last_c is c
