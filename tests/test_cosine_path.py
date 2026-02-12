from __future__ import annotations

import math

import pytest
import torch


class TestCosinePathSampleXt:
    """Tests for CosinePath.sample_xt method."""

    def test_t0_returns_target(self, cosine_path, sample_tensor_4d):
        """At t=0, xt should equal x_target."""
        x_source = torch.randn_like(sample_tensor_4d)
        t = torch.zeros(2)

        xt = cosine_path.sample_xt(sample_tensor_4d, x_source, t)

        assert torch.allclose(xt, sample_tensor_4d)

    def test_t1_returns_source(self, cosine_path, sample_tensor_4d):
        """At t=1, xt should equal x_source."""
        x_source = torch.randn_like(sample_tensor_4d)
        t = torch.ones(2)

        xt = cosine_path.sample_xt(sample_tensor_4d, x_source, t)

        assert torch.allclose(xt, x_source, atol=1e-5)

    def test_t_mid_interpolation(self, cosine_path):
        """At t=0.5, should be cosine-weighted interpolation."""
        x_target = torch.ones(2, 3, 4, 5)
        x_source = torch.zeros(2, 3, 4, 5)
        t = torch.full((2,), 0.5)

        xt = cosine_path.sample_xt(x_target, x_source, t)

        expected = math.cos(math.pi / 4) * x_target + math.sin(math.pi / 4) * x_source
        assert torch.allclose(xt, expected, rtol=1e-5)

    def test_broadcasting(self, cosine_path, sample_tensor_4d):
        """Time tensor should broadcast correctly over batch dimensions."""
        x_source = torch.randn_like(sample_tensor_4d)
        t = torch.tensor([0.0, 0.5])

        xt = cosine_path.sample_xt(sample_tensor_4d, x_source, t)

        assert xt.shape == sample_tensor_4d.shape
        assert torch.allclose(xt[0], sample_tensor_4d[0])
        expected_mid = (
            math.cos(math.pi / 4) * sample_tensor_4d[1]
            + math.sin(math.pi / 4) * x_source[1]
        )
        assert torch.allclose(xt[1], expected_mid)


class TestCosinePathTargetUt:
    """Tests for CosinePath.target_ut method."""

    def test_t0_velocity(self, cosine_path, sample_tensor_4d):
        """At t=0, velocity should be (pi/2) * x_source."""
        x_source = torch.randn_like(sample_tensor_4d)
        t = torch.zeros(2)

        ut = cosine_path.target_ut(sample_tensor_4d, x_source, t)

        expected = (math.pi / 2) * x_source
        assert torch.allclose(ut, expected, rtol=1e-5)

    def test_t1_velocity(self, cosine_path, sample_tensor_4d):
        """At t=1, velocity should be -(pi/2) * x_target."""
        x_source = torch.randn_like(sample_tensor_4d)
        t = torch.ones(2)

        ut = cosine_path.target_ut(sample_tensor_4d, x_source, t)

        # sigma_prime(1) = (pi/2)*cos(pi/2) is not exactly zero in float32,
        # so the x_source term contributes a small residual (~1e-7).
        expected = -(math.pi / 2) * sample_tensor_4d
        assert torch.allclose(ut, expected, rtol=1e-5, atol=1e-5)


class TestCosinePathInternals:
    """Tests for CosinePath internal methods."""

    @pytest.mark.parametrize(
        ("t_val", "expected_alpha", "expected_sigma"),
        [
            (0.0, 1.0, 0.0),
            (0.5, math.cos(math.pi / 4), math.sin(math.pi / 4)),
            (1.0, 0.0, 1.0),
        ],
        ids=["t0", "t_mid", "t1"],
    )
    def test_alpha_sigma_values(
        self, cosine_path, t_val, expected_alpha, expected_sigma
    ):
        """Test alpha and sigma at key time points."""
        t = torch.tensor([t_val])

        alpha = cosine_path._alpha(t)
        sigma = cosine_path._sigma(t)

        assert alpha.item() == pytest.approx(expected_alpha, abs=1e-5)
        assert sigma.item() == pytest.approx(expected_sigma, abs=1e-5)

    def test_derivatives_at_t0(self, cosine_path):
        """Test alpha' and sigma' at t=0."""
        t = torch.tensor([0.0])

        alpha_prime = cosine_path._alpha_prime(t)
        sigma_prime = cosine_path._sigma_prime(t)

        assert alpha_prime.item() == pytest.approx(0.0, abs=1e-5)
        assert sigma_prime.item() == pytest.approx(math.pi / 2, rel=1e-5)

    def test_all_methods_vectorized(self, cosine_path):
        """Test that all methods work with batched time tensors."""
        t = torch.tensor([0.0, 0.5, 1.0])

        alpha = cosine_path._alpha(t)
        sigma = cosine_path._sigma(t)
        alpha_prime = cosine_path._alpha_prime(t)
        sigma_prime = cosine_path._sigma_prime(t)

        assert alpha.shape == (3,)
        assert sigma.shape == (3,)
        assert alpha_prime.shape == (3,)
        assert sigma_prime.shape == (3,)

        # Boundary checks
        assert alpha[0].item() == pytest.approx(1.0, abs=1e-5)
        assert alpha[2].item() == pytest.approx(0.0, abs=1e-5)
        assert sigma[0].item() == pytest.approx(0.0, abs=1e-5)
        assert sigma[2].item() == pytest.approx(1.0, abs=1e-5)
