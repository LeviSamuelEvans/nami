"""Protocol definitions for flow-based generative models.

This module defines abstract interfaces (Protocols) for key components
in flow matching and diffusion models, including divergence estimators,
probability paths, noise schedules, and ODE/SDE solvers.
"""

from __future__ import annotations

from typing import Protocol

import torch


class DivergenceEstimator(Protocol):
    """Interface for computing divergence of velocity field.
    
    Used in log-likelihood calculations via change of variables formula.
    Implementations include exact Jacobian computation and Hutchinson's
    trace estimator.
    
    Parameters
    ----------
    field : callable
        Velocity field function.
    x : Tensor
        State tensor.
    t : Tensor
        Time values.
    c : Tensor, optional
        Conditioning context.
    
    Returns
    -------
    Tensor
        Divergence of the velocity field at (x, t, c).
    """
    def __call__(
        self, field, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None
    ) -> torch.Tensor: ...


class ProbabilityPath(Protocol):
    """Interface for interpolation paths in flow matching models.
    
    Defines the continuous interpolation between source and target
    distributions, parameterized by time t ∈ [0, 1].

    Methods
    -------
    sample_xt(x_target, x_source, t)
        Sample interpolated point along the path at time ``t``.
    target_ut(x_target, x_source, t)
        Compute ground truth velocity field used in the loss.
    
    Notes
    -----
    Common implementations include linear interpolation and
    geodesic paths on manifolds.
    """
    def sample_xt(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor: ...

    def target_ut(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor: ...


class NoiseSchedule(Protocol):
    """Interface for diffusion model noise schedules.
    
    Defines the forward process: x_t = α(t) * x_0 + σ(t) * ε,
    where ε ~ N(0, I).

    Methods
    -------
    alpha(t)
        Signal scaling coefficient at time ``t``.
    sigma(t)
        Noise scaling coefficient at time ``t``.
    snr(t)
        Signal-to-noise ratio: α²(t) / σ²(t).
    drift(x, t)
        Drift term in SDE: dx = f(x,t)dt + g(t)dW.
    diffusion(t)
        Diffusion coefficient g(t) in the SDE.
    
    Notes
    -----
    Common schedules include linear, cosine, and variance-preserving schedules.
    """
    def alpha(self, t: torch.Tensor) -> torch.Tensor: ...

    def sigma(self, t: torch.Tensor) -> torch.Tensor: ...

    def snr(self, t: torch.Tensor) -> torch.Tensor: ...

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...

    def diffusion(self, t: torch.Tensor) -> torch.Tensor: ...


class ODESolver(Protocol):
    """Interface for ODE integrators.
    
    Solves ordinary differential equations dx/dt = f(x, t) for flow-based
    generative models, with optional augmented integration for log-likelihood.

    Attributes
    ----------
    requires_steps : bool
        Whether solver requires a fixed number of steps.
    supports_rsample : bool
        Whether solver supports reparameterized sampling (gradients flow
        through the sampling process).
    is_sde : bool
        Should be False for ODE solvers.
    
    Methods
    -------
    integrate(f, x0, t0, t1, atol, rtol, steps)
        Solve dx/dt = f(x,t) from ``t0`` to ``t1`` given initial state ``x0``.
    integrate_augmented(f_aug, x0, logp0, t0, t1, atol, rtol, steps)
        Jointly solve for state and log-probability change.
    
    Notes
    -----
    Common implementations include Euler, Heun, and Runge-Kutta methods,
    as well as adaptive solvers like Dopri5.
    """
    
    # does the solver require a fixed number of step counts to be specified?
    requires_steps: bool
    
    # can the solver produce reparameterised samples (i.e. gradients flow through the sampling)
    supports_rsample: bool
    
    is_sde: bool  # should be False for ODE solvers

    def integrate(
        self,
        f,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        steps: int | None = None,
    ) -> torch.Tensor: ...

    def integrate_augmented(
        self,
        f_aug,
        x0: torch.Tensor,
        logp0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class SDESolver(Protocol):
    """Interface for SDE integrators.
    
    Solves stochastic differential equations dx = f(x,t)dt + g(t)dW
    for diffusion-based generative models.

    Attributes
    ----------
    is_sde : bool
        Should be True for SDE solvers.
    
    Methods
    -------
    integrate(drift, diffusion, x0, t0, t1, steps)
        Integrate SDE from ``t0`` to ``t1`` with ``steps`` discrete steps.
    
    Notes
    -----
    Common implementations include Euler-Maruyama and other
    stochastic integrators for diffusion models.
    """
    is_sde: bool  # should be True for SDE solvers

    def integrate(
        self,
        drift,
        diffusion,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int,
    ) -> torch.Tensor: ...
