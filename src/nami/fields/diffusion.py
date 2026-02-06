from __future__ import annotations

import torch


def _expand_like(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand scale tensor to broadcast with target by adding trailing dimensions."""
    while scale.ndim < target.ndim:
        scale = scale.unsqueeze(-1)
    return scale


def eps_to_score(eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return -eps / _expand_like(sigma, eps)


def score_to_eps(score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return -score * _expand_like(sigma, score)


def eps_to_x0(
    x: torch.Tensor, eps: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    return (x - _expand_like(sigma, x) * eps) / _expand_like(alpha, x)


def x0_to_eps(
    x: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    return (x - _expand_like(alpha, x) * x0) / _expand_like(sigma, x)


def score_to_x0(
    x: torch.Tensor, score: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    sigma_exp = _expand_like(sigma, x)
    return (x + (sigma_exp**2) * score) / _expand_like(alpha, x)


def x0_to_score(
    x: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    sigma_exp = _expand_like(sigma, x)
    return (_expand_like(alpha, x) * x0 - x) / (sigma_exp**2)
