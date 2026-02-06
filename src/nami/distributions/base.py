from __future__ import annotations

from torch.distributions import Distribution


def expand_distribution(
    dist: Distribution, batch_shape: tuple[int, ...]
) -> Distribution:
    if dist.batch_shape == batch_shape:
        return dist
    if not hasattr(dist, "expand"):
        raise ValueError("distribution does not support expand")
    return dist.expand(batch_shape)


def has_rsample(dist: Distribution) -> bool:
    return bool(getattr(dist, "has_rsample", False))
