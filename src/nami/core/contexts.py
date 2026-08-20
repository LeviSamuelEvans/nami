"""Structured conditioning contexts.

Data contracts for fields whose conditioning is richer than the
fixed-width tensor ``c`` of the ``(x, t, c)`` convention.  Kept in
``core`` so runtime consumers (processes, path adapters) can depend on
the contract without importing any concrete field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["EmpiricalTangent"]


@dataclass(frozen=True)
class EmpiricalTangent:
    """Empirical representation of a density-tangent pair.

    A context cloud of samples from a density together with the Poisson
    source (the parameter score, or its along-path directional
    contraction) evaluated at each sample.  The conditioning input of
    :class:`~nami.fields.green_operator.GreenOperatorPotentialField`.

    Attributes
    ----------
    points : torch.Tensor
        Context samples ``x_j ~ rho``, shape
        ``(*tasks, n_context, *event_shape)``.
    source : torch.Tensor
        Source values ``s(x_j)``, shape ``(*tasks, n_context)``.
    mask : torch.Tensor or None
        Optional boolean validity mask, shape ``(*tasks, n_context)``.
        ``False`` entries are ignored (padding).
    """

    points: torch.Tensor
    source: torch.Tensor
    mask: torch.Tensor | None = None
