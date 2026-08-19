r"""Learned Green's operator for the weighted Poisson problem.

Amortises the parameter-flow solve *across densities*: instead of one
potential per density (or per simulator family), a single operator

.. math::

    \mathcal{G}_\psi : (\rho, s) \mapsto \phi,
    \qquad -\nabla_x\!\cdot(\rho\,\nabla_x\phi) = \rho\, s,

is learned as a map from an **empirical** representation of the density
(a context cloud of samples) and a source (the parameter score evaluated
on that cloud) to a queryable scalar potential.  The solution of the
weighted Poisson problem can be written through the Green's function of
the weighted Laplacian,

.. math::

    \phi(x) = \int G_\rho(x, y)\, s(y)\, \rho(y)\, dy,

so the operator is implicitly learning the family
:math:`\{G_\rho\}` indexed by the density.

Two structural facts drive the architecture:

1. :math:`A_\rho^{-1}` is **nonlinear in** :math:`\rho` **but exactly
   linear in** :math:`s`.  The spectral form below routes the source
   only through per-mode averages, so linearity holds by construction.
2. The operator must be **queryable**: the query point is separate from
   the conditioning cloud, so :math:`v(q) = \nabla_q \phi_\psi(q; \mathcal C)`
   is a genuine velocity field obtainable anywhere by autodiff.

The realised form is a density-conditioned low-rank eigenfunction
expansion,

.. math::

    \phi(q) = \sum_{k=1}^{K} \lambda_k(\rho)\,
        \tilde h_k(q; \rho)\,
        \underbrace{\tfrac{1}{N_c}\textstyle\sum_j \tilde h_k(x_j; \rho)\, s_j}_{\hat a_k},

with learned density-dependent modes :math:`h_k`, learned positive
inverse eigenvalues :math:`\lambda_k`, and modes centred on the context
cloud (:math:`\tilde h_k = h_k - \bar h_k`), which removes the additive
gauge constant (:math:`\mathbb E_\rho[\phi] = 0`) by construction.

.. note::
   Not to be confused with :mod:`nami.generators.operators`, where
   "operator" means the *infinitesimal generator of a Markov semigroup*
   (the Generator-Matching runtime).  Here "operator" means the
   *solution operator of an elliptic problem* — the inverse of the
   weighted Laplacian.  Two unrelated mathematical objects sharing one
   English word.

This field is **context-conditioned**, not time-conditioned: its
``forward`` signature is ``(query, context)`` with an
:class:`~nami.core.contexts.EmpiricalTangent` context, deliberately
outside the ``(x, t, c)`` convention because the conditioning is a
structured cloud rather than a fixed-width tensor.
"""

from __future__ import annotations

import torch
from torch import nn

from nami.components import MLPBackbone
from nami.core.contexts import EmpiricalTangent
from nami.core.specs import TensorSpec, flatten_event, validate_shapes
from nami.fields._common import normalise_event_shape

__all__ = ["EmpiricalTangent", "GreenOperatorPotentialField"]


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int,
) -> torch.Tensor:
    """Mean over ``dim``, ignoring entries where ``mask`` is False."""
    if mask is None:
        return values.mean(dim=dim)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(values.dtype)
    total = (values * mask).sum(dim=dim)
    count = mask.sum(dim=dim).clamp_min(1.0)
    return total / count


class GreenOperatorPotentialField(nn.Module):
    r"""Density-conditioned solution operator :math:`(\rho, s) \mapsto \phi`.

    The context cloud enters permutation-invariantly through a deep-sets
    density encoder; the source enters **only linearly**, through the
    per-mode projections :math:`\hat a_k`; the query enters only through
    the mode functions, so :math:`\nabla_q \phi` never differentiates
    the density state.

    Guaranteed by construction: permutation invariance in the context
    cloud, exact linearity in the source, zero source
    :math:`\Rightarrow` zero potential, and the
    :math:`\mathbb E_\rho[\phi] = 0` gauge via centred modes.

    Parameters
    ----------
    dim : int or tuple[int, ...]
        Data dimensionality (event shape).
    n_modes : int
        Number of learned spectral modes :math:`K`.
    density_dim : int
        Width of the encoded density state.
    hidden : int
        Hidden layer width of the constituent MLPs.
    layers : int
        Number of hidden layers of the constituent MLPs.
    activation : str
        Activation function.
    """

    def __init__(
        self,
        dim: int | tuple[int, ...],
        *,
        n_modes: int = 32,
        density_dim: int = 64,
        hidden: int = 128,
        layers: int = 3,
        activation: str = "silu",
    ):
        super().__init__()
        if n_modes <= 0:
            msg = f"n_modes must be positive, got {n_modes}"
            raise ValueError(msg)

        self.spec = TensorSpec(normalise_event_shape(dim))
        self.n_modes = int(n_modes)
        self.density_dim = int(density_dim)

        self.point_encoder = MLPBackbone(
            self.flat_dim,
            density_dim,
            hidden=hidden,
            layers=layers,
            activation=activation,
        )
        self.density_head = MLPBackbone(
            density_dim,
            density_dim,
            hidden=hidden,
            layers=layers,
            activation=activation,
        )
        self.basis = MLPBackbone(
            self.flat_dim + density_dim,
            n_modes,
            hidden=hidden,
            layers=layers,
            activation=activation,
        )
        self.spectrum = MLPBackbone(
            density_dim,
            n_modes,
            hidden=hidden,
            layers=layers,
            activation=activation,
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.spec.event_shape

    @property
    def event_ndim(self) -> int:
        return self.spec.event_ndim

    @property
    def flat_dim(self) -> int:
        return self.spec.numel

    def encode_density(self, context: EmpiricalTangent) -> torch.Tensor:
        """Permutation-invariant density state from the context cloud.

        Returns shape ``(*tasks, density_dim)``.  Depends on
        ``context.points`` only — never on the source — so the state is
        reusable across sources on the same cloud and the operator's
        linearity in the source is untouched.
        """
        validate_shapes(context.points, self.spec)
        pts = flatten_event(context.points, self.event_ndim)
        embedded = self.point_encoder(pts)
        pooled = _masked_mean(embedded, context.mask, dim=-2)
        return self.density_head(pooled)

    def _modes(
        self,
        points_flat: torch.Tensor,
        density_state: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the ``n_modes`` learned mode functions at points.

        ``points_flat``: ``(*tasks, n, flat_dim)``;
        ``density_state``: ``(*tasks, density_dim)``.
        Returns ``(*tasks, n, n_modes)``.
        """
        state = density_state.unsqueeze(-2).expand(
            *points_flat.shape[:-1], self.density_dim
        )
        return self.basis(torch.cat([points_flat, state], dim=-1))

    def forward(
        self,
        query: torch.Tensor,
        context: EmpiricalTangent,
    ) -> torch.Tensor:
        r"""Evaluate :math:`\phi_\psi(q; \mathcal C)`.

        Parameters
        ----------
        query : torch.Tensor
            Query points, shape ``(*tasks, n_query, *event_shape)``.
            Kept separate from the context cloud so the velocity is
            defined anywhere.
        context : EmpiricalTangent
            The empirical density-tangent pair.

        Returns
        -------
        Tensor, shape ``(*tasks, n_query)``
            Potential values; gradient w.r.t. ``query`` is the
            transport velocity.
        """
        validate_shapes(query, self.spec)
        if context.source.shape != context.points.shape[
            : context.points.ndim - self.event_ndim
        ]:
            msg = (
                f"context source shape {tuple(context.source.shape)} must "
                "match the leading shape of context points "
                f"{tuple(context.points.shape[: context.points.ndim - self.event_ndim])}"
            )
            raise ValueError(msg)

        density_state = self.encode_density(context)

        ctx_flat = flatten_event(context.points, self.event_ndim)
        ctx_modes = self._modes(ctx_flat, density_state)
        mode_mean = _masked_mean(ctx_modes, context.mask, dim=-2)

        # Source enters only here, linearly: per-mode projections a_k.
        centred_ctx = ctx_modes - mode_mean.unsqueeze(-2)
        coefficients = _masked_mean(
            centred_ctx * context.source.unsqueeze(-1), context.mask, dim=-2
        )

        query_flat = flatten_event(query, self.event_ndim)
        query_modes = self._modes(query_flat, density_state)
        centred_query = query_modes - mode_mean.unsqueeze(-2)

        eigenvalues = torch.nn.functional.softplus(self.spectrum(density_state))

        return (
            centred_query
            * eigenvalues.unsqueeze(-2)
            * coefficients.unsqueeze(-2)
        ).sum(dim=-1)

    def velocity(
        self,
        query: torch.Tensor,
        context: EmpiricalTangent,
        *,
        create_graph: bool = True,
    ) -> torch.Tensor:
        r"""Recover :math:`v(q) = \nabla_q \phi_\psi(q; \mathcal C)` by autograd.

        Mirrors
        :meth:`~nami.fields.scalar_potential.ScalarPotentialField.velocity`:
        always runs under ``torch.enable_grad()``; ``create_graph=True``
        keeps the graph for second-order objectives.
        """
        with torch.enable_grad():
            if create_graph:
                qq = query if query.requires_grad else query.clone().requires_grad_(True)
            else:
                qq = query.detach().requires_grad_(True)
            phi = self(qq, context)
            (grad_phi,) = torch.autograd.grad(
                outputs=phi.sum(),
                inputs=qq,
                create_graph=create_graph,
            )
        return grad_phi
