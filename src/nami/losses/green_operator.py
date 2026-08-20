r"""Variational (Ritz) loss for the learned Green's operator.

The weak form of the weighted Poisson problem
:math:`-\nabla\!\cdot(\rho\nabla\phi) = \rho\,s` is the Ritz functional

.. math::

    \mathcal J(\phi) = \mathbb E_{\rho}\Bigl[
        \tfrac12 \|\nabla_x \phi(x)\|^2 - s(x)\,\phi(x)
    \Bigr],

whose Euler-Lagrange equation recovers the strong form.  Batched over
distribution-level *tasks* — each task one ``(rho, s)`` pair carried by
an :class:`~nami.fields.green_operator.EmpiricalTangent` — this trains a
single operator across densities rather than one potential per density.

Only the parameter score enters (no spatial score, no Laplacian, no
divergence estimator in the training path), and it enters **linearly**,
so a mined simulator score is an unbiased substitute for the marginal
parameter score.

Two obligations the caller carries:

- **Query/context independence.**  The query set must be sampled
  independently of the context cloud; otherwise the operator can
  memorise the supplied source values instead of learning a solution
  operator that interpolates (the empirical-Ritz exploit).
- **Mean-zero source.**  The weighted Poisson problem is solvable only
  for a shape-preserving source (:math:`\mathbb E_\rho[s] = 0`).  A
  :math:`\theta`-dependent yield must be split off and only the
  normalised shape transported.
"""

from __future__ import annotations

import torch

from nami.core.contexts import EmpiricalTangent
from nami.losses._common import reduce_loss


def green_operator_loss(
    operator,
    *,
    context: EmpiricalTangent,
    query_x: torch.Tensor,
    query_source: torch.Tensor,
    reduction: str = "mean",
    create_graph: bool = True,
) -> torch.Tensor:
    r"""Ritz objective of the Green's operator on one batch of tasks.

    Parameters
    ----------
    operator
        Field with the
        :class:`~nami.fields.green_operator.GreenOperatorPotentialField`
        surface: ``(query, context) -> (*tasks, n_query)`` plus
        ``event_ndim``.
    context : EmpiricalTangent
        Context clouds and source values, shapes
        ``(*tasks, n_context, *event_shape)`` / ``(*tasks, n_context)``.
    query_x : torch.Tensor
        Query points sampled from the same densities, **independently
        of the context clouds**, shape ``(*tasks, n_query, *event_shape)``.
    query_source : torch.Tensor
        Source values at the query points, shape ``(*tasks, n_query)``.
        Consumed detached — a frozen target.  A mined (joint/latent)
        score is an unbiased substitute for the marginal score here.
    reduction : str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    create_graph : bool
        Build the graph through :math:`\nabla_q\phi` so
        ``loss.backward()`` reaches the operator's parameters.  Set
        ``False`` only for eval-time monitoring.
    """
    event_ndim = int(operator.event_ndim)
    lead = tuple(query_x.shape[: query_x.ndim - event_ndim])
    if tuple(query_source.shape) != lead:
        msg = (
            f"query_source shape {tuple(query_source.shape)} must match "
            f"the leading shape of query_x {lead}"
        )
        raise ValueError(msg)

    query_x = query_x.detach().requires_grad_(True)
    phi = operator(query_x, context)
    (grad_phi,) = torch.autograd.grad(
        outputs=phi.sum(),
        inputs=query_x,
        create_graph=create_graph,
    )

    kinetic = 0.5 * grad_phi.reshape(*lead, -1).square().sum(dim=-1)
    loss_per_sample = kinetic - query_source.detach() * phi

    assert loss_per_sample.shape == lead
    return reduce_loss(loss_per_sample, reduction)
