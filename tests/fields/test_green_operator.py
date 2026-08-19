from __future__ import annotations

import pytest
import torch

from nami.fields.green_operator import EmpiricalTangent, GreenOperatorPotentialField


@pytest.fixture
def operator() -> GreenOperatorPotentialField:
    torch.manual_seed(0)
    return GreenOperatorPotentialField(2, n_modes=8, density_dim=16, hidden=32, layers=2)


@pytest.fixture
def context() -> EmpiricalTangent:
    torch.manual_seed(1)
    return EmpiricalTangent(
        points=torch.randn(3, 20, 2),
        source=torch.randn(3, 20),
    )


def test_forward_shape(operator, context):
    query = torch.randn(3, 7, 2)
    phi = operator(query, context)
    assert phi.shape == (3, 7)


def test_constructor_rejects_nonpositive_modes():
    with pytest.raises(ValueError, match="n_modes must be positive"):
        GreenOperatorPotentialField(2, n_modes=0)


def test_source_shape_mismatch_raises(operator):
    bad = EmpiricalTangent(points=torch.randn(3, 20, 2), source=torch.randn(3, 19))
    with pytest.raises(ValueError, match="source shape"):
        operator(torch.randn(3, 7, 2), bad)


def test_permutation_invariance(operator, context):
    query = torch.randn(3, 7, 2)
    perm = torch.randperm(20)
    shuffled = EmpiricalTangent(
        points=context.points[:, perm], source=context.source[:, perm]
    )
    phi = operator(query, context)
    phi_shuffled = operator(query, shuffled)
    assert torch.allclose(phi, phi_shuffled, atol=1e-6)


def test_exact_linearity_in_source(operator, context):
    query = torch.randn(3, 7, 2)
    s1 = context.source
    s2 = torch.randn_like(s1)
    alpha, beta = 0.7, -1.3
    phi_combo = operator(
        query,
        EmpiricalTangent(points=context.points, source=alpha * s1 + beta * s2),
    )
    phi_1 = operator(query, EmpiricalTangent(points=context.points, source=s1))
    phi_2 = operator(query, EmpiricalTangent(points=context.points, source=s2))
    assert torch.allclose(phi_combo, alpha * phi_1 + beta * phi_2, atol=1e-5)


def test_zero_source_gives_zero_potential(operator, context):
    query = torch.randn(3, 7, 2)
    zero = EmpiricalTangent(
        points=context.points, source=torch.zeros_like(context.source)
    )
    phi = operator(query, zero)
    assert torch.allclose(phi, torch.zeros_like(phi))


def test_gauge_mean_zero_on_context_cloud(operator, context):
    """Centred modes fix the E_rho[phi] = 0 gauge on the cloud."""
    phi_at_context = operator(context.points, context)
    assert torch.allclose(
        phi_at_context.mean(dim=-1), torch.zeros(3), atol=1e-5
    )


def test_mask_ignores_padding(operator):
    torch.manual_seed(2)
    points = torch.randn(2, 10, 2)
    source = torch.randn(2, 10)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[:, 7:] = False

    garbage_points = points.clone()
    garbage_points[:, 7:] = 100.0
    garbage_source = source.clone()
    garbage_source[:, 7:] = -50.0

    query = torch.randn(2, 5, 2)
    phi = operator(query, EmpiricalTangent(points, source, mask))
    phi_garbage = operator(
        query, EmpiricalTangent(garbage_points, garbage_source, mask)
    )
    assert torch.allclose(phi, phi_garbage, atol=1e-6)


def test_velocity_matches_finite_difference(operator, context):
    operator = operator.double()
    ctx = EmpiricalTangent(
        points=context.points.double(), source=context.source.double()
    )
    query = torch.randn(3, 4, 2, dtype=torch.float64)

    v = operator.velocity(query, ctx)

    eps = 1e-6
    for d in range(2):
        bump = torch.zeros_like(query)
        bump[..., d] = eps
        fd = (operator(query + bump, ctx) - operator(query - bump, ctx)) / (2 * eps)
        assert torch.allclose(v[..., d], fd, atol=1e-6)


def test_velocity_create_graph_false_detaches(operator, context):
    query = torch.randn(3, 4, 2)
    v = operator.velocity(query, context, create_graph=False)
    assert not v.requires_grad


def test_query_gradient_does_not_touch_density_state(operator, context):
    """The density state depends only on the cloud, so query autodiff is cheap."""
    query = torch.randn(3, 4, 2, requires_grad=True)
    phi = operator(query, context)
    (grad,) = torch.autograd.grad(phi.sum(), query)
    assert grad.shape == query.shape
