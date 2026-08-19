from __future__ import annotations

import pytest
import torch

from nami.fields.green_operator import EmpiricalTangent, GreenOperatorPotentialField
from nami.losses.green_operator import green_operator_loss


@pytest.fixture
def operator() -> GreenOperatorPotentialField:
    torch.manual_seed(0)
    return GreenOperatorPotentialField(2, n_modes=8, density_dim=16, hidden=32, layers=2)


@pytest.fixture
def batch():
    torch.manual_seed(1)
    context = EmpiricalTangent(
        points=torch.randn(3, 20, 2),
        source=torch.randn(3, 20),
    )
    query_x = torch.randn(3, 15, 2)
    query_source = torch.randn(3, 15)
    return context, query_x, query_source


def test_loss_is_scalar(operator, batch):
    context, query_x, query_source = batch
    loss = green_operator_loss(
        operator, context=context, query_x=query_x, query_source=query_source
    )
    assert loss.shape == ()


def test_reduction_none_keeps_task_query_shape(operator, batch):
    context, query_x, query_source = batch
    loss = green_operator_loss(
        operator,
        context=context,
        query_x=query_x,
        query_source=query_source,
        reduction="none",
    )
    assert loss.shape == (3, 15)


def test_query_source_shape_mismatch_raises(operator, batch):
    context, query_x, _ = batch
    with pytest.raises(ValueError, match="query_source shape"):
        green_operator_loss(
            operator,
            context=context,
            query_x=query_x,
            query_source=torch.randn(3, 14),
        )


def test_backward_reaches_operator_parameters(operator, batch):
    context, query_x, query_source = batch
    loss = green_operator_loss(
        operator, context=context, query_x=query_x, query_source=query_source
    )
    loss.backward()
    grads = [p.grad for p in operator.parameters() if p.grad is not None]
    assert grads
    assert any(g.abs().sum() > 0 for g in grads)


def test_training_reduces_loss_on_fixed_task(operator, batch):
    context, query_x, query_source = batch
    optim = torch.optim.Adam(operator.parameters(), lr=1e-2)

    def evaluate() -> float:
        loss = green_operator_loss(
            operator,
            context=context,
            query_x=query_x,
            query_source=query_source,
            create_graph=False,
        )
        return float(loss.detach())

    before = evaluate()
    for _ in range(50):
        optim.zero_grad()
        loss = green_operator_loss(
            operator, context=context, query_x=query_x, query_source=query_source
        )
        loss.backward()
        optim.step()
    after = evaluate()
    assert after < before
