"""Tensor broadcasting utilities for generative models.

This module provides utilities to broadcast data, time, and context tensors
to compatible shapes while preserving event and context dimensions.
"""

from __future__ import annotations

import torch

from .specs import split_event


def broadcast(
    x: torch.Tensor,
    t: torch.Tensor | None,
    c: torch.Tensor | None,
    *,
    event_ndim: int,
    validate_args: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Broadcast data, time, and context tensors to compatible shapes.
    
    Expands tensors *x*, *t*, and *c* to have compatible batch dimensions, 
    preserving the event dimensions of *x* and the context dimension of *c*.
    
    Parameters
    ----------
    x : Tensor
        Data tensor with shape ``(*batch_shape, *event_shape)``.
    t : Tensor, optional
        Time values with shape broadcastable to ``batch_shape``.
        If None, not broadcasted.
    c : Tensor, optional
        Context tensor with shape ``(*context_batch, context_dim)``.
        If None, not broadcasted.
    event_ndim : int
        Number of trailing event dimensions in ``x``.
    validate_args : bool, default=True
        If True, raises ValueError when broadcasting fails.
    
    Returns
    -------
    x : Tensor
        Broadcasted data tensor.
    t : Tensor or None
        Broadcasted time tensor, or None if input was None.
    c : Tensor or None
        Broadcasted context tensor, or None if input was None.
    
    Raises
    ------
    ValueError
        If ``validate_args`` is True and tensors cannot be broadcast together,
        or if context tensor has less than 1 dimension.
    """
    lead, event_shape = split_event(x, event_ndim)
    shapes: list[tuple[int, ...]] = [lead]

    if t is not None:
        shapes.append(tuple(t.shape))
    if c is not None:
        if c.ndim < 1:
            if validate_args:
                raise ValueError("context tensor must have at least 1 dim")
        else:
            shapes.append(tuple(c.shape[:-1]))

    try:
        target = torch.broadcast_shapes(*shapes)
    except RuntimeError as exc:
        if validate_args:
            raise ValueError("failed to broadcast x, t, c") from exc
        raise

    if tuple(x.shape[:-event_ndim] if event_ndim else x.shape) != target:
        x = x.expand(target + event_shape)

    if t is not None and tuple(t.shape) != target:
        t = t.expand(target)

    if c is not None and c.ndim >= 1:
        ctx = c.shape[-1]
        if tuple(c.shape[:-1]) != target:
            c = c.expand(target + (ctx,))

    return x, t, c
