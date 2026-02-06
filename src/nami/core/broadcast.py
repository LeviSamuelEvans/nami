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
