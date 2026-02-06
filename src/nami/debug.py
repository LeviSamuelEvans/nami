from __future__ import annotations

import torch

from .core.specs import split_event


def describe(
    x: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    event_ndim: int | None = None,
) -> str:
    lines = []
    if x is not None:
        lines.append(f"x: shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
        if event_ndim is not None:
            lead, event_shape = split_event(x, event_ndim)
            lines.append(f"  lead={lead} event_shape={event_shape}")
    if c is not None:
        lines.append(f"c: shape={tuple(c.shape)} dtype={c.dtype} device={c.device}")
    if t is not None:
        lines.append(f"t: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
    return "\n".join(lines)
