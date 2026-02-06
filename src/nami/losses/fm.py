from __future__ import annotations

import torch

from ..paths.linear import LinearPath


def fm_loss(
    field,
    x_target: torch.Tensor,
    x_source: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    path=None,
    reduction: str = "mean",
) -> torch.Tensor:
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        raise ValueError("field.event_ndim is required")

    # default to LinearPath if no path is provided
    if path is None:
        path = LinearPath()

    lead = x_target.shape[:-event_ndim] if event_ndim else x_target.shape

    if t is None:
        # always use float for time, even if x_target is integer or low-precision
        dtype = x_target.dtype if x_target.dtype.is_floating_point else torch.float32
        t = torch.rand(lead, device=x_target.device, dtype=dtype)
    elif t.shape != lead:
        t = t.expand(lead)

    xt = path.sample_xt(x_target, x_source, t)
    ut = path.target_ut(x_target, x_source, t)
    vt = field(xt, t, c)

    mse = (vt - ut).pow(2).reshape(*lead, -1).mean(dim=-1)

    if reduction == "none":
        return mse
    if reduction == "sum":
        return mse.sum()
    if reduction == "mean":
        return mse.mean()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")
