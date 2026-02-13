from __future__ import annotations

import torch

from ..paths.linear import LinearPath
from ._common import (
    leading_shape,
    per_sample_mse,
    prepare_time,
    reduce_loss,
    require_event_ndim,
)


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
    event_ndim = require_event_ndim(field)

    # default to LinearPath if no path is provided
    if path is None:
        path = LinearPath()

    lead = leading_shape(x_target, event_ndim)
    t = prepare_time(x_target, lead, t)

    xt = path.sample_xt(x_target, x_source, t)
    ut = path.target_ut(x_target, x_source, t)
    vt = field(xt, t, c)
    mse = per_sample_mse(vt, ut, lead)

    return reduce_loss(mse, reduction)
