from __future__ import annotations

from .specs import (
    TensorSpec,
    as_tuple,
    event_numel,
    flatten_event,
    split_event,
    unflatten_event,
    validate_shapes,
)

__all__ = [
    "TensorSpec",
    "as_tuple",
    "event_numel",
    "flatten_event",
    "split_event",
    "unflatten_event",
    "validate_shapes",
]
