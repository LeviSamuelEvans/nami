"""Core utilities and protocols for flow-based generative models.

This package provides fundamental building blocks for flow matching and
diffusion models:

- **specs**: Tensor shape specification and manipulation utilities
- **typing**: Protocol definitions for model components
- **broadcast**: Tensor broadcasting utilities

Exported Functions
------------------
TensorSpec
    Specification of tensor shape and dtype.
as_tuple
    Normalize input to tuple of integers.
event_numel
    Compute total elements in event shape.
flatten_event
    Collapse event dimensions to flat dimension.
split_event
    Split tensor shape into batch and event parts.
unflatten_event
    Restore flattened event dimensions.
validate_shapes
    Validate tensor shapes against expectations.
"""

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
