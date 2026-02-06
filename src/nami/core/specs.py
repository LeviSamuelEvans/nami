from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch


def as_tuple(x: Iterable[int] | int | None) -> tuple[int, ...]:
    """normaliser to take flexible input and return always a tuple for convenience.
    """
    if x is None:
        return ()
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(int(v) for v in x)
    return (int(x),)


def event_numel(event_shape: Iterable[int] | None) -> int:
    """ returns the total number of elements in the event shape
    """
    shape = as_tuple(event_shape)
    if not shape:
        return 1
    return int(math.prod(shape))


def split_event(
    x: torch.Tensor, event_ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """ given a tensor, return shape split into leading shape and event shape
    """
    if event_ndim < 0:
        raise ValueError("event_ndim must be >= 0")
    if event_ndim > x.ndim:
        raise ValueError("event_ndim exceeds x.ndim")
    if event_ndim == 0:
        return tuple(x.shape), ()
    return tuple(x.shape[:-event_ndim]), tuple(x.shape[-event_ndim:])


def flatten_event(x: torch.Tensor, event_ndim: int) -> torch.Tensor:
    """ collapse all event dimensions into a single flat dimension
    """
    if event_ndim < 0:
        raise ValueError("event_ndim must be >= 0")
    if event_ndim > x.ndim:
        raise ValueError("event_ndim exceeds x.ndim")
    if event_ndim == 0:
        return x
    return x.reshape(*x.shape[:-event_ndim], -1)


def unflatten_event(x: torch.Tensor, event_shape: tuple[int, ...]) -> torch.Tensor:
    """ inverse of `flatten_event`
    """
    if not event_shape:
        return x
    return x.reshape(*x.shape[:-1], *event_shape)


def validate_shapes(
    tensor: torch.Tensor,
    event_ndim: int,
    expected_event_shape: tuple[int, ...] | None = None,
    batch_shape: tuple[int, ...] | None = None,
) -> None:
    """ Runtime assertion helper to enforce explicit shapes and
    prevent silent broadcasting 
    """
    if event_ndim < 0:
        raise ValueError("event_ndim must be >= 0")
    if event_ndim > tensor.ndim:
        raise ValueError("event_ndim exceeds tensor.ndim")

    if expected_event_shape is not None:
        actual_event_shape = tuple(tensor.shape[-event_ndim:] if event_ndim > 0 else ())
        if actual_event_shape != expected_event_shape:
            raise ValueError(
                f"event_shape mismatch: expected {expected_event_shape}, got {actual_event_shape}"
            )

    if batch_shape is not None:
        actual_batch_shape = tuple(
            tensor.shape[:-event_ndim] if event_ndim > 0 else tensor.shape
        )
        if actual_batch_shape != batch_shape:
            raise ValueError(
                f"batch_shape mismatch: expected {batch_shape}, got {actual_batch_shape}"
            )


@dataclass(frozen=True)
class TensorSpec:
    """minimal specification of tensor for models, samplers, and distribution components.

    Attributes:
        event_shape (tuple[int, ...]): The shape of a single event (sample, vector, matrix, etc).
        dtype (torch.dtype | None): The expected data type of the tensor.
    """
    event_shape: tuple[int, ...]
    dtype: torch.dtype | None = None

    @property
    def event_ndim(self) -> int:
        return len(self.event_shape)

    @property
    def numel(self) -> int:
        return event_numel(self.event_shape)
