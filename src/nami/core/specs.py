"""Tensor specification and shape manipulation utilities.

This module provides utilities for working with tensor shapes in generative models,
including splitting, flattening, and validating event and batch dimensions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch


def as_tuple(x: Iterable[int] | int | None) -> tuple[int, ...]:
    """Normalize flexible input to a tuple of integers.
    
    Parameters
    ----------
    x : Iterable[int], int, or None
        Value to convert. Can be *None*, a single integer, a tuple,
        or a list of integers.
    
    Returns
    -------
    tuple of int
        Empty tuple if *x* is *None*, otherwise a tuple of integers.
    
    Examples
    --------
    >>> as_tuple(None)
    ()
    >>> as_tuple(5)
    (5,)
    >>> as_tuple([2, 3, 4])
    (2, 3, 4)
    """
    if x is None:
        return ()
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(int(v) for v in x)
    return (int(x),)


def event_numel(event_shape: Iterable[int] | None) -> int:
    """Compute the total number of elements in an event shape.
    
    Parameters
    ----------
    event_shape : Iterable[int] or None
        Shape of the event dimensions. If *None* or empty, returns 1.
    
    Returns
    -------
    int
        Product of all dimensions in *event_shape*, or 1 if empty.
    
    Examples
    --------
    >>> event_numel(None)
    1
    >>> event_numel((2, 3, 4))
    24
    """
    shape = as_tuple(event_shape)
    if not shape:
        return 1
    return int(math.prod(shape))


def split_event(
    x: torch.Tensor, event_ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a tensor's shape into batch dimensions and event dimensions.
    
    Parameters
    ----------
    x : Tensor
        Input tensor to split.
    event_ndim : int
        Number of trailing dimensions to treat as event dimensions.
        Must be >= 0 and <= ``x.ndim``.
    
    Returns
    -------
    batch_shape : tuple of int
        Leading (batch) dimensions of *x*.
    event_shape : tuple of int
        Trailing (event) dimensions of *x*.
    
    Raises
    ------
    ValueError
        If *event_ndim* is negative or exceeds ``x.ndim``.
    
    Examples
    --------
    >>> x = torch.randn(2, 3, 4, 5)
    >>> split_event(x, 2)
    ((2, 3), (4, 5))
    >>> split_event(x, 0)
    ((2, 3, 4, 5), ())
    """
    if event_ndim < 0:
        raise ValueError("event_ndim must be >= 0")
    if event_ndim > x.ndim:
        raise ValueError("event_ndim exceeds x.ndim")
    if event_ndim == 0:
        return tuple(x.shape), ()
    return tuple(x.shape[:-event_ndim]), tuple(x.shape[-event_ndim:])


def flatten_event(x: torch.Tensor, event_ndim: int) -> torch.Tensor:
    """Collapse all event dimensions into a single flat dimension.
    
    Parameters
    ----------
    x : Tensor
        Input tensor with shape ``(*batch_shape, *event_shape)``.
    event_ndim : int
        Number of trailing dimensions to flatten.
        Must be >= 0 and <= ``x.ndim``.
    
    Returns
    -------
    Tensor
        Tensor with shape ``(*batch_shape, prod(event_shape))``.
        If ``event_ndim`` is 0, returns ``x`` unchanged.
    
    Raises
    ------
    ValueError
        If ``event_ndim`` is negative or exceeds ``x.ndim``.
    
    See Also
    --------
    unflatten_event : Inverse operation.
    
    Examples
    --------
    >>> x = torch.randn(2, 3, 4, 5)
    >>> flatten_event(x, 2).shape
    torch.Size([2, 3, 20])
    """
    if event_ndim < 0:
        raise ValueError("event_ndim must be >= 0")
    if event_ndim > x.ndim:
        raise ValueError("event_ndim exceeds x.ndim")
    if event_ndim == 0:
        return x
    return x.reshape(*x.shape[:-event_ndim], -1)


def unflatten_event(x: torch.Tensor, event_shape: tuple[int, ...]) -> torch.Tensor:
    """Restore flattened event dimensions to their original shape.
    
    Inverse operation of :func:`flatten_event`.
    
    Parameters
    ----------
    x : Tensor
        Input tensor with shape ``(*batch_shape, flat_dim)``.
    event_shape : tuple of int
        Target event shape. Product must equal the last dimension of ``x``.
    
    Returns
    -------
    Tensor
        Tensor with shape ``(*batch_shape, *event_shape)``.
        If ``event_shape`` is empty, returns ``x`` unchanged.
    
    See Also
    --------
    flatten_event : Inverse operation.
    
    Examples
    --------
    >>> x = torch.randn(2, 3, 20)
    >>> unflatten_event(x, (4, 5)).shape
    torch.Size([2, 3, 4, 5])
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
    """Validate tensor shape against expected event and batch dimensions.
    
    Runtime assertion helper to enforce explicit shapes and prevent
    silent broadcasting errors.
    
    Parameters
    ----------
    tensor : Tensor
        Tensor to validate.
    event_ndim : int
        Number of trailing dimensions to treat as event dimensions.
        Must be >= 0 and <= ``tensor.ndim``.
    expected_event_shape : tuple of int, optional
        Expected shape of event dimensions. If provided, raises *ValueError*
        if actual event shape does not match.
    batch_shape : tuple of int, optional
        Expected shape of batch dimensions. If provided, raises *ValueError*
        if actual batch shape does not match.
    
    Raises
    ------
    ValueError
        If *event_ndim* is invalid, or if shapes do not match expectations.
    
    Examples
    --------
    >>> x = torch.randn(2, 3, 4, 5)
    >>> validate_shapes(x, event_ndim=2, expected_event_shape=(4, 5))
    >>> validate_shapes(x, event_ndim=2, batch_shape=(2, 3))
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
    """Minimal specification of tensor properties for models and distributions.
    
    Defines the expected shape and data type for tensors in flow-based
    generative models, enabling shape inference and validation.

    Parameters
    ----------
    event_shape : tuple of int
        The shape of a single event (sample, vector, matrix, etc).
    dtype : torch.dtype, optional
        The expected data type of the tensor.

    Attributes
    ----------
    event_shape : tuple of int
        Shape of a single event.
    dtype : torch.dtype or None
        Expected data type.
    event_ndim : int
        Number of dimensions in the event shape (read-only property).
    numel : int
        Total number of elements in the event shape (read-only property).
    
    Examples
    --------
    >>> spec = TensorSpec(event_shape=(3, 64, 64), dtype=torch.float32)
    >>> spec.event_ndim
    3
    >>> spec.numel
    12288
    """
    event_shape: tuple[int, ...]
    dtype: torch.dtype | None = None

    @property
    def event_ndim(self) -> int:
        return len(self.event_shape)

    @property
    def numel(self) -> int:
        return event_numel(self.event_shape)
