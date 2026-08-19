"""Neural fields: heads consumed by Process classes.

Two conditioning contracts live here:

- **Time-conditioned fields** (the default): ``nn.Module`` honouring
  ``forward(x, t, c=None)`` with an ``event_ndim`` attribute, where
  ``c`` is a fixed-width conditioning tensor.  Concrete heads cover
  velocity prediction (flow matching), scalar action potentials, scalar
  log-density heads (consistency models), and operator-parameter heads
  (generator matching).
- **Context-conditioned operator fields**:
  ``forward(query, context)`` with a structured context (see
  :mod:`nami.core.contexts`), for heads conditioned on an empirical
  cloud rather than a tensor.  Currently
  :class:`~nami.fields.green_operator.GreenOperatorPotentialField`.
  Generic consumers of the ``(x, t, c)`` contract must not assume it
  holds for these.
"""

from __future__ import annotations

from nami.fields.action import ActionHead
from nami.fields.adaln import AdaLNVelocityField
from nami.fields.composite import (
    DriftFromVelocityScore,
    MarkovizationDriftFromVelocityScore,
    TwoHeadField,
)
from nami.fields.consistency import LogDensityHead
from nami.fields.ctmc import CTMCField
from nami.fields.generator import GeneratorField
from nami.fields.green_operator import (
    EmpiricalTangent,
    GreenOperatorPotentialField,
)
from nami.fields.scalar_potential import ScalarPotentialField
from nami.fields.transformer_velocity import TransformerVelocityField
from nami.fields.velocity import VelocityField

__all__ = [
    "ActionHead",
    "AdaLNVelocityField",
    "CTMCField",
    "DriftFromVelocityScore",
    "EmpiricalTangent",
    "GeneratorField",
    "GreenOperatorPotentialField",
    "LogDensityHead",
    "MarkovizationDriftFromVelocityScore",
    "ScalarPotentialField",
    "TransformerVelocityField",
    "TwoHeadField",
    "VelocityField",
]
