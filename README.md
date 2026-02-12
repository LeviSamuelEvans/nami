# nami

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![codecov](https://codecov.io/gh/LeviSamuelEvans/nami/branch/main/graph/badge.svg)](https://codecov.io/gh/LeviSamuelEvans/nami)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A minimal, modular flow matching library for research studies, providing the building blocks for flow-based generative models with explicit shape semantics and pluggable components. Primarily, this is a personal tool for use in other projects (and for fun). Much of the design is heavily inspired by the fantastic [Zuko](https://github.com/probabilists/zuko/tree/master) library, who coined the term `LazyDistribution`.

## Installation

### Prerequisites

- Python >= 3.10
- [pixi](https://pixi.sh) package manager

### Setup

Clone the repository and run the setup task, which installs nami in editable mode:

```bash
git clone https://github.com/LeviSamuelEvans/nami
cd nami
```

```bash
pixi run setup
```

This creates a conda environment via pixi and installs the package with `pip install -e .`.

### Without pixi

If you prefer not to use pixi, you can install directly with pip (requires PyTorch >= 2.0):

```bash
pip install -e .
```

### Development tasks

pixi provides several convenience tasks:

| Command | Description |
|---------|-------------|
| `pixi run test` | Run tests with pytest |
| `pixi run cov` | Run tests with coverage report |
| `pixi run lint` | Lint with ruff |
| `pixi run fmt` | Format with ruff |
| `pixi run typecheck` | Type-check with mypy |
| `pixi run docs` | Build Sphinx documentation |
| `pixi run kernel` | Install a Jupyter kernel named `nami` |

## Examples

Hopefully you got everything nicely set up, so now here's are a few examples to demonstrate how to use `nami`. You can also find additional examples and walkthroughs in the `books/examples` folder.

### Flow Matching

A bare-bones setup for training an unconditional flow-matching model. We first define a simple neural network that will serve as our velocity field, then set up a base distribution, solver, and training loop.

```python
import torch
import torch.nn as nn
import nami

# Define a neural network for the velocity field.
# The field must expose `event_ndim` so nami knows how many
# trailing dimensions make up a single sample.
class VelocityField(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),  # +1 for time
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    @property
    def event_ndim(self) -> int:
        return 1  # output is a 1D vector per sample

    def forward(self, x: torch.Tensor, t: torch.Tensor, c=None) -> torch.Tensor:
        t_expanded = t.unsqueeze(-1).expand(*x.shape[:-1], 1)
        return self.net(torch.cat([x, t_expanded], dim=-1))

# Components
dim = 8
field = VelocityField(dim)
base = nami.StandardNormal(event_shape=(dim,))
solver = nami.RK4(steps=32)

# Training loop
optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
for step in range(1000):
    x_target = sample_your_data()  # <-- your data here
    x_source = torch.randn_like(x_target)

    # by default, if no path is passed to the loss function,
    # a `LinearPath` will be used.
    loss = nami.fm_loss(field, x_target, x_source)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# When sampling bind everything into a FlowMatching object,
# call it with context=None for unconditional generation,
# then draw samples from the resulting process.
fm = nami.FlowMatching(field, base, solver)
process = fm(None)  # None = no conditioning variable
samples = process.sample((64,))  # generate 64 samples
```

### Conditional Generation

To condition on external information, build a field that accepts a context tensor `c` alongside `x` and `t`. At sampling time, pass the context when binding the process.

```python
import torch
import torch.nn as nn
import nami

# context-aware field
class ConditionalField(nn.Module):
    def __init__(self, dim: int, context_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + context_dim + 1, 128),
            nn.SiLU(),
            nn.Linear(128, dim),
        )

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, x, t, c=None):
        t_exp = t.unsqueeze(-1).expand(*x.shape[:-1], 1)
        inputs = [x, t_exp]
        if c is not None:
            inputs.append(c)
        return self.net(torch.cat(inputs, dim=-1))

# setup
dim = 8
field = ConditionalField(dim=dim, context_dim=4)
base = nami.StandardNormal(event_shape=(dim,))
solver = nami.RK4(steps=32)
fm = nami.FlowMatching(field, base, solver)

# conditional sampling
context = torch.randn(16, 4)  # 16 different conditions
process = fm(context)  # bind context
samples = process.sample((1,))  # 1 sample per condition -> (1, 16, 8)
```

### Diffusion Models

nami also supports score-based diffusion. The same field architecture can be reused, just swap in a noise schedule and an SDE solver.

```python
import nami

# same field architecture works
schedule = nami.VPSchedule(beta_min=0.1, beta_max=20.0)
solver = nami.EulerMaruyama(steps=100)

diffusion = nami.Diffusion(
    model=field,
    schedule=schedule,
    solver=solver,
    parameterization="eps",  # or "score", "x0"
    event_shape=(dim,),
)

process = diffusion(None)
samples = process.sample((64,))
```

## Some Core Concepts

### Shape Convention

All tensors follow: `sample_shape + batch_shape + event_shape`. The `sample_shape` correpsonds to independent draws, `batch_shape` to the parallel compuations and `event_shape` to a single data point. The `event_ndim` property tells nami how many trailing dimensions form one sample.


### Lazy Binding

Models are defined once, then bound to specific contexts, for example:

```python
fm = FlowMatching(field, base, solver)  # configuration
process = fm(context)  # bind context -> FlowMatchingProcess
samples = process.sample((n,))  # generate samples
```

This helps separate what the model is from what context it operates on.

### Time Convention

nami uses the following time convention:

- `t=0` correpsonds to the target distribution (e.g. data)
- `t=1`: corresponds to the source distribution (e.g. noise)
- sampling integrates from `t=1` to `t=0`
- likelihood computation integrates instead from `t=0` to `t=1`

## A quick breakdown of the components in `nami`

> ToDo: propagate to a docs page.

Here's a non-exhaustive list of what's currently available:

#### Distributions

| Class | Description |
|-------|-------------|
| `StandardNormal` | N(0, I) with specified event_shape |
| `DiagonalNormal` | N(loc, scale) with diagonal covariance |

#### Solvers

| Class | Type | Description |
|-------|------|-------------|
| `RK4` | ODE | 4th-order Runge-Kutta, fixed-step |
| `Heun` | ODE | 2nd-order Heun's method, fixed-step |
| `EulerMaruyama` | SDE | Stochastic Euler-Maruyama |

#### Schedules (for diffusion)

| Class | Description |
|-------|-------------|
| `VPSchedule` | Variance Preserving (DDPM-style) [Ho et al.](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)|
| `VESchedule` | Variance Exploding (SMLD-style) [Song et al. 2021](https://openreview.net/pdf?id=St1giarCHLP), [Song et al. 2020](https://arxiv.org/abs/2011.13456)|
| `EDMSchedule` | EDM schedule [Karras et al.](https://arxiv.org/abs/2206.00364)|

#### Paths (for flow matching)

| Class | Description |
|-------|-------------|
| `LinearPath` | Linear interpolation: $x_t = (1-t) x_{target} + tx_{source}$ |
| `CosinePath` | Cosine interpolation: $x_t = \cos( \pi t/2)x_{target} + \sin(\pi t/2)x_{source}$ |

#### Divergence Estimators

For likelihood computation via continuous normalising flows:

| Class | Description |
|-------|-------------|
| `ExactDivergence` | Exact Jacobian trace (small dimensions only!) |
| `HutchinsonDivergence` | Stochastic trace estimation (scalable) |

```python
# compute log-likelihood
log_prob = process.log_prob(x, estimator=HutchinsonDivergence())
```

For more information on the Hutchinson Trace estimator, you can take a look at[Hutchinson, 1990](https://www.researchgate.net/publication/245083270_A_stochastic_estimator_of_the_trace_of_the_influence_matrix_for_Laplacian_smoothing_splines)

and the following example from the `BackPack` library which nicely demonstrates the accuracy of the trace estimation and techniques for computational speedups [BackPack Example](https://docs.backpack.pt/en/master/use_cases/example_trace_estimation.html), which is where the above reference is taken from.

## Custom Components

In this section, we'll go over how to add your own custom components

#### Custom Path

You can implement a `ProbabilityPath` by simply defining two methods, a `sample_xt` and a `target_ut`.

```python
from nami.paths.base import ProbabilityPath

class MyPath(ProbabilityPath):
    def sample_xt(self, x_target, x_source, t):
        # return interpolated position at time t
        return x_t

    def target_ut(self, x_target, x_source, t):
        # return target velocity at time t
        return u_t
```

Use it with flow matching:

```python
from nami import fm_loss

path = MyPath()
loss = fm_loss(field, x_target, x_source, path=path)
```

#### Custom Solver

You can implement `integrate(f, x0, t0, t1)` for ODE solvers or `integrate(drift, diffusion, x0, t0, t1, steps)` for SDE solvers.

```python
class MySolver:
    is_sde = False  # set True for SDE solvers

    def __init__(self, steps=32):
        self.steps = steps

    def integrate(self, f, x0, t0=0.0, t1=1.0, steps=None):
        # integrate dx/dt = f(x, t) from t0 to t1
        # x0: initial state
        # f: vector field function f(x, t) -> dx/dt
        # Return: x(t1)
        return x1
```

Set the following class attributes:
- `is_sde = True` for SDE solvers
- `requires_steps = True` if solver requires explicit step count
- `supports_rsample = True` if solver supports reparameterisation

#### Custom Schedule

You can implement a custom `NoiseSchedule` by defining:

```python
from nami.schedules.base import NoiseSchedule

class MySchedule(NoiseSchedule):
    def alpha(self, t):
        # signal coefficient at time t
        return alpha_t

    def sigma(self, t):
        # soise coefficient at time t
        return sigma_t

    def drift(self, x, t):
        # drift term dx = drift(x, t) dt
        return drift_x

    def diffusion(self, t):
        # diffusion coefficient g(t)
        return g_t
```

### General Utilities


```python
from nami.core.specs import (
    flatten_event,      # flatten event dimensions
    unflatten_event,    # restore event dimensions
    validate_shapes,    # validate tensor shapes
)

# flatten event dimensions
x = torch.randn(2, 3, 4, 5)
x_flat = flatten_event(x, event_ndim=2)  # (2, 3, 20)

# restore event dimensions
x_restored = unflatten_event(x_flat, event_shape=(4, 5))  # (2, 3, 4, 5)

# validate shapes (this will raise a ValueError if their is a mismatch)
validate_shapes(x, event_ndim=2, expected_event_shape=(4, 5), batch_shape=(2, 3))
```

## Diagnostics

nami also has a couple of methods for quick diagnostics of the trained field

```python
from nami import diagnostics

# field output statistics
stats = diagnostics.field_stats(field, x, t)
print(f"Field norm: {stats['mean']:.3f} +/- {stats['std']:.3f}")

# numerical reversibility check
err = diagnostics.reversibility_error(field, solver, x)
print(f"Round-trip error: {err['mean']:.2e}")
```

## Design Principles

nami is opinionated about a few things. Shapes are always explicit so there is no silent broadcasting across event dimensions. Components are pluggable, so you can swap solvers, schedules, and paths without changing model code. Dependencies are kept minimal: the core requires only `PyTorch` (but this may grow slightly).
