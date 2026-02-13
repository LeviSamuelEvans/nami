Quickstart
==========

Install
-------

From the project root:

.. code-block:: bash

   pixi run setup

If you are not using pixi:

.. code-block:: bash

   pip install -e .


Core deterministic flow matching
--------------------------------

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_target = torch.randn(32, 8)
   x_source = torch.randn_like(x_target)

   loss = nami.fm_loss(field, x_target, x_source)
   loss.backward()

Stochastic flow matching (linear-path MVP)
------------------------------------------

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_target = torch.randn(32, 8)
   x_source = torch.randn_like(x_target)

   loss = nami.stochastic_fm_loss(
       field,
       x_target,
       x_source,
       gamma=nami.BrownianGamma(),
   )

Deterministic parity with ``ZeroGamma``
---------------------------------------

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_target = torch.randn(32, 8)
   x_source = torch.randn_like(x_target)

   det = nami.fm_loss(field, x_target, x_source, reduction="none")
   stoch = nami.stochastic_fm_loss(
       field,
       x_target,
       x_source,
       gamma=nami.ZeroGamma(),
       reduction="none",
   )
   assert torch.allclose(det, stoch, atol=1e-6)

Conditional generation pattern
------------------------------

.. code-block:: python

   import torch
   from torch import nn
   import nami

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

       def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
           t_exp = t.unsqueeze(-1).expand(*x.shape[:-1], 1)
           inputs = [x, t_exp]
           if c is not None:
               inputs.append(c)
           return self.net(torch.cat(inputs, dim=-1))

   field = ConditionalField(dim=8, context_dim=4)
   base = nami.StandardNormal(event_shape=(8,))
   solver = nami.RK4(steps=32)
   fm = nami.FlowMatching(field, base, solver)

   context = torch.randn(16, 4)
   process = fm(context)
   samples = process.sample((1,))  # shape: (1, 16, 8)

Diffusion process
-----------------

Nami supports score-based diffusion models. The same field architecture can be reused, just swap in a noise schedule and an SDE solver.

.. code-block:: python

   import nami

   # same field architecture works
   model = nami.VelocityField(dim=8)
   schedule = nami.VPSchedule(beta_min=0.1, beta_max=20.0)
   solver = nami.EulerMaruyama(steps=100)

   diffusion = nami.Diffusion(
       model=model,
       schedule=schedule,
       solver=solver,
       parameterization="eps",  # or "score", "x0"
       event_shape=(8,),
   )

   process = diffusion(None)
   samples = process.sample((64,))

Transforms between parameterisations
------------------------------------

.. code-block:: python

   import nami

   score_model = nami.ScoreFromNoise(eta_model, nami.BrownianGamma())
   drift_model = nami.DriftFromVelocityScore(v_model, score_model, nami.BrownianGamma())
   mirror_velocity = nami.MirrorVelocityFromScore(score_model, nami.BrownianGamma())

Build and serve docs locally
----------------------------

.. code-block:: bash

   # one-shot HTML build
   pixi run docs

   # live docs server with auto rebuild
   pixi run docs-serve
