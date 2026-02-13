Overview
========

Nami is a modular research library for flow matching and diffusion-style
generative modelling. The library is built around small, composable abstractions; paths, losses, solvers, schedules, and processes that can be freely combined. Conditional and unconditional workflows are supported through lazy binding, and multiple parameterisations (velocity, score, drift, and noise wrappers) are interoperable out of the box.


Core Concepts
=============

The core concepts underpinning the library consist of shape and time conventions plus context binding logic.

Shape convention
----------------

All tensors follow ``sample_shape + batch_shape + event_shape``. The ``sample_shape`` correpsonds to independent draws, ``batch_shape`` to the parallel compuations and ``event_shape`` to a single data point. The ``event_ndim`` property tells nami how many trailing dimensions form one sample.

Lazy binding
------------

Nami separates model configuration from context binding, where models are defined once then bound to specific contexts, for example:

.. code-block:: python

   import nami

   fm = FlowMatching(field, base, solver)  # configuration
   process = fm(context)  # bind context -> FlowMatchingProcess
   samples = process.sample((n,))  # generate samples

This helps separate what the model is from what context it operates on. The pattern supports both unconditional and conditional workflows with the same process interface.

Time convention
---------------

Nami uses the following time convention;``t = 0`` for the target/data distribution and ``t = 1`` for the source/noise distribution. As a result, sampling generally integrates from ``t=1`` to ``t=0`` and Likelihood-style forward mapping integrates from ``t=0`` to ``t=1``.


Main workflows
--------------

Deterministic flow matching is handled by ``nami.fm_loss`` paired with the ``nami.FlowMatching`` sampler. Stochastic flow matching uses ``nami.stochastic_fm_loss`` with class-based gamma schedules. Diffusion processes are supported via ``nami.Diffusion`` with VP, VE, and EDM schedules.

Where to go next
----------------

- :doc:`quickstart` for setup and first training/sampling examples.
- :doc:`components` for a quick component catalog.
- :doc:`api/index` for complete module-level API docs.
