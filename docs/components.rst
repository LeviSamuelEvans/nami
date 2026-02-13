Component Catalog
=================

This page gives a quick map of the main public building blocks.

Distributions
-------------

.. list-table::
   :header-rows: 1

   * - Component
     - Description
   * - ``StandardNormal``
     - Isotropic Gaussian ``N(0, I)`` with explicit ``event_shape``.
   * - ``DiagonalNormal``
     - Gaussian with diagonal covariance.

Solvers
-------

.. list-table::
   :header-rows: 1

   * - Component
     - Type
     - Description
   * - ``RK4``
     - ODE
     - Fourth-order Runge-Kutta (fixed-step).
   * - ``Heun``
     - ODE
     - Second-order predictor-corrector method (fixed-step).
   * - ``EulerMaruyama``
     - SDE
     - Stochastic Euler-Maruyama integrator.

Diffusion schedules
-------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Description
   * - ``VPSchedule``
     - Variance-preserving schedule (DDPM-style) [1].
   * - ``VESchedule``
     - Variance-exploding schedule (SMLD-style) [2].
   * - ``EDMSchedule``
     - EDM schedule variant [3].

Flow-matching paths
-------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Description
   * - ``LinearPath``
     - Linear interpolation between target and source samples.
   * - ``CosinePath``
     - Trigonometric interpolation with smooth endpoint behavior.

Stochastic FM gamma schedules
-----------------------------

From M.Albergo et al. [4,5]

.. list-table::
   :header-rows: 1

   * - Component
     - Description
   * - ``ZeroGamma``
     - Deterministic fallback with zero stochastic term.
   * - ``BrownianGamma``
     - Brownian-bridge style gamma schedule.
   * - ``ScaledBrownianGamma``
     - Brownian gamma with positive scale multiplier.

Losses
------

.. list-table::
   :header-rows: 1

   * - Component
     - Description
   * - ``fm_loss``
     - Deterministic flow matching objective.
   * - ``stochastic_fm_loss``
     - Stochastic flow matching objective with gamma noise.
   * - ``masked_fm_loss``
     - Flow matching objective with variable-cardinality masking.

Parameterisation transforms
---------------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Description
   * - ``ScoreFromNoise``
     - Converts noise prediction into score prediction.
   * - ``DriftFromVelocityScore``
     - Combines velocity and score into SDE drift.
   * - ``MirrorVelocityFromScore``
     - Builds mirror velocity from score and gamma schedule.

Divergence estimators
---------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Description
   * - ``ExactDivergence``
     - Exact Jacobian trace; practical in low dimensions.
   * - ``HutchinsonDivergence``
     - Stochastic trace estimator for scalable likelihood computation.

For more information on the Hutchinson Trace estimator, you can take a look at [5]

and the following example from the `BackPack` library which nicely demonstrates the accuracy of the trace estimation and techniques for computational speedups `BackPack Example <https://docs.backpack.pt/en/master/use_cases/example_trace_estimation.html>`_, which is where the above reference is taken from.


References
----------

.. [1] Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
       https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

.. [2] Song et al., *Score-Based Generative Modeling through Stochastic
       Differential Equations*, ICLR 2021. https://openreview.net/pdf?id=St1giarCHLP

       Song et al., *Denoising Diffusion Implicit Models*, 2020.
       https://arxiv.org/abs/2011.13456

.. [3] Karras et al., *Elucidating the Design Space of Diffusion-Based
       Generative Models*, 2022. https://arxiv.org/abs/2206.00364

.. [4] Albergo, M. S. and Vanden-Eijnden, E., *Building Normalizing Flows with
       Stochastic Interpolants*, ICLR 2023. https://arxiv.org/abs/2209.15571

.. [5] Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E., *Stochastic
       Interpolants: A Unifying Framework for Flows and Diffusions*, arXiv 2023.
       https://arxiv.org/abs/2303.08797
  
.. [6] Hutchinson, M. F., *A Stochastic Estimator of the Trace of the
       Influence Matrix for Laplacian Smoothing Splines*, 1990.
       https://www.researchgate.net/publication/245083270_A_stochastic_estimator_of_the_trace_of_the_influence_matrix_for_Laplacian_smoothing_splines
