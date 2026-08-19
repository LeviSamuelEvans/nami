r"""Rung-3 toy: a Green's operator across 1-D Gaussian-mixture densities.

The experiment the learned-Green's-operator proposal demands: train ONE
operator ``(rho, s) -> phi`` over a *density-varying* task family (not a
single theta-family, which would let the encoder rediscover theta), then
evaluate the transport velocity ``v = d phi / dx`` on held-out densities
against the closed-form truth.

Task family
-----------
Random 1-D Gaussian mixtures ``p = sum_i w_i N(mu_i, sig_i)`` with
``n_comp in {1..4}`` at train time and a *held-out* ``n_comp = 6`` at
eval.  Each task perturbs one random component ``j`` in one of two ways;
both give closed-form source and exact minimum-energy velocity via
``rho v = -int_{-inf}^x d_theta p``:

- location (``theta = mu_j``):
  ``s = w_j N_j (x - mu_j) / sig_j^2 / p``,  ``v* = w_j N_j / p``.
- log-scale (``theta = log sig_j``):
  ``s = w_j N_j (z^2 - 1) / p``,  ``v* = w_j (x - mu_j) N_j / p``.

Baselines (per the spec): a separately optimised per-task Ritz MLP, and
a fixed Galerkin basis (linear + Gaussian RBFs) solved as ``G c = b``.

Metric: relative L2(rho) velocity error on fresh samples,
``sqrt(E[(v_hat - v*)^2] / E[v*^2])``.

Run:  pixi run python books/experiment/green_operator/toy_green_operator.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from nami.fields.green_operator import EmpiricalTangent, GreenOperatorPotentialField
from nami.losses.green_operator import green_operator_loss

HERE = Path(__file__).parent
DEVICE = "cpu"
MAX_COMP = 6


# --------------------------------------------------------------------------
# Task family: random mixtures with closed-form source and exact velocity
# --------------------------------------------------------------------------


def sample_tasks(n_tasks: int, n_comp_range: tuple[int, int], gen: torch.Generator):
    """Sample mixture parameters, padded to MAX_COMP with zero weights."""
    lo, hi = n_comp_range
    n_comp = torch.randint(lo, hi + 1, (n_tasks,), generator=gen)
    comp_idx = torch.arange(MAX_COMP)
    valid = comp_idx[None, :] < n_comp[:, None]

    raw_w = torch.rand(n_tasks, MAX_COMP, generator=gen).clamp_min(0.1) * valid
    w = raw_w / raw_w.sum(-1, keepdim=True)
    mu = (torch.rand(n_tasks, MAX_COMP, generator=gen) * 6.0 - 3.0) * valid
    sig = torch.where(
        valid, 0.5 + torch.rand(n_tasks, MAX_COMP, generator=gen), torch.ones(1)
    )
    # perturbed component: uniform over the valid ones
    j = (torch.rand(n_tasks, generator=gen) * n_comp).long().clamp(max=MAX_COMP - 1)
    # perturbation type: 0 = location, 1 = log-scale
    kind = torch.randint(0, 2, (n_tasks,), generator=gen)
    return {"w": w, "mu": mu, "sig": sig, "j": j, "kind": kind}


def _normal_pdf(x, mu, sig):
    z = (x - mu) / sig
    return torch.exp(-0.5 * z * z) / (sig * math.sqrt(2 * math.pi))


def mixture_sample(tasks, n_points: int, gen: torch.Generator) -> torch.Tensor:
    """Sample (n_tasks, n_points) from each task's mixture."""
    w = tasks["w"]
    comp = torch.multinomial(w, n_points, replacement=True, generator=gen)
    mu = torch.gather(tasks["mu"], 1, comp)
    sig = torch.gather(tasks["sig"], 1, comp)
    eps = torch.randn(w.shape[0], n_points, generator=gen)
    return mu + sig * eps


def source_and_exact_velocity(tasks, x: torch.Tensor):
    """Closed-form source s(x) and exact velocity v*(x); x is (n_tasks, n)."""
    w, mu, sig = tasks["w"], tasks["mu"], tasks["sig"]
    comps = _normal_pdf(x.unsqueeze(-1), mu.unsqueeze(1), sig.unsqueeze(1))
    p = (w.unsqueeze(1) * comps).sum(-1).clamp_min(1e-300)

    j = tasks["j"]
    w_j = w.gather(1, j[:, None])  # (T, 1)
    mu_j = tasks["mu"].gather(1, j[:, None])
    sig_j = tasks["sig"].gather(1, j[:, None])
    n_j = _normal_pdf(x, mu_j, sig_j)
    z = (x - mu_j) / sig_j

    loc = tasks["kind"][:, None] == 0
    source = torch.where(
        loc,
        w_j * n_j * z / sig_j / p,
        w_j * n_j * (z * z - 1.0) / p,
    )
    v_exact = torch.where(loc, w_j * n_j / p, w_j * (x - mu_j) * n_j / p)
    return source, v_exact


def make_batch(tasks, n_context: int, n_query: int, gen: torch.Generator):
    """Independent context and query sets for one batch of tasks."""
    x_ctx = mixture_sample(tasks, n_context, gen)
    s_ctx, _ = source_and_exact_velocity(tasks, x_ctx)
    x_qry = mixture_sample(tasks, n_query, gen)
    s_qry, _ = source_and_exact_velocity(tasks, x_qry)
    context = EmpiricalTangent(points=x_ctx.unsqueeze(-1), source=s_ctx)
    return context, x_qry.unsqueeze(-1), s_qry


def relative_l2_error(v_hat: torch.Tensor, v_exact: torch.Tensor) -> torch.Tensor:
    """Per-task relative L2(rho) error over fresh mixture samples."""
    num = (v_hat - v_exact).square().mean(-1)
    den = v_exact.square().mean(-1).clamp_min(1e-30)
    return (num / den).sqrt()


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def galerkin_velocity(x_fit, s_fit, x_eval, n_rbf: int = 24, ridge: float = 1e-6):
    """Fixed-basis Galerkin solve per task: linear + Gaussian RBF features."""
    out = torch.empty_like(x_eval)
    for t in range(x_fit.shape[0]):
        pts, src, qry = x_fit[t], s_fit[t], x_eval[t]
        qs = torch.linspace(0.02, 0.98, n_rbf)
        centers = torch.quantile(pts, qs)
        h = ((pts.max() - pts.min()) / n_rbf * 1.5).clamp_min(1e-3)

        def feats(xx):
            r = (xx[:, None] - centers[None, :]) / h
            rbf = torch.exp(-0.5 * r * r)
            drbf = -r / h * rbf
            psi = torch.cat([xx[:, None], rbf], 1)
            dpsi = torch.cat([torch.ones_like(xx)[:, None], drbf], 1)
            return psi, dpsi

        psi, dpsi = feats(pts)
        g = dpsi.T @ dpsi / pts.shape[0]
        b = psi.T @ src / pts.shape[0]
        c = torch.linalg.solve(g + ridge * torch.eye(g.shape[0]), b)
        _, dpsi_eval = feats(qry)
        out[t] = dpsi_eval @ c
    return out


class _RitzMLP(torch.nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        mods, d = [], 1
        for _ in range(layers):
            mods += [torch.nn.Linear(d, hidden), torch.nn.SiLU()]
            d = hidden
        mods.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x[:, None]).squeeze(-1)


def per_task_ritz_mlp_velocity(x_fit, s_fit, x_eval, steps: int = 800):
    """The per-task oracle baseline: one small MLP trained per task."""
    out = torch.empty_like(x_eval)
    for t in range(x_fit.shape[0]):
        torch.manual_seed(1234 + t)
        mlp = _RitzMLP()
        optim = torch.optim.Adam(mlp.parameters(), lr=1e-2)
        pts, src = x_fit[t], s_fit[t]
        for _ in range(steps):
            optim.zero_grad()
            xx = pts.detach().requires_grad_(True)
            phi = mlp(xx)
            (grad,) = torch.autograd.grad(phi.sum(), xx, create_graph=True)
            loss = (0.5 * grad.square() - src * phi).mean()
            loss.backward()
            optim.step()
        xe = x_eval[t].detach().requires_grad_(True)
        phi = mlp(xe)
        (grad,) = torch.autograd.grad(phi.sum(), xe)
        out[t] = grad.detach()
    return out


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------


def evaluate_operator(operator, tasks, n_context: int, n_eval: int, gen, *, normalise: bool = False):
    x_ctx = mixture_sample(tasks, n_context, gen)
    s_ctx, _ = source_and_exact_velocity(tasks, x_ctx)
    rms = (
        s_ctx.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        if normalise
        else torch.ones(s_ctx.shape[0], 1)
    )
    context = EmpiricalTangent(points=x_ctx.unsqueeze(-1), source=s_ctx / rms)
    x_eval = mixture_sample(tasks, n_eval, gen)
    _, v_exact = source_and_exact_velocity(tasks, x_eval)
    v_hat = rms * operator.velocity(
        x_eval.unsqueeze(-1), context, create_graph=False
    ).squeeze(-1)
    return relative_l2_error(v_hat, v_exact), (x_ctx, s_ctx, x_eval, v_exact, v_hat)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-modes", type=int, default=48)
    ap.add_argument("--density-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n-context", type=int, default=256)
    ap.add_argument("--n-query", type=int, default=256)
    ap.add_argument("--batch-tasks", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-comps", type=int, nargs=2, default=[1, 4])
    ap.add_argument("--normalise-source", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    n_context, n_query = args.n_context, args.n_query
    batch_tasks, steps = args.batch_tasks, args.steps

    operator = GreenOperatorPotentialField(
        1,
        n_modes=args.n_modes,
        density_dim=args.density_dim,
        hidden=args.hidden,
        layers=args.layers,
    )
    optim = torch.optim.Adam(operator.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps)

    print(f"training operator: {steps} steps x {batch_tasks} tasks")
    t0 = time.time()
    for step in range(steps):
        tasks = sample_tasks(batch_tasks, tuple(args.train_comps), gen)
        context, x_qry, s_qry = make_batch(tasks, n_context, n_query, gen)
        if args.normalise_source:
            rms = context.source.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            context = EmpiricalTangent(context.points, context.source / rms, context.mask)
            s_qry = s_qry / rms
        optim.zero_grad()
        loss = green_operator_loss(
            operator, context=context, query_x=x_qry, query_source=s_qry
        )
        loss.backward()
        optim.step()
        sched.step()
        if step % 250 == 0 or step == steps - 1:
            # held-out Ritz monitor (empirical-Ritz guard): fresh tasks, fresh queries
            with torch.enable_grad():
                mt = sample_tasks(64, tuple(args.train_comps), gen)
                mc, mx, ms = make_batch(mt, n_context, n_query, gen)
                monitor = green_operator_loss(
                    operator, context=mc, query_x=mx, query_source=ms,
                    create_graph=False,
                ).detach()
            print(
                f"  step {step:5d}  train {float(loss):+.4f}  "
                f"held-out {float(monitor):+.4f}  ({time.time() - t0:.0f}s)"
            )

    # ---------------- evaluation ----------------
    results = {}
    gen_eval = torch.Generator().manual_seed(999)

    tasks_in = sample_tasks(64, (1, 4), gen_eval)
    err_in, _ = evaluate_operator(operator, tasks_in, n_context, 2048, gen_eval, normalise=args.normalise_source)
    results["operator_in_dist"] = err_in.median().item()

    tasks_out = sample_tasks(64, (6, 6), gen_eval)
    err_out, detail = evaluate_operator(operator, tasks_out, n_context, 2048, gen_eval, normalise=args.normalise_source)
    results["operator_heldout_ncomp6"] = err_out.median().item()

    # baselines on a 16-task subset of the held-out family (cost)
    sub = {k: v[:16] for k, v in tasks_out.items()}
    x_fit = mixture_sample(sub, n_context + n_query, gen_eval)
    s_fit, _ = source_and_exact_velocity(sub, x_fit)
    x_eval = mixture_sample(sub, 2048, gen_eval)
    _, v_exact = source_and_exact_velocity(sub, x_eval)

    v_gal = galerkin_velocity(x_fit, s_fit, x_eval)
    results["galerkin_heldout"] = relative_l2_error(v_gal, v_exact).median().item()

    print("training per-task Ritz MLPs (16 tasks x 800 steps)...")
    v_mlp = per_task_ritz_mlp_velocity(x_fit, s_fit, x_eval)
    results["per_task_mlp_heldout"] = relative_l2_error(v_mlp, v_exact).median().item()

    rms16 = (
        s_fit[:, :n_context].square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        if args.normalise_source
        else torch.ones(x_fit.shape[0], 1)
    )
    sub_ctx = EmpiricalTangent(
        points=x_fit[:, :n_context].unsqueeze(-1),
        source=s_fit[:, :n_context] / rms16,
    )
    v_op = rms16 * operator.velocity(
        x_eval.unsqueeze(-1), sub_ctx, create_graph=False
    ).squeeze(-1)
    results["operator_heldout_same16"] = (
        relative_l2_error(v_op, v_exact).median().item()
    )

    print("\nmedian relative L2(rho) velocity error")
    for k, v in results.items():
        print(f"  {k:28s} {v:.4f}")
    (HERE / f"results{suffix}.json").write_text(json.dumps(results, indent=2))

    # ---------------- overlay plot ----------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    order = x_eval.argsort(dim=1)
    for ax, t in zip(axes, range(3)):
        idx = order[t]
        ax.plot(x_eval[t, idx], v_exact[t, idx], "k-", lw=2, label="exact $v^*$")
        ax.plot(x_eval[t, idx], v_op[t, idx], "C0--", lw=1.5, label="operator")
        ax.plot(x_eval[t, idx], v_gal[t, idx], "C1:", lw=1.5, label="Galerkin")
        ax.plot(x_eval[t, idx], v_mlp[t, idx], "C2-.", lw=1.2, label="per-task MLP")
        ax.set_title(f"held-out task {t} (6 comps)")
        ax.set_xlabel("x")
    axes[0].set_ylabel("velocity")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / f"overlay{suffix}.png", dpi=150)
    print(f"\nwrote results{suffix}.json and overlay{suffix}.png")


if __name__ == "__main__":
    main()
