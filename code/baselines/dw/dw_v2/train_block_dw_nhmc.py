#!/usr/bin/env python3
"""Train standard many-well NHMC with optional block-equivariant momenta."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from neural_nhmc_dw import (
    ContinuousNHMCKernel,
    ManyWellTarget,
    MixedSignKernel,
    UnderdampedNHMCKernel,
    ess_logz,
    rollout,
    rollout_underdamped,
)


def seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# Defaults for kernel flags added after some runs were already trained. The
# evaluator rebuilds the kernel from the run's own config.json, so reading these
# with getattr is what lets an old checkpoint still be evaluated; each default
# has to be the behaviour that config had before the flag existed.
_KERNEL_DEFAULTS = {
    "stagewise_epsilon": 0,
    "sigma_mode": "learned",
    "log_mass_init": -0.7,
    "learn_drift": 0,
    "drift_scale": 1.0,
    "anneal_floor": 1.0,
    "anneal_power": 1.0,
    "learn_schedule": 0,
    "score_injection": 0,
    "score_clip": 20.0,
    "pisgrad_score": 1,
    "mass_precond": "none",
    "stl": 0,
    "underdamped": 0,
    "friction_init": 0.5,
    "flip_score": 0,
    "fixed_momentum": 0,
}


def build_kernel(target: ManyWellTarget, args):
    def opt(name):
        return getattr(args, name, _KERNEL_DEFAULTS[name])

    base = ContinuousNHMCKernel(
        target,
        n_leapfrog=args.n_leapfrog,
        hidden=args.hidden,
        log_eps_init=args.log_eps_init,
        architecture=args.architecture,
        num_steps=args.num_steps,
        stagewise_epsilon=bool(opt("stagewise_epsilon")),
        sigma_mode=opt("sigma_mode"),
        log_mass_init=opt("log_mass_init"),
        learn_drift=bool(opt("learn_drift")),
        drift_scale=opt("drift_scale"),
        anneal_floor=opt("anneal_floor"),
        anneal_power=opt("anneal_power"),
        learn_schedule=bool(opt("learn_schedule")),
        score_injection=bool(opt("score_injection")),
        score_clip=opt("score_clip"),
        pisgrad_score=bool(opt("pisgrad_score")),
        mass_precond=opt("mass_precond"),
        stl=bool(opt("stl")),
        fixed_momentum=bool(opt("fixed_momentum")),
    )
    if opt("underdamped"):
        kernel = UnderdampedNHMCKernel(
            base, args.num_steps,
            friction_init=opt("friction_init"),
            use_flip=bool(args.use_flip),
        )
    else:
        flip_score = bool(getattr(args, "flip_score", _KERNEL_DEFAULTS["flip_score"]))
        kernel = (
            MixedSignKernel(base, args.num_steps, flip_score=flip_score)
            if args.use_flip else base
        )
    return base, kernel


def run_rollout(kernel, target, n_batch, num_steps, device, stagewise=False):
    """Dispatch to the augmented rollout when the momentum is carried."""
    if isinstance(kernel, UnderdampedNHMCKernel):
        return rollout_underdamped(kernel, target, n_batch, num_steps, device,
                                   stagewise=stagewise)
    return rollout(kernel, target, n_batch, num_steps, device, stagewise=stagewise)


def log_mean_exp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values, dim=0) - math.log(values.numel())


class FreeEnergyHead(torch.nn.Module):
    """NETS-style learned free-energy rate, one scalar per annealing stage.

    The stage work increments sum to the path work exactly, so E[w_t] is the
    free-energy difference across stage t. Regressing a learned A_t onto w_t
    and penalizing the residual asks each stage to be individually
    near-deterministic, which the global Var(W) does not: the global objective
    is happy to let one stage's fluctuation cancel another's.

    This only enters the loss. The sampler, the path work and the importance
    weights are untouched, so the estimator stays unbiased whatever A_t learns.
    """

    def __init__(self, num_steps: int):
        super().__init__()
        self.log_delta_f = torch.nn.Parameter(torch.zeros(num_steps))

    def residual(self, stage_work: torch.Tensor) -> torch.Tensor:
        return (stage_work - self.log_delta_f[:, None]).square().mean()


def bernoulli_score_term(work: torch.Tensor, log_r: torch.Tensor,
                        var_weight: float = 0.0, ess_weight: float = 0.0,
                        objective: str = "meanvar") -> torch.Tensor:
    """Score-function contribution for J, with zero loss value.

    The discrete flip is not pathwise-differentiable. Autograd through W at
    fixed s only gives E[∇W | s]. The missing piece is

        E[ c(W) ∇ log r ],

    where c(W) = W for J = E[W], and

        c(W) = W + λ (W² - 2 μ W),   μ = E[W]

    for J = E[W] + λ Var(W).  ``objective='lv'`` drops the leading W.
    ``ess_weight`` is ignored here (headline runs use 0). The surrogate does
    not change the logged scalar.
    """
    w = work.detach()
    mu = w.mean()
    if objective == "lv":
        coeff = w.square() - 2.0 * mu * w
    else:
        coeff = w + float(var_weight) * (w.square() - 2.0 * mu * w)
    extra = (coeff * log_r).mean()
    return extra - extra.detach()


def training_loss(work: torch.Tensor, var_weight: float, ess_weight: float,
                  objective: str = "meanvar"):
    mean = work.mean()
    variance = work.var(unbiased=False)
    # This is log(N / ESS). It directly penalizes heavy path weights while
    # retaining the path-RKL term as the leading objective.
    log_inverse_ess = log_mean_exp(-2.0 * work) - 2.0 * log_mean_exp(-work)
    if objective == "lv":
        # Log-variance loss: unchanged by a constant shift of the work, so the
        # unknown log-normalizer drops out of the gradient.
        loss = variance + ess_weight * log_inverse_ess
    else:
        loss = mean + var_weight * variance + ess_weight * log_inverse_ess
    return loss, log_inverse_ess


@torch.no_grad()
def validation_pools(kernel, target, args, device):
    pool_results = []
    devices = [device.index or 0] if device.type == "cuda" else []
    for eval_seed in args.eval_seeds:
        work_parts = []
        with torch.random.fork_rng(devices=devices):
            seed_all(eval_seed, device)
            remaining = args.eval_samples
            while remaining:
                batch = min(args.eval_batch, remaining)
                _, work = run_rollout(kernel, target, batch, args.num_steps, device)
                work_parts.append(work.detach().cpu().to(torch.float64))
                remaining -= batch
        all_work = torch.cat(work_parts)
        finite = torch.isfinite(all_work)
        work = all_work[finite]
        ess, log_z = ess_logz(work)
        normalized = torch.softmax(-work, dim=0)
        top_count = max(1, int(math.ceil(0.01 * work.numel())))
        pool_results.append(
            {
                "eval_seed": eval_seed,
                "normalized_ess": ess,
                "log_z_hat": log_z,
                "abs_log_z_error": abs(log_z - target.true_log_z()),
                "valid_fraction": float(finite.to(torch.float64).mean()),
                "work_std": float(work.std(unbiased=False)),
                "top1_weight_mass": float(torch.topk(normalized, top_count).values.sum()),
            }
        )
    ess_values = torch.tensor(
        [item["normalized_ess"] for item in pool_results], dtype=torch.float64
    )
    return {
        "selection_score": float(ess_values.min()),
        "median_ess": float(ess_values.median()),
        "pools": pool_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--fixed_momentum", type=int, default=0,
                        help="state-independent N(0, sigma_t^2) momentum laws (use --sigma_mode analytic)")
    parser.add_argument("--stl", type=int, default=0,
                        help="sticking-the-landing: score the drawn momentum "
                             "under a detached forward density")
    parser.add_argument("--underdamped", type=int, default=0,
                        help="carry the momentum across stages and refresh it "
                             "only partially")
    parser.add_argument("--friction_init", type=float, default=0.5,
                        help="initial momentum retention; 0 is a full refresh")
    parser.add_argument("--mass_precond", choices=("none", "scalar", "block"),
                        default="none",
                        help="stage-wise diagonal inverse mass in the position "
                             "update; 'block' splits the double-well and "
                             "Gaussian coordinates")
    parser.add_argument("--stage_var_weight", type=float, default=0.0,
                        help="NETS-style per-stage free-energy residual penalty "
                             "(0 disables the head)")
    parser.add_argument("--pisgrad_score", type=int, default=1,
                        help="keep the PIS-Grad target-score branch (0 ablates it)")
    parser.add_argument("--architecture",
                        choices=("mlp", "block_equivariant", "pisgrad"),
                        default="block_equivariant")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--n_leapfrog", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--target_lr", type=float, default=1.0e-5)
    parser.add_argument("--var_weight", type=float, default=0.02)
    parser.add_argument("--ess_weight", type=float, default=0.05)
    parser.add_argument("--loss", choices=("meanvar", "lv"), default="meanvar")
    parser.add_argument("--sigma_mode", choices=("analytic", "learned"),
                        default="analytic")
    parser.add_argument("--log_mass_init", type=float, default=-0.7)
    parser.add_argument("--learn_drift", type=int, default=0,
                        help="add a learned control u(x,t) to the force inside "
                             "every leapfrog step, not just at momentum refresh")
    parser.add_argument("--drift_scale", type=float, default=1.0)
    parser.add_argument("--anneal_floor", type=float, default=1.0,
                        help="1.0 keeps the original kernel, which applies the "
                             "full target force from the first stage; below 1.0 "
                             "ramps the force along a geometric path")
    parser.add_argument("--anneal_power", type=float, default=1.0)
    parser.add_argument("--learn_schedule", type=int, default=0)
    parser.add_argument("--score_injection", type=int, default=0,
                        help="PIS-Grad term: add a learned per-dimension, "
                             "time-only coefficient times the clipped score to "
                             "the momentum mean")
    parser.add_argument("--score_clip", type=float, default=20.0)
    parser.add_argument("--log_eps_init", type=float, default=-3.0)
    parser.add_argument("--stagewise_epsilon", type=int, default=1)
    parser.add_argument("--use_flip", type=int, default=1)
    parser.add_argument("--flip_score", type=int, default=0,
                        help="add the missing Bernoulli score-function term "
                             "W * grad log r for MixedSignKernel; 0 keeps the "
                             "current estimator so old checkpoints stay valid")
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=20000)
    parser.add_argument("--eval_batch", type=int, default=2048)
    parser.add_argument("--eval_seeds", default="81031,91033,101039")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()
    args.eval_seeds = [int(value) for value in args.eval_seeds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    owner_path = os.path.join(args.save_dir, ".slurm_owner")
    jobid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    if os.path.exists(owner_path):
        existing = open(owner_path).read().strip()
        if existing != jobid:
            print(f"skip train: {args.save_dir} owned by job {existing}", flush=True)
            return
    else:
        with open(owner_path, "w") as handle:
            handle.write(jobid)
    seed_all(args.seed, device)
    target = ManyWellTarget(args.dim)
    base, kernel = build_kernel(target, args)
    kernel = kernel.to(device)

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "target": "factorized FAB many-well",
            "true_log_z": target.true_log_z(),
            "n_parameters": sum(parameter.numel() for parameter in kernel.parameters()),
        }
    )
    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    if args.epochs <= 0:
        # Identity / untrained control: same kernel, init weights, no steps.
        torch.save(kernel.state_dict(), os.path.join(args.save_dir, "kernel_best.pt"))
        torch.save(kernel.state_dict(), os.path.join(args.save_dir, "kernel_final.pt"))
        summary = {
            "best_worst_pool_ess": None,
            "wall_seconds": 0.0,
            "history": [],
            "untrained": True,
            **config,
        }
        with open(os.path.join(args.save_dir, "training_summary.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print("untrained dump only; no training steps", flush=True)
        return

    # Kept outside the kernel so the checkpoint stays loadable by runs and
    # evaluators that know nothing about the head.
    free_energy_head = None
    if args.stage_var_weight > 0.0:
        free_energy_head = FreeEnergyHead(args.num_steps).to(device)

    trained_parameters = list(kernel.parameters())
    if free_energy_head is not None:
        trained_parameters += list(free_energy_head.parameters())
    optimizer = torch.optim.Adam(trained_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.target_lr
    )
    best_score = -1.0
    history = []
    start = time.time()
    print(
        f"DW{args.dim} architecture={args.architecture} steps={args.num_steps} "
        f"LF={args.n_leapfrog} batch={args.batch_size} device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        kernel.train()
        seed_all(args.seed * 1_000_003 + epoch, device)
        stage_work = None
        if free_energy_head is not None:
            _, work_all, stage_work = run_rollout(
                kernel, target, args.batch_size, args.num_steps, device,
                stagewise=True,
            )
        else:
            _, work_all = run_rollout(kernel, target, args.batch_size,
                                      args.num_steps, device)
        finite = torch.isfinite(work_all)
        work = work_all[finite]
        if work.numel() < 2:
            raise RuntimeError(f"non-finite training batch at epoch {epoch}")
        loss, log_inverse_ess = training_loss(
            work, args.var_weight, args.ess_weight, args.loss
        )
        if bool(getattr(args, "flip_score", 0)):
            consume = getattr(kernel, "consume_flip_logprob", None)
            log_r = consume() if callable(consume) else None
            if log_r is not None:
                loss = loss + bernoulli_score_term(
                    work, log_r[finite],
                    var_weight=args.var_weight,
                    ess_weight=args.ess_weight,
                    objective=args.loss,
                )
        stage_residual = 0.0
        if free_energy_head is not None:
            residual = free_energy_head.residual(stage_work[:, finite])
            loss = loss + args.stage_var_weight * residual
            stage_residual = float(residual)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(trained_parameters, 10.0))
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % args.log_interval == 0:
            step_sizes = [
                float(base.epsilon_at(step).detach())
                for step in range(1, args.num_steps + 1)
            ]
            print(
                f"[{epoch:4d}/{args.epochs}] loss={float(loss):.4f} "
                f"meanW={float(work.mean()):.4f} stdW={float(work.std(unbiased=False)):.4f} "
                f"logNoverESS={float(log_inverse_ess):.3f} "
                f"valid={float(finite.float().mean()):.4f} "
                f"eps=[{min(step_sizes):.5f},{max(step_sizes):.5f}] grad={gradient_norm:.3f}"
                + (f" stageRes={stage_residual:.3f}" if free_energy_head is not None else ""),
                flush=True,
            )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            kernel.eval()
            metrics = validation_pools(kernel, target, args, device)
            metrics["epoch"] = epoch
            history.append(metrics)
            print(
                f"  eval median ESS={100*metrics['median_ess']:.2f}% "
                f"worst ESS={100*metrics['selection_score']:.2f}%",
                flush=True,
            )
            if metrics["selection_score"] > best_score:
                best_score = metrics["selection_score"]
                torch.save(kernel.state_dict(), os.path.join(args.save_dir, "kernel_best.pt"))
                with open(
                    os.path.join(args.save_dir, "best_validation.json"), "w", encoding="utf-8"
                ) as handle:
                    json.dump(metrics, handle, indent=2)

    torch.save(kernel.state_dict(), os.path.join(args.save_dir, "kernel_final.pt"))
    summary = {
        "best_worst_pool_ess": best_score,
        "wall_seconds": time.time() - start,
        "history": history,
        **config,
    }
    with open(os.path.join(args.save_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"done best worst-pool ESS={100*best_score:.2f}%", flush=True)


if __name__ == "__main__":
    main()
