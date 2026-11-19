#!/usr/bin/env python3
"""Evaluate a block-equivariant many-well NHMC checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

from neural_nhmc_dw import ManyWellTarget, ess_logz
from train_block_dw_nhmc import build_kernel, run_rollout, seed_all


def exact_positive_probability(dtype=torch.float64) -> float:
    grid = torch.linspace(-8.0, 8.0, 400001, dtype=dtype)
    density = torch.exp(-grid.pow(4) + 6.0 * grid.pow(2) + 0.5 * grid)
    total = torch.trapz(density, grid)
    positive = torch.trapz(density[grid >= 0], grid[grid >= 0])
    return float(positive / total)


def mode_probabilities(n_wells: int, positive_probability: float) -> torch.Tensor:
    count = 2 ** n_wells
    probabilities = torch.empty(count, dtype=torch.float64)
    for mode in range(count):
        positive = bin(mode).count('1')
        probabilities[mode] = (
            positive_probability ** positive
            * (1.0 - positive_probability) ** (n_wells - positive)
        )
    return probabilities / probabilities.sum()


def mode_indices(samples: torch.Tensor) -> torch.Tensor:
    positive = samples[:, 0::2] > 0
    powers = 2 ** torch.arange(positive.shape[1], device=samples.device)
    return (positive.to(torch.long) * powers).sum(dim=1)


def stationary_imh_acceptance(log_weights: torch.Tensor, n_pairs: int, seed: int) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probabilities = torch.softmax(log_weights, dim=0)
    current = torch.multinomial(probabilities, n_pairs, replacement=True, generator=generator)
    proposed = torch.randint(log_weights.numel(), (n_pairs,), generator=generator)
    log_alpha = torch.minimum(
        torch.zeros(n_pairs, dtype=log_weights.dtype),
        log_weights[proposed] - log_weights[current],
    )
    return float(torch.exp(log_alpha).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default="kernel_best.pt")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--imh_pairs", type=int, default=1000000)
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    namespace = argparse.Namespace(**config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed, device)
    target = ManyWellTarget(namespace.dim)
    _, kernel = build_kernel(target, namespace)
    state = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location=device)
    kernel.load_state_dict(state)
    kernel = kernel.to(device).eval()

    sample_parts = []
    work_parts = []
    remaining = args.samples
    with torch.no_grad():
        while remaining:
            batch = min(args.batch, remaining)
            samples, work = run_rollout(kernel, target, batch,
                                        namespace.num_steps, device)
            sample_parts.append(samples.cpu().to(torch.float64))
            work_parts.append(work.cpu().to(torch.float64))
            remaining -= batch
    samples = torch.cat(sample_parts)
    work_all = torch.cat(work_parts)
    finite = torch.isfinite(work_all) & torch.isfinite(samples).all(dim=1)
    samples = samples[finite]
    work = work_all[finite]
    log_weights = -work
    normalized_weights = torch.softmax(log_weights, dim=0)
    ess, log_z = ess_logz(work)
    top_count = max(1, int(math.ceil(0.01 * work.numel())))

    modes = mode_indices(samples)
    count = 2 ** target.n_wells
    raw_mode_mass = torch.bincount(modes, minlength=count).to(torch.float64)
    raw_mode_mass /= raw_mode_mass.sum()
    weighted_mode_mass = torch.zeros(count, dtype=torch.float64)
    weighted_mode_mass.scatter_add_(0, modes, normalized_weights)
    exact_mode_mass = mode_probabilities(target.n_wells, exact_positive_probability())

    summary = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "architecture": namespace.architecture,
        "dim": namespace.dim,
        "n_paths_requested": args.samples,
        "n_paths_valid": int(work.numel()),
        "valid_fraction": float(finite.to(torch.float64).mean()),
        "normalized_ess": ess,
        "log_z_hat": log_z,
        "true_log_z": target.true_log_z(),
        "abs_log_z_error": abs(log_z - target.true_log_z()),
        "work_std": float(work.std(unbiased=False)),
        "top1_weight_mass": float(torch.topk(normalized_weights, top_count).values.sum()),
        "stationary_path_imh_acceptance": stationary_imh_acceptance(
            log_weights, args.imh_pairs, args.seed + 1
        ),
        "mode_coverage_raw": int((raw_mode_mass > 0).sum()),
        "mode_coverage_weighted": int((weighted_mode_mass > 0).sum()),
        "mode_count": count,
        "raw_mode_l1": float(torch.abs(raw_mode_mass - exact_mode_mass).sum()),
        "weighted_mode_l1": float(torch.abs(weighted_mode_mass - exact_mode_mass).sum()),
        "raw_mode_mass": raw_mode_mass.tolist(),
        "weighted_mode_mass": weighted_mode_mass.tolist(),
        "exact_mode_mass": exact_mode_mass.tolist(),
    }
    output = os.path.join(args.run_dir, "final_evaluation.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
