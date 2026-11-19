#!/usr/bin/env python3
"""Frozen-checkpoint Bernoulli sign-flip ablation for DW NHMC.

Purpose
-------
Test whether the *learned* Bernoulli flip schedule is needed at deployment.
The continuous NHMC parameters are kept frozen.  Only the per-stage flip
probability is intervened on:

    learned, 0, 0.05, 0.10, 0.15, ...

The path probability is still evaluated with the same forward/reverse
Bernoulli terms, so the reported SNIS/IMH quantities remain corrected for the
intervened proposal law.

Place this script in the same directory as ``neural_nhmc_dw.py`` and
``train_block_dw_nhmc.py`` (e.g. from_mainz/dw_v2/) or run it from there.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev

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
    p = torch.empty(count, dtype=torch.float64)
    for mode in range(count):
        npos = bin(mode).count("1")
        p[mode] = positive_probability ** npos * (1.0 - positive_probability) ** (n_wells - npos)
    return p / p.sum()


def mode_indices(samples: torch.Tensor) -> torch.Tensor:
    positive = samples[:, 0::2] > 0
    powers = 2 ** torch.arange(positive.shape[1], device=samples.device)
    return (positive.to(torch.long) * powers).sum(dim=1)


def stationary_imh_acceptance(log_weights: torch.Tensor, n_pairs: int, seed: int) -> float:
    g = torch.Generator(device="cpu").manual_seed(seed)
    probs = torch.softmax(log_weights, dim=0)
    current = torch.multinomial(probs, n_pairs, replacement=True, generator=g)
    proposed = torch.randint(log_weights.numel(), (n_pairs,), generator=g)
    log_alpha = torch.minimum(
        torch.zeros(n_pairs, dtype=log_weights.dtype),
        log_weights[proposed] - log_weights[current],
    )
    return float(torch.exp(log_alpha).mean())


def set_fixed_flip_probability(kernel, p: float) -> None:
    if not hasattr(kernel, "flip_logits"):
        raise RuntimeError("Checkpoint/kernel has no learned Bernoulli flip wrapper.")
    # p=0 and p=1 are represented by saturated finite logits. In float64,
    # sigmoid(±1000) is exactly 0/1 to machine precision; the implementation's
    # 1e-8 log guard keeps realized log-probabilities finite.
    if p <= 0.0:
        logit = -1000.0
    elif p >= 1.0:
        logit = 1000.0
    else:
        logit = math.log(p / (1.0 - p))
    with torch.no_grad():
        kernel.flip_logits.fill_(logit)


def learned_flip_stats(kernel) -> dict:
    if not hasattr(kernel, "flip_logits"):
        return {"learned_flip_mean": None, "learned_flip_min": None, "learned_flip_max": None}
    p = torch.sigmoid(kernel.flip_logits.detach()).cpu().to(torch.float64)
    return {
        "learned_flip_mean": float(p.mean()),
        "learned_flip_min": float(p.min()),
        "learned_flip_max": float(p.max()),
    }


def evaluate_one(run_dir: Path, checkpoint: str, condition: str, p_fixed: float | None,
                 samples_n: int, batch_size: int, seed: int, imh_pairs: int) -> dict:
    with open(run_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    ns = argparse.Namespace(**config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed, device)
    target = ManyWellTarget(ns.dim)
    _, kernel = build_kernel(target, ns)
    state = torch.load(run_dir / checkpoint, map_location=device)
    kernel.load_state_dict(state)
    kernel = kernel.to(device).eval()

    learned_stats = learned_flip_stats(kernel)
    if p_fixed is not None:
        set_fixed_flip_probability(kernel, p_fixed)

    sample_parts, work_parts = [], []
    remaining = samples_n
    with torch.no_grad():
        while remaining:
            b = min(batch_size, remaining)
            x, work = run_rollout(kernel, target, b, ns.num_steps, device)
            sample_parts.append(x.cpu().to(torch.float64))
            work_parts.append(work.cpu().to(torch.float64))
            remaining -= b

    x = torch.cat(sample_parts)
    work_all = torch.cat(work_parts)
    finite = torch.isfinite(work_all) & torch.isfinite(x).all(dim=1)
    x = x[finite]
    work = work_all[finite]
    logw = -work
    w = torch.softmax(logw, dim=0)
    ess_frac, logz = ess_logz(work)

    modes = mode_indices(x)
    n_modes = 2 ** target.n_wells
    raw_mass = torch.bincount(modes, minlength=n_modes).to(torch.float64)
    raw_mass /= raw_mass.sum()
    weighted_mass = torch.zeros(n_modes, dtype=torch.float64)
    weighted_mass.scatter_add_(0, modes, w)
    exact_mass = mode_probabilities(target.n_wells, exact_positive_probability())

    top_count = max(1, int(math.ceil(0.01 * work.numel())))
    out = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "condition": condition,
        "fixed_p": p_fixed,
        "seed": seed,
        "dim": int(ns.dim),
        "num_steps": int(ns.num_steps),
        "n_requested": int(samples_n),
        "n_valid": int(work.numel()),
        "valid_fraction": float(finite.to(torch.float64).mean()),
        "ess_fraction": float(ess_frac),
        "ess": float(ess_frac * work.numel()),
        "work_std": float(work.std(unbiased=False)),
        "log_z_hat": float(logz),
        "true_log_z": float(target.true_log_z()),
        "abs_log_z_error": abs(float(logz) - float(target.true_log_z())),
        "imh_acceptance": stationary_imh_acceptance(logw, imh_pairs, seed + 1),
        "raw_mode_coverage": int((raw_mass > 0).sum()),
        "weighted_mode_coverage": int((weighted_mass > 0).sum()),
        "mode_count": int(n_modes),
        "raw_mode_l1": float(torch.abs(raw_mass - exact_mass).sum()),
        "weighted_mode_l1": float(torch.abs(weighted_mass - exact_mass).sum()),
        "top1pct_weight_mass": float(torch.topk(w, top_count).values.sum()),
        **learned_stats,
    }
    return out


def parse_probs(text: str) -> list[float]:
    vals = []
    for s in text.split(","):
        s = s.strip()
        if not s:
            continue
        p = float(s)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"invalid probability {p}")
        vals.append(p)
    return vals


def parse_seeds(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = [
        "ess_fraction", "work_std", "abs_log_z_error", "imh_acceptance",
        "raw_mode_l1", "weighted_mode_l1", "top1pct_weight_mass",
    ]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["condition"], []).append(r)
    out = []
    for cond, rs in groups.items():
        row = {
            "condition": cond,
            "fixed_p": rs[0]["fixed_p"],
            "replicates": len(rs),
            "mode_coverage_min": min(r["weighted_mode_coverage"] for r in rs),
            "mode_count": rs[0]["mode_count"],
        }
        for m in metrics:
            vals = [float(r[m]) for r in rs]
            row[m + "_mean"] = mean(vals)
            row[m + "_sd"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", default="kernel_best.pt")
    ap.add_argument("--samples", type=int, default=120000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seeds", default="271828,314159,161803")
    ap.add_argument("--probs", default="0,0.05,0.10,0.15,0.25,0.50")
    ap.add_argument("--imh-pairs", type=int, default=200000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "flip_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = parse_probs(args.probs)
    seeds = parse_seeds(args.seeds)
    rows = []
    for seed in seeds:
        rows.append(evaluate_one(
            run_dir, args.checkpoint, "learned", None,
            args.samples, args.batch, seed, args.imh_pairs,
        ))
        for p in probs:
            rows.append(evaluate_one(
                run_dir, args.checkpoint, f"fixed_{p:g}", p,
                args.samples, args.batch, seed, args.imh_pairs,
            ))

    raw_csv = out_dir / "flip_ablation_replicates.csv"
    write_csv(raw_csv, rows)
    agg = aggregate(rows)
    agg_csv = out_dir / "flip_ablation_summary.csv"
    write_csv(agg_csv, agg)

    payload = {
        "purpose": "Frozen continuous model; intervene only on Bernoulli sign-flip probability.",
        "interpretation": (
            "If a prescribed fixed flip probability retains mode coverage and gives ESS/IMH/mode-L1 "
            "comparable to the learned schedule, learned Bernoulli logits are not needed for deployment. "
            "A subsequent fixed-p retraining is then the clean final check."
        ),
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "samples_per_condition_per_seed": args.samples,
        "seeds": seeds,
        "fixed_probabilities": probs,
        "summary": agg,
    }
    with open(out_dir / "flip_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote {raw_csv}")
    print(f"wrote {agg_csv}")
    print(f"wrote {out_dir / 'flip_ablation_summary.json'}")
    for r in agg:
        print(
            f"{r['condition']:>12s}  ESS/N={r['ess_fraction_mean']:.4f}  "
            f"IMH={r['imh_acceptance_mean']:.4f}  "
            f"modeL1={r['weighted_mode_l1_mean']:.4g}  "
            f"coverage={r['mode_coverage_min']}/{r['mode_count']}"
        )


if __name__ == "__main__":
    main()
