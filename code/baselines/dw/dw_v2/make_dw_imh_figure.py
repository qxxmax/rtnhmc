#!/usr/bin/env python3
"""Figure 2: raw path endpoints, path-IMH samples, and the exact target.

The point of the panel is that the raw endpoints carry the wrong mode masses
and the path-IMH correction moves them onto the target, so the three panels
have to share every plotting convention or the comparison is not a comparison.
Taken from the script that produced the published figure:

  * the pair plotted is coordinate 0 against coordinate 2, one from each well;
  * 96 bin edges spanning [-3.2, 3.2] in both directions, density normalized;
  * the white contours are the *same* exact-target histogram in all three
    panels, drawn at the 0.70, 0.86 and 0.94 quantiles of its positive values;
  * the chain panel shows the tail of the chain, the raw panel its head, so
    neither is a burn-in artifact.

The IMH chain runs over a fixed pool of independent paths, which is exactly
what the transition does -- the proposal does not depend on the current state,
so drawing from a pool and drawing fresh are the same chain.
"""
import argparse
import json
import math
import pathlib
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval_block_dw_nhmc import (  # noqa: E402
    exact_positive_probability,
    mode_probabilities,
)
from neural_nhmc_dw import ManyWellTarget, rollout  # noqa: E402
from train_block_dw_nhmc import build_kernel, seed_all  # noqa: E402

RANGE = (-3.2, 3.2)
N_BIN_EDGES = 96
CONTOUR_QUANTILES = (0.70, 0.86, 0.94)


def path_imh_from_pool(log_weights, n_steps, burn, seed):
    """Independence sampler over a fixed pool of paths."""
    rng = np.random.default_rng(seed)
    n = len(log_weights)
    # Start from the weighted empirical target so the panel shows the chain's
    # behaviour rather than its burn-in.
    probabilities = np.exp(log_weights - log_weights.max())
    probabilities /= probabilities.sum()
    current = int(rng.choice(n, p=probabilities))
    chain = np.empty(n_steps, dtype=np.int64)
    accept = np.zeros(n_steps, dtype=bool)
    for step in range(n_steps):
        proposal = int(rng.integers(n))
        log_alpha = min(0.0, float(log_weights[proposal] - log_weights[current]))
        if math.log(float(rng.random())) < log_alpha:
            current, accept[step] = proposal, True
        chain[step] = current
    return chain[burn:], float(accept[burn:].mean())


def draw_density(ax, x, bins, extent, target_hist, panel):
    hist, _, _ = np.histogram2d(x[:, 0], x[:, 2], bins=[bins, bins],
                                density=True)
    ax.imshow(hist.T, origin='lower', extent=extent, aspect='equal',
              cmap='magma')
    centers = 0.5 * (bins[:-1] + bins[1:])
    levels = np.quantile(target_hist[target_hist > 0], CONTOUR_QUANTILES)
    ax.contour(centers, centers, target_hist.T, levels=levels, colors='white',
               linewidths=0.45, alpha=0.55)
    ax.set_title(panel, fontsize=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_2$')


# Bit order used by mode_indices: well0 is the 1s bit, well1 the 2s bit.
MODE_KEYS = ((False, False), (True, False), (False, True), (True, True))


def mode_masses(x, weights=None):
    signs = x[:, 0::2] > 0
    if weights is None:
        weights = np.full(len(x), 1.0 / len(x))
    return {key: float(weights[np.all(signs[:, :2] == np.asarray(key)[None, :],
                                      axis=1)].sum())
            for key in MODE_KEYS}


def mode_l1(masses, exact):
    observed = np.array([masses[key] for key in MODE_KEYS], dtype=np.float64)
    return float(np.abs(observed - np.asarray(exact, dtype=np.float64)).sum())


def bootstrap_weighted_l1(pool, log_weights, exact, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pool)
    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        lw = log_weights[idx]
        weights = np.exp(lw - lw.max())
        weights /= weights.sum()
        values[i] = mode_l1(mode_masses(pool[idx], weights), exact)
    return {
        'mean': float(values.mean()),
        'std': float(values.std(ddof=1)),
        'q05': float(np.quantile(values, 0.05)),
        'q95': float(np.quantile(values, 0.95)),
        'n_boot': n_boot,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--checkpoint', default='kernel_best.pt')
    parser.add_argument('--samples', type=int, default=200000)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--imh_steps', type=int, default=200000)
    parser.add_argument('--imh_burn', type=int, default=20000)
    parser.add_argument('--panel_samples', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=271828)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    namespace = argparse.Namespace(**config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(args.seed, device)

    target = ManyWellTarget(namespace.dim)
    _, kernel = build_kernel(target, namespace)
    kernel.load_state_dict(torch.load(run_dir / args.checkpoint,
                                      map_location=device))
    kernel = kernel.to(device).eval()

    pool, works = [], []
    remaining = args.samples
    with torch.no_grad():
        while remaining:
            batch = min(args.batch, remaining)
            x, work = rollout(kernel, target, batch, namespace.num_steps,
                              device)
            pool.append(x.cpu().double())
            works.append(work.cpu().double())
            remaining -= batch
    pool = torch.cat(pool)
    work = torch.cat(works)

    finite = torch.isfinite(work) & torch.isfinite(pool).all(1)
    pool, work = pool.numpy()[finite.numpy()], work.numpy()[finite.numpy()]
    log_weights = -work

    chain, acceptance = path_imh_from_pool(log_weights, args.imh_steps,
                                           args.imh_burn, args.seed + 17)
    x_imh = pool[chain]
    x_target = target.sample(args.panel_samples, seed=args.seed + 101).numpy()

    bins = np.linspace(RANGE[0], RANGE[1], N_BIN_EDGES)
    extent = [RANGE[0], RANGE[1], RANGE[0], RANGE[1]]
    target_hist, _, _ = np.histogram2d(x_target[:, 0], x_target[:, 2],
                                       bins=[bins, bins], density=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), constrained_layout=True)
    draw_density(axes[0], pool[:args.panel_samples], bins, extent, target_hist,
                 '(a) raw path endpoints')
    draw_density(axes[1], x_imh[-args.panel_samples:], bins, extent,
                 target_hist, '(b) path-IMH samples')
    draw_density(axes[2], x_target, bins, extent, target_hist,
                 '(c) exact target')

    out = pathlib.Path(args.out) if args.out else run_dir / 'dw_imh_figure.png'
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)

    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    ess = float(np.exp(2 * np.logaddexp.reduce(log_weights)
                       - np.logaddexp.reduce(2 * log_weights))
                / len(log_weights))
    exact = mode_probabilities(2, exact_positive_probability()).numpy()
    raw_mass = mode_masses(pool)
    weighted_mass = mode_masses(pool, weights)
    imh_mass = mode_masses(x_imh)
    target_mass = mode_masses(x_target)
    summary = {
        'run_dir': str(run_dir),
        'dim': namespace.dim,
        'n_paths': int(len(log_weights)),
        'normalized_ess': ess,
        'log_z_hat': float(np.logaddexp.reduce(log_weights)
                           - math.log(len(log_weights))),
        'true_log_z': float(target.true_log_z()),
        'imh_acceptance': acceptance,
        'exact_mode_mass': exact.tolist(),
        'mode_mass_raw': {str(k): v for k, v in raw_mass.items()},
        'mode_mass_weighted': {str(k): v for k, v in weighted_mass.items()},
        'mode_mass_imh': {str(k): v for k, v in imh_mass.items()},
        'mode_mass_target': {str(k): v for k, v in target_mass.items()},
        'raw_mode_l1': mode_l1(raw_mass, exact),
        'weighted_mode_l1': mode_l1(weighted_mass, exact),
        'imh_mode_l1': mode_l1(imh_mass, exact),
        'target_sample_mode_l1': mode_l1(target_mass, exact),
        'weighted_mode_l1_bootstrap': bootstrap_weighted_l1(
            pool, log_weights, exact, seed=args.seed + 29),
        'figure': str(out),
    }
    (out.with_suffix('.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
