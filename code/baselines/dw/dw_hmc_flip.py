#!/usr/bin/env python3
"""Physical baseline on the v4 many-well targets: HMC plus sign-flip Metropolis moves.

One transition = one HMC trajectory (L leapfrog steps, unit mass, step size tuned once to a target
acceptance and then frozen) followed by `--n_flip` Metropolis sign flips of single double-well
coordinates (symmetric proposal; the target's 0.5 x tilt makes the flip a proposal, not a symmetry).
Observables, exact references and statistics are those of dw_compare.py, so the numbers are directly
comparable; cost is reported in force evaluations (L + 1 per trajectory) and seconds.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dw_v2'))
sys.path.insert(0, str(HERE.parents[1]))   # code/ (chainstats, chain_ckpt)
sys.path.insert(0, str(HERE))

import chainstats as cs  # noqa: E402
from dw_compare import exact_moments, mode_index, observables, summarise_chain  # noqa: E402
from neural_nhmc_dw import ManyWellTarget  # noqa: E402


@torch.no_grad()
def hmc_step(target, x, eps, L):
    p = torch.randn_like(x)
    h0 = -target.log_prob(x) + 0.5 * (p ** 2).sum(1)
    q = x.clone()
    p = p + 0.5 * eps * target.force(q)
    for i in range(L):
        q = q + eps * p
        if i < L - 1:
            p = p + eps * target.force(q)
    p = p + 0.5 * eps * target.force(q)
    h1 = -target.log_prob(q) + 0.5 * (p ** 2).sum(1)
    la = torch.nan_to_num(h0 - h1, nan=-math.inf)
    a = torch.log(torch.rand_like(la)) < la
    return torch.where(a[:, None], q, x), a


@torch.no_grad()
def flip_step(target, x):
    j = torch.randint(target.n_wells, (x.shape[0],), device=x.device)
    y = x.clone()
    idx = torch.arange(x.shape[0], device=x.device)
    y[idx, 2 * j] = -y[idx, 2 * j]
    la = target.log_prob(y) - target.log_prob(x)
    a = torch.log(torch.rand_like(la)) < la
    return torch.where(a[:, None], y, x), a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--n_flip', type=int, default=1, help='sign-flip proposals per transition (0 = plain HMC)')
    ap.add_argument('--target_acc', type=float, default=0.75)
    ap.add_argument('--n_chains', type=int, default=256)
    ap.add_argument('--force_budget', type=int, default=256 * 2000 * 80, help='force evaluations per run')
    ap.add_argument('--n_runs', type=int, default=3)
    ap.add_argument('--burn_fraction', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    target = ManyWellTarget(a.dim)
    n_wells = a.dim // 2
    exact, p_pos = exact_moments(n_wells)

    # step-size tuning on separate chains, then frozen
    torch.manual_seed(a.seed - 1)
    x = target.sample(a.n_chains, seed=a.seed - 1, dtype=torch.float64)
    eps = 0.1
    for it in range(400):
        x, acc = hmc_step(target, x, eps, a.L)
        eps *= math.exp(0.05 * (float(acc.double().mean()) - a.target_acc))
    fe_per_trans = a.L + 1
    n_trans = max(int(a.force_budget / (a.n_chains * fe_per_trans)), 20)
    result = dict(args=vars(a), eps=eps, force_evals_per_transition=fe_per_trans, exact=exact, runs=[])
    for r in range(a.n_runs):
        seed = a.seed + 1000 * r
        torch.manual_seed(seed)
        x = target.sample(a.n_chains, seed=seed, dtype=torch.float64)
        keys = ('sign1', 'frac_pos', 'abs_even', 'odd_sq', 'energy')
        traces = {k: np.empty((n_trans, a.n_chains)) for k in keys}
        modes = np.empty((n_trans, a.n_chains), dtype=np.int64)
        acc, facc, t0 = np.zeros(n_trans), [], time.time()
        for it in range(n_trans):
            x, ah = hmc_step(target, x, eps, a.L)
            acc[it] = float(ah.double().mean())
            for _ in range(a.n_flip):
                x, af = flip_step(target, x)
                facc.append(float(af.double().mean()))
            for k, v in observables(x, target).items():
                traces[k][it] = v.numpy()
            modes[it] = mode_index(x).numpy()
        sec = time.time() - t0
        burn = int(a.burn_fraction * n_trans)
        cost = dict(force_evals=fe_per_trans, seconds=sec / n_trans)
        s = summarise_chain(traces, modes, acc, burn, exact, p_pos, n_wells, cost)
        s.update(seconds=sec, n_transitions=n_trans, burn=burn,
                 flip_acceptance=float(np.mean(facc)) if facc else None)
        result['runs'].append(s)
        print(f'[hmc+flip run {r}] eps {eps:.4f} ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                                for k, v in s.items() if not isinstance(v, dict)}),
              flush=True)
    agg = {}
    for k in exact:
        agg[k] = cs.between_runs([rr[k]['mean'] for rr in result['runs']], exact_var=exact[k][1],
                                 cost_per_run=dict(force_evals=a.force_budget,
                                                   seconds=float(np.mean([rr['seconds'] for rr in result['runs']]))))
    result['between_runs'] = agg
    json.dump(result, open(a.out, 'w'), indent=1, default=float)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
