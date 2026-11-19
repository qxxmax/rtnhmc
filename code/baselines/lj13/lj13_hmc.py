#!/usr/bin/env python3
"""Physical baseline on LJ13 (official potential): plain HMC with the exact force.

Chains start from the reference validation split; the step size is tuned once to a target acceptance and
frozen. Observables, reference and statistics are those of lj13_compare.py. Cost: L + 1 force evaluations
per transition and measured seconds on the same device. The force is clamped like the NHMC kernels'
(x-dependent only), so the leapfrog stays reversible and volume preserving and the MH test is exact.
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # code/ (chainstats, chain_ckpt)
sys.path.insert(0, str(HERE))
LJ_CODE = os.environ.get('LJ_CODE', str(HERE / 'particle'))
sys.path.insert(0, LJ_CODE)

import chainstats as cs  # noqa: E402
from lj13_compare import KEYS, observables  # noqa: E402
from particle_targets import make_target  # noqa: E402


@torch.no_grad()
def hmc_step(target, x, eps, L):
    U = lambda z: -target.log_prob(z.double())
    p = torch.randn_like(x)
    h0 = U(x) + 0.5 * (p.double() ** 2).sum(1)
    q = x.clone()
    p = p + 0.5 * eps * target.force(q)
    for i in range(L):
        q = q + eps * p
        if i < L - 1:
            p = p + eps * target.force(q)
    p = p + 0.5 * eps * target.force(q)
    h1 = U(q) + 0.5 * (p.double() ** 2).sum(1)
    la = torch.nan_to_num(h0 - h1, nan=-math.inf)
    a = torch.log(torch.rand_like(la)) < la
    return torch.where(a[:, None], q, x), a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, help='config.json of the NHMC run (target definition)')
    ap.add_argument('--refs', required=True)
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--target-acc', type=float, default=0.8)
    ap.add_argument('--n-chains', type=int, default=256)
    ap.add_argument('--force-budget', type=int, default=256 * 600 * 616)
    ap.add_argument('--n-runs', type=int, default=3)
    ap.add_argument('--burn-fraction', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = json.load(open(a.config))
    target = make_target(cfg)
    n = cfg['n_particles']
    val = torch.tensor(np.load(os.path.join(a.refs, 'val_split_LJ13-1000.npy')), dtype=torch.float32, device=dev)
    test = torch.tensor(np.load(os.path.join(a.refs, 'test_split_LJ13-1000.npy')), dtype=torch.float32, device=dev)
    ref = {k: (float(v.mean()), float(v.var())) for k, v in observables(test, target, n).items()}
    g = torch.Generator().manual_seed(a.seed - 1)
    x = val[torch.randperm(val.shape[0], generator=g)[:a.n_chains].to(dev)]
    eps = 0.005
    for _ in range(300):
        x, acc = hmc_step(target, x, eps, a.L)
        eps *= math.exp(0.05 * (float(acc.double().mean()) - a.target_acc))
    fe = a.L + 1
    n_trans = max(int(a.force_budget / (a.n_chains * fe)), 20)
    result = dict(args=vars(a), eps=eps, force_evals_per_transition=fe, reference=ref, runs=[])
    for r in range(a.n_runs):
        seed = a.seed + 1000 * r
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        x = val[torch.randperm(val.shape[0], generator=g)[:a.n_chains].to(dev)]
        tr = {k: np.empty((n_trans, a.n_chains)) for k in KEYS}
        acc = np.zeros(n_trans)
        t0 = time.time()
        for it in range(n_trans):
            x, ah = hmc_step(target, x, eps, a.L)
            acc[it] = float(ah.double().mean())
            for k, v in observables(x, target, n).items():
                tr[k][it] = v.cpu().numpy()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        sec = time.time() - t0
        burn = int(a.burn_fraction * n_trans)
        s = dict(acceptance=float(acc[burn:].mean()), seconds=sec, n_transitions=n_trans, burn=burn)
        for k in KEYS:
            s[k] = cs.observable_summary(tr[k][burn:], exact_mean=ref[k][0], exact_var=ref[k][1],
                                         cost_per_transition=dict(force_evals=fe, seconds=sec / n_trans))
        result['runs'].append(s)
        print(f'[hmc run {r}] eps {eps:.5f} acc {s["acceptance"]:.3f} '
              + ' '.join(f'{k} {s[k]["mean"]:.4f} tau {s[k]["tau_int"]:.1f}' for k in KEYS), flush=True)
    result['between_runs'] = {k: cs.between_runs([rr[k]['mean'] for rr in result['runs']], exact_var=ref[k][1],
                                                 cost_per_run=dict(force_evals=a.force_budget)) for k in KEYS}
    json.dump(result, open(a.out, 'w'), indent=1, default=float)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
