#!/usr/bin/env python3
"""HMC plus the global Z2 flip phi -> -phi for 2D lattice phi^4 (the HMC reference of the paper).

At zero external field the flip is an exact symmetry (always accepted), so each transition is one HMC
trajectory followed by a flip with probability 1/2. The step size is tuned once to a target acceptance and then
frozen. Observables (|M|, M^2, S/V, sign M), tau_int (tau = 1 + 2 sum rho, code/chainstats.py) and the
between-run statistics are those of the NHMC chain scripts. The output also holds long-run reference means and
variances pooled over runs.

Run from the repository root:
    python code/baselines/hmc_z2/phi4_hmc_z2.py --L 8 --kappa 0.27 --out hmc_L8_k027.json
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # code/

import chainstats as cs  # noqa: E402
from phi4 import phi4_action, phi4_force  # noqa: E402

KEYS = ('absM', 'M2', 'action', 'sign')


def observables(x, lam, kappa):
    V = x[0].numel()
    M = x.mean(dim=(1, 2, 3)).double()
    return {'absM': M.abs(), 'M2': M ** 2,
            'action': phi4_action(x.squeeze(1), lam=lam, kappa=kappa).double() / V, 'sign': torch.sign(M)}


@torch.no_grad()
def summarise(tr, acc, burn, ref, cost, acc_trace=None, relax=False):
    out = dict(acceptance=float(acc[burn:].mean()))
    if acc_trace is not None:
        out['holding'] = cs.holding_times(acc_trace[burn:])
    if relax:                                   # full traces from transition 1 (biased start)
        out['relaxation'] = {k: cs.relaxation_curve(tr[k]) for k in KEYS}
        out['relaxation']['frac_pos'] = cs.relaxation_curve((tr['sign'] > 0).astype(float))
    for k in KEYS:
        m, v = (ref.get(k, (None, None)) if ref else (None, None))
        out[k] = cs.observable_summary(tr[k][burn:], exact_mean=m, exact_var=v, cost_per_transition=cost)
    s = tr['sign'][burn:]
    out['sector_change_rate'] = float((s[1:] != s[:-1]).mean())
    out['frac_pos'] = float((s > 0).mean())
    series = {k: tr[k] for k in ('absM', 'M2', 'action', 'sign')}
    series['frac_pos'] = (tr['sign'] > 0).astype(float)
    out['burnin_check'] = cs.burnin_check(series, burn, early_keys=('absM', 'frac_pos'))
    return out


@torch.no_grad()
def hmc_z2_step(x, lam, kappa, eps, n_lf, flip=True):
    S = lambda z: phi4_action(z.squeeze(1), lam=lam, kappa=kappa)
    F = lambda z: phi4_force(z.squeeze(1), lam=lam, kappa=kappa).unsqueeze(1)
    p = torch.randn_like(x)
    h0 = S(x) + 0.5 * (p ** 2).sum((1, 2, 3))
    q, p = x.clone(), p + 0.5 * eps * F(x)
    for i in range(n_lf):
        q = q + eps * p
        if i < n_lf - 1:
            p = p + eps * F(q)
    p = p + 0.5 * eps * F(q)
    h1 = S(q) + 0.5 * (p ** 2).sum((1, 2, 3))
    la = torch.nan_to_num(h0 - h1, nan=-math.inf)
    a = torch.log(torch.rand_like(la)) < la
    x = torch.where(a.view(-1, 1, 1, 1), q, x)
    if flip:
        f = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(f.view(-1, 1, 1, 1), -x, x)
    return x, a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--L', type=int, default=8)
    ap.add_argument('--lam', type=float, default=0.022)
    ap.add_argument('--kappa', type=float, required=True)
    ap.add_argument('--n-lf', type=int, default=10)
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--target-acc', type=float, default=0.8)
    ap.add_argument('--n-chains', type=int, default=256)
    ap.add_argument('--force-budget', type=int, default=500_000 * 96, help='force evaluations per run')
    ap.add_argument('--n-runs', type=int, default=3)
    ap.add_argument('--burn-fraction', type=float, default=0.1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed - 1)
    sign = torch.where(torch.arange(a.n_chains) % 2 == 0, 1.0, -1.0).view(-1, 1, 1, 1)
    x = (sign + 0.3 * torch.randn(a.n_chains, 1, a.L, a.L)).to(dev)
    eps = 0.1
    for _ in range(500):
        x, acc = hmc_z2_step(x, a.lam, a.kappa, eps, a.n_lf, flip=not a.no_flip)
        eps *= math.exp(0.05 * (float(acc.double().mean()) - a.target_acc))
    fe = a.n_lf + 1
    n_trans = max(int(a.force_budget / (a.n_chains * fe)), 20)
    result = dict(args=vars(a), eps=eps, force_evals_per_transition=fe, runs=[])
    pooled = {kk: [] for kk in KEYS}
    for r in range(a.n_runs):
        seed = a.seed + 1000 * r
        torch.manual_seed(seed)
        xr = x.clone()
        tr = {kk: np.empty((n_trans, a.n_chains)) for kk in KEYS}
        acc = np.zeros(n_trans)
        t0 = time.time()
        for it in range(n_trans):
            xr, ah = hmc_z2_step(xr, a.lam, a.kappa, eps, a.n_lf, flip=not a.no_flip)
            acc[it] = float(ah.double().mean())
            for kk, v in observables(xr, a.lam, a.kappa).items():
                tr[kk][it] = v.cpu().numpy()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        sec = time.time() - t0
        burn = int(a.burn_fraction * n_trans)
        s = summarise(tr, acc, burn, None, dict(force_evals=fe, seconds=sec / n_trans))
        s.update(seconds=sec, n_transitions=n_trans, burn=burn)
        result['runs'].append(s)
        for kk in KEYS:
            pooled[kk].append(tr[kk][burn:].reshape(-1))
        print(f'[hmc+z2 k={a.kappa} run {r}] eps {eps:.4f} acc {s["acceptance"]:.3f} '
              + ' '.join(f'{kk} {s[kk]["mean"]:.5f} tau {s[kk]["tau_int"]:.2f}' for kk in KEYS), flush=True)
    result['reference'] = {kk: dict(mean=float(np.concatenate(pooled[kk]).mean()),
                                    var=float(np.concatenate(pooled[kk]).var())) for kk in KEYS}
    result['between_runs'] = {kk: cs.between_runs([rr[kk]['mean'] for rr in result['runs']],
                                                  pooled_var=result['reference'][kk]['var'],
                                                  cost_per_run=dict(force_evals=a.force_budget))
                              for kk in KEYS}
    json.dump(result, open(a.out, 'w'), indent=1, default=float)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
