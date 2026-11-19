#!/usr/bin/env python3
"""NE-HMC baseline for the protocol-length scan: NHMC with every learned part switched off.
x0 ~ N(0, I); stage t = 1..K: p ~ N(0, I), n_lf leapfrog substeps of size eps with
    F_t(phi) = beta_t (-grad S_kappa) - (1 - beta_t) phi,   beta_t = t / K (linear protocol),
reverse law N(0, I) for -p. W = log q0(x0) + sum_t [log N(p_t) - log N(-p'_t)] + S(x_K) is the exact path ratio
(Crooks), same convention as exp_forward_eval.py (log Z = logsumexp(-W) - log n); no MH step, as in the trained kernel.
This is the 'purely stochastic' NE-MCMC counterpart of Bulgarelli, Cellini, Nada (arXiv:2412.00200) for our kernel.
Outputs one JSON list entry per eps (free_energy.summarize fields + K, eps, force evaluations per path)."""
import argparse
import json
import math
import os
import time

import torch

from phi4 import phi4_action, phi4_force
from free_energy import summarize

LOG2PI = math.log(2 * math.pi)


def lognormal01(z):
    return -0.5 * (z * z).sum(dim=(1, 2)) - 0.5 * z[0].numel() * LOG2PI


@torch.no_grad()
def paths(n, L, kappa, lam, K, nlf, eps):
    x = torch.randn(n, L, L, dtype=torch.float64)
    logP = lognormal01(x)
    for t in range(1, K + 1):
        b = t / K
        F = lambda z: b * phi4_force(z, lam=lam, kappa=kappa) - (1 - b) * z
        p = torch.randn_like(x)
        logP = logP + lognormal01(p)
        p = p + 0.5 * eps * F(x)
        for s in range(nlf - 1):
            x = x + eps * p
            p = p + eps * F(x)
        x = x + eps * p
        p = p + 0.5 * eps * F(x)
        logP = logP - lognormal01(-p)
    return x, logP + phi4_action(x, lam=lam, kappa=kappa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.022)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n-leapfrog", type=int, default=5)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25])
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = []
    for eps in a.eps:
        t0 = time.time(); torch.manual_seed(a.seed)
        Ws, Ms = [], []
        for _ in range(a.n // a.batch):
            x, W = paths(a.batch, a.L, a.kappa, a.lam, a.K, a.n_leapfrog, eps)
            Ws.append(W); Ms.append(x.mean(dim=(1, 2)))
        out = summarize(torch.cat(Ws), torch.cat(Ms), a.L * a.L)
        out.update(kernel="nehmc_baseline", kappa=a.kappa, L=a.L, stages=a.K, substeps=a.n_leapfrog, eps=eps,
                   force_evals_per_path=a.K * (a.n_leapfrog + 1), n=a.n, seed=a.seed, wall=time.time() - t0)
        res.append(out); print(json.dumps(out), flush=True)
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
