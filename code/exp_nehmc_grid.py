#!/usr/bin/env python3
"""Extended-grid NE-HMC baseline, extends exp_nehmc.py: for one (L, kappa, K, n_leapfrog) scan the base
width s, the protocol exponent a (beta_t = (t/K)^a) and the leapfrog step eps. Learning stays off: momentum N(0, I),
no learned force. Stage t = 1..K: p ~ N(0, I), n_lf leapfrog substeps with
    F_t(phi) = beta_t (-grad S_kappa) - (1 - beta_t) phi / s^2,
reverse law N(0, I); q0 = N(0, s^2 I). W = log q0(x0) + sum_t [log N(p_t) - log N(-p'_t)] + S(x_K) is the exact path
ratio (same convention as exp_forward_eval.py / exp_nehmc.py). One JSON list entry per (s, a, eps)."""
import argparse
import itertools
import json
import math
import os
import time

import torch

from phi4 import phi4_action, phi4_force
from free_energy import summarize

LOG2PI = math.log(2 * math.pi)


def lognormal(z, s=1.0):
    return -0.5 * (z * z).sum(dim=(1, 2)) / s ** 2 - z[0].numel() * (math.log(s) + 0.5 * LOG2PI)


@torch.no_grad()
def paths(n, L, kappa, lam, K, nlf, eps, s, a):
    x = s * torch.randn(n, L, L, dtype=torch.float64)
    logP = lognormal(x, s)
    for t in range(1, K + 1):
        b = (t / K) ** a
        F = lambda z: b * phi4_force(z, lam=lam, kappa=kappa) - (1 - b) * z / s ** 2
        p = torch.randn_like(x)
        logP = logP + lognormal(p)
        p = p + 0.5 * eps * F(x)
        for _ in range(nlf - 1):
            x = x + eps * p
            p = p + eps * F(x)
        x = x + eps * p
        p = p + 0.5 * eps * F(x)
        logP = logP - lognormal(-p)
    return x, logP + phi4_action(x, lam=lam, kappa=kappa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.022)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n-leapfrog", type=int, required=True)
    ap.add_argument("--s", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--a", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []          # resume: skip finished configs
    done = {(r["s"], r["a"], r["eps"]) for r in res}
    for s, pa, eps in itertools.product(a.s, a.a, a.eps):
        if (s, pa, eps) in done:
            continue
        t0 = time.time(); torch.manual_seed(a.seed)
        Ws, Ms = [], []
        for _ in range(a.n // a.batch):
            x, W = paths(a.batch, a.L, a.kappa, a.lam, a.K, a.n_leapfrog, eps, s, pa)
            Ws.append(W); Ms.append(x.mean(dim=(1, 2)))
        W = torch.cat(Ws)
        fin = torch.isfinite(W)
        M = torch.cat(Ms)
        if fin.sum() < 2:
            continue
        out = summarize(W[fin], M[fin], a.L * a.L)          # non-finite paths (leapfrog blow-up) dropped and counted
        out.update(kernel="nehmc_grid", kappa=a.kappa, L=a.L, stages=a.K, substeps=a.n_leapfrog, eps=eps, s=s, a=pa,
                   n_nonfinite=int((~fin).sum()), force_evals_per_path=a.K * (a.n_leapfrog + 1), n=a.n, seed=a.seed,
                   wall=time.time() - t0)
        res.append(out)
        json.dump(res, open(a.out + ".tmp", "w"), indent=0); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
