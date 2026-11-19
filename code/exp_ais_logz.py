#!/usr/bin/env python3
"""Reference log Z and observables for 2D phi^4 (L x L, periodic).

S = sum_x (1-2 lam) phi^2 + lam phi^4 - 2 kappa sum_mu phi(x) phi(x+mu)
Z = int exp(-S) dphi (Lebesgue), same convention as NHMC full_is_eval and the
UDB Phi4Distr target.

At kappa = 0 the sites are independent, so log Z_0 = V log z1 with z1 a 1D
quadrature, and phi can be sampled exactly by inverse CDF. AIS then anneals
kappa linearly from 0 to the target with MH-corrected HMC moves.
"""
import argparse
import json
import math
import time

import numpy as np
import torch

from free_energy import site_quadrature


def action(x, lam, kappa):
    s = (1 - 2 * lam) * x ** 2 + lam * x ** 4
    s = s - 2 * kappa * x * (torch.roll(x, 1, -1) + torch.roll(x, 1, -2))
    return s.sum((-1, -2))


def coupling(x):
    # d S / d kappa = -2 * sum phi phi'
    return -2 * (x * (torch.roll(x, 1, -1) + torch.roll(x, 1, -2))).sum((-1, -2))


def grad_action(x, lam, kappa):
    g = 2 * (1 - 2 * lam) * x + 4 * lam * x ** 3
    nb = (torch.roll(x, 1, -1) + torch.roll(x, -1, -1)
          + torch.roll(x, 1, -2) + torch.roll(x, -1, -2))
    return g - 2 * kappa * nb


def hmc_step(x, lam, kappa, eps, n_lf, gen):
    p = torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)
    h0 = action(x, lam, kappa) + 0.5 * (p ** 2).sum((-1, -2))
    xn = x.clone()
    pn = p - 0.5 * eps * grad_action(xn, lam, kappa)
    for i in range(n_lf):
        xn = xn + eps * pn
        if i < n_lf - 1:
            pn = pn - eps * grad_action(xn, lam, kappa)
    pn = pn - 0.5 * eps * grad_action(xn, lam, kappa)
    h1 = action(xn, lam, kappa) + 0.5 * (pn ** 2).sum((-1, -2))
    acc = torch.rand(x.shape[0], generator=gen, dtype=x.dtype, device=x.device) < torch.exp(h0 - h1)
    acc = acc & torch.isfinite(h1)
    x = torch.where(acc[:, None, None], xn, x)
    return x, acc.double().mean().item()


def run_ais(L, lam, kappa, n, K, n_hmc, n_lf, seed, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    logz1, grid, cdf = site_quadrature(lam)
    V = L * L
    u = rng.random((n, L, L))
    x = torch.tensor(np.interp(u, cdf, grid), dtype=torch.float64, device=device)
    logw = torch.zeros(n, dtype=torch.float64, device=device)
    kappas = np.linspace(0.0, kappa, K + 1)
    eps = 0.3
    accs = []
    for k in range(1, K + 1):
        dk = kappas[k] - kappas[k - 1]
        logw = logw - dk * coupling(x)
        for _ in range(n_hmc):
            x, a = hmc_step(x, lam, kappas[k], eps, n_lf, gen)
            accs.append(a)
            # simple step-size adaptation toward ~0.8 acceptance
            eps *= math.exp(0.05 * (a - 0.8))
    m = logw.max()
    w = torch.exp(logw - m)
    logZ = V * logz1 + (m + torch.log(w.mean())).item()
    wn = w / w.sum()
    ess = (1.0 / (wn ** 2).sum()).item() / n
    M = x.mean((-1, -2))
    absm = (wn * M.abs()).sum().item()
    m2 = (wn * M ** 2).sum().item()
    chi = V * (m2 - absm ** 2)
    return dict(logZ=logZ, ess_frac=ess, abs_mag=absm, m2=m2, chi=chi,
                mean_acc=float(np.mean(accs)), final_eps=eps, logZ0=V * logz1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.022)
    ap.add_argument("--kappas", type=str, default="0.20,0.27,0.30")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--n_hmc", type=int, default=2)
    ap.add_argument("--n_lf", type=int, default=5)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", type=str, default="logz_ref.json")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = {}
    for kap in [float(s) for s in a.kappas.split(",")]:
        reps = []
        for r in range(a.reps):
            t0 = time.time()
            o = run_ais(a.L, a.lam, kap, a.n, a.K, a.n_hmc, a.n_lf, seed=1000 + r, device=a.device)
            o["wall"] = time.time() - t0
            reps.append(o)
            print(f"kappa={kap} rep={r} " + " ".join(f"{k}={v:.6g}" for k, v in o.items()), flush=True)
        lz = np.array([o["logZ"] for o in reps])
        res[f"{kap:.2f}"] = dict(
            logZ_mean=float(lz.mean()), logZ_sem=float(lz.std(ddof=1) / math.sqrt(len(lz))) if len(lz) > 1 else None,
            abs_mag=float(np.mean([o["abs_mag"] for o in reps])),
            chi=float(np.mean([o["chi"] for o in reps])),
            reps=reps)
        with open(a.out, "w") as f:
            json.dump(dict(config=vars(a), results=res), f, indent=2)
    print(json.dumps({k: {kk: v[kk] for kk in ("logZ_mean", "logZ_sem", "abs_mag", "chi")} for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
