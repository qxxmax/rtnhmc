#!/usr/bin/env python3
"""Conditional work variance sigma_s^2 of a frozen NHMC model, with a jackknife over pivot groups.

A truncated round trip through stage s compares two records that share the intermediate field u = x_s, so the quantity
that sets its acceptance is sigma_s^2 = E_u[Var(W_s | x_s = u)], W_s = S(x_T) + sum_{t > s}(log qF_t - log qR_t)
(s = 0: full round trip; W_0 has no log q0 term). It is estimated from forward paths only, before any chain is run:
n / R pivots x_s are drawn on forward paths, each continued R times to x_T (nested sampling). The Gaussian-work
prediction of the mean acceptance is erfc(sigma_s / 2).
Defaults as in the paper runs (seed 11, n 16384, R 16, batch 4096, 64 jackknife groups of pivots).
Outputs <out> (json: per stage cond_var, jk error, var_of_cond_mean, jk error, Gaussian acceptance prediction and its
jk error) and <out>.npz (per-pivot within variances and group means, per stage)."""
import argparse
import json
import math
import os

import numpy as np
import torch

from phi4 import phi4_action
from kernels import base_draw, create_cond_kernel


@torch.no_grad()
def cond_var_pivots(k, n_outer, R, L, kappa, lam, T, s):
    x, _ = base_draw(k, (n_outer, 1, L, L), "cpu", kappa)
    for t in range(1, s + 1):
        x, _, _ = k.forward_step(x, t, T, kappa=kappa)
    x = x.repeat_interleave(R, 0)
    acc = torch.zeros(x.shape[0], dtype=torch.float64)
    for t in range(s + 1, T + 1):
        x, lf, lb = k.forward_step(x, t, T, kappa=kappa)
        acc += (lf - lb).double()
    W = (acc + phi4_action(x.squeeze(1), lam=lam, kappa=kappa).double()).view(n_outer, R)
    return W.var(1).numpy(), W.mean(1).numpy()


def jk(groups, stat):
    G = len(groups)
    reps = np.array([stat(np.concatenate([g for j, g in enumerate(groups) if j != i])) for i in range(G)])
    return float(np.sqrt((G - 1) / G * ((reps - reps.mean()) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="kernel_final.pt")
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    k = create_cond_kernel(cfg["kernel"], embed_dim=cfg.get("embed_dim", 128), channels=tuple(cfg.get("channels", [32, 32])),
                           n_leapfrog=cfg["n_leapfrog"], lam=cfg["lam"], num_steps=cfg["num_steps"],
                           use_z2=cfg.get("use_z2", True), odd_nets=cfg.get("odd_nets", False))
    k.load_state_dict(torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu")); k.eval()
    T, L = cfg["num_steps"], cfg["L"]
    torch.manual_seed(a.seed)
    res, arrs = {}, {}
    for st in sorted({0, 2, 4, T // 2, 3 * T // 4, T - 2}):
        wv, gm = [], []
        for _ in range(a.n // a.batch):
            w, m = cond_var_pivots(k, a.batch // a.R, a.R, L, a.kappa, cfg["lam"], T, st)
            wv.append(w); gm.append(m)
        wv, gm = np.concatenate(wv), np.concatenate(gm)
        # same as wtail_var2: mean of per-batch means (equal batch sizes -> equal to the pooled mean)
        cv = float(wv.mean())
        vb = float(np.mean([g.var(ddof=1) for g in np.split(gm, a.n // a.batch)]))
        gw, gg = np.array_split(wv, a.groups), np.array_split(gm, a.groups)
        pred = lambda c: math.erfc(math.sqrt(max(c, 0)) / 2)
        res[st] = dict(cond_var=cv, cond_var_jk=jk(gw, np.mean), var_of_cond_mean=vb, var_of_cond_mean_jk=jk(gg, lambda g: g.var(ddof=1)),
                       acc_pred=pred(cv), acc_pred_jk=jk(gw, lambda g: pred(g.mean())), n_pivots=int(len(wv)))
        arrs[f"within_s{st}"] = wv.astype(np.float32); arrs[f"gmean_s{st}"] = gm.astype(np.float32)
        print(st, res[st], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(run_dir=a.run_dir, kappa=a.kappa, L=L, T=T, n=a.n, R=a.R, seed=a.seed, groups=a.groups, stages=res,
                   note="as tasks/wtail_var2.py + jackknife over pivot groups; acc_pred = erfc(sqrt(cond_var)/2)"), open(a.out, "w"), indent=1)
    np.savez_compressed(a.out[:-5] + ".npz", **arrs)


if __name__ == "__main__":
    main()
