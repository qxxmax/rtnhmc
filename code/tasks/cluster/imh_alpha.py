#!/usr/bin/env python3
"""Per-state acceptance heterogeneity of path-IMH (2026-11-14). For IMH the acceptance from a state with work W_x is
alpha(W_x) = E_q[min(1, exp(W_x - W_y))] (fresh forward path y), and the stationary law of W_x is q tilted by e^{-W}.
From one pool of N forward paths (same RNG order as eval_generic) we estimate
  mean acceptance      abar = E_pi[alpha]            (self-normalised weights e^{-W})
  heterogeneity        E_pi[1/alpha]  and  tau_pred = 2 E_pi[1/alpha] - 1 (rejection-run bound for the IMH chain)
  Gaussian reference   erfc(sigma/2) with sigma^2 = Var_q(W)   (same mean acceptance as RT at equal variance)
The contrast with RT: RT redraws the record of the current state at every update, so its acceptance is
approximately the same for every state; IMH carries W_x, and states in the upper tail of e^{-W} keep alpha small.
alpha(W) is evaluated on a grid of W quantiles against the full pool (exact for the pool), then interpolated."""
import argparse
import json
import math
import os

import numpy as np
import torch

from eval_generic import paths
from cond_kernels import create_cond_kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="kernel_final.pt")
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--n", type=int, default=262144)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    k = create_cond_kernel(cfg["kernel"], embed_dim=cfg.get("embed_dim", 128), channels=tuple(cfg.get("channels", [32, 32])),
                           n_leapfrog=cfg["n_leapfrog"], lam=cfg["lam"], num_steps=cfg["num_steps"],
                           use_z2=cfg.get("use_z2", True), odd_nets=cfg.get("odd_nets", False))
    k.load_state_dict(torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu")); k.eval()
    torch.manual_seed(a.seed)
    W = torch.cat([paths(k, a.batch, cfg["L"], a.kappa, cfg["lam"], cfg["num_steps"], "cpu")[1] for _ in range(a.n // a.batch)]).double().numpy()
    W = W[np.isfinite(W)]
    Ws = np.sort(W)
    grid = np.quantile(W, np.linspace(0, 1, 401))
    # alpha(Wx) = mean_y min(1, exp(Wx - Wy)) over the pool, via the sorted pool and cumulative sums of exp(-Wy)
    c = np.concatenate([[0.0], np.cumsum(np.exp(-(Ws - Ws.min())))])
    def alpha(wx):
        j = np.searchsorted(Ws, wx)                         # pool paths with Wy < Wx are accepted with probability 1
        return (j + np.exp(wx - Ws.min()) * (c[-1] - c[j])) / len(Ws)   # the rest with exp(Wx - Wy)
    ag = np.array([alpha(g) for g in grid])
    aw = np.interp(W, grid, ag)
    lw = -(W - W.min()); w = np.exp(lw - lw.max()); w /= w.sum()
    abar = float((w * aw).sum()); inv = float((w / aw).sum())
    sig2 = float(W.var())
    out = dict(run_dir=a.run_dir, kappa=a.kappa, L=cfg["L"], T=cfg["num_steps"], n=int(len(W)), var_W=sig2,
               abar_imh=abar, E_inv_alpha=inv, tau_pred_imh=2 * inv - 1, gauss_mean_acc=math.erfc(math.sqrt(sig2) / 2),
               ess_frac=float(1 / (w ** 2).sum() / len(W)),
               alpha_quantiles={q: float(np.interp(np.quantile(W, q), grid, ag)) for q in (0.001, 0.01, 0.1, 0.5, 0.9)})
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({x: out[x] for x in ("L", "kappa", "var_W", "abar_imh", "gauss_mean_acc", "tau_pred_imh")}), flush=True)


if __name__ == "__main__":
    main()
