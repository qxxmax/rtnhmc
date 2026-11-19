#!/usr/bin/env python3
"""Variance of the partial-round-trip work W_s (2026-11-14). For pRT with turning stage s the acceptance uses
    W_s = S(x_T) + sum_{k > s} [log qF_k - log qR_k]          (stages s+1..T, the redrawn part; eval_generic sign)
(rt_cmcd_phi4 / longchain convention; s = 0 is full RT, whose W_0 also lacks log q0). This script draws fresh forward
paths of a frozen NHMC model (same RNG order as eval_generic.paths) and reports Var(W_s) for every s, so the share of
the full-path work variance carried by the last d = T - s stages is measured directly (no bridge choice involved).
Gaussian-work prediction for the pRT acceptance: erfc(sqrt(2 Var W_s) / (2 sqrt 2)), compared with the measured chains."""
import argparse
import json
import os

import torch

from action import phi4_action
from cond_kernels import base_draw, create_cond_kernel


@torch.no_grad()
def tails(k, n, L, kappa, lam, T):
    x, _ = base_draw(k, (n, 1, L, L), "cpu", kappa)
    inc = []
    for t in range(1, T + 1):
        x, lf, lb = k.forward_step(x, t, T, kappa=kappa)
        inc.append((lf - lb).double())
    S = phi4_action(x.squeeze(1), lam=lam, kappa=kappa).double()
    inc = torch.stack(inc, 1)                                  # (n, T), stage t at column t-1
    rev = torch.flip(torch.cumsum(torch.flip(inc, [1]), 1), [1])   # rev[:, j] = sum_{t >= j+1} inc
    W = [S + rev[:, s] for s in range(T)] + [S]                # W_s for s = 0..T (s = T: endpoint only)
    return torch.stack(W, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="kernel_final.pt")
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--batch", type=int, default=4096)
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
    T, L = cfg["num_steps"], cfg["L"]
    torch.manual_seed(a.seed)
    Ws = torch.cat([tails(k, a.batch, L, a.kappa, cfg["lam"], T) for _ in range(a.n // a.batch)])
    var = Ws.var(0).tolist()
    out = dict(run_dir=a.run_dir, kappa=a.kappa, L=L, T=T, n=a.n, seed=a.seed, var_W_s=var,
               note="var_W_s[s] = Var(S(x_T) + sum_{t>s} (log qF_t - log qR_t)); the paper's pRT (longchain rtp:12) is s = 12, depth T - s = 4")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dict(L=L, kappa=a.kappa, var=[round(v, 3) for v in var])), flush=True)


if __name__ == "__main__":
    main()
