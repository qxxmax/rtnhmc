#!/usr/bin/env python3
"""Forward-path work W per path for a frozen NHMC model, with block-jackknife Var(log w) and Kish ESS/N (2026-11-17).
NEEDS CLUSTER: N fresh forward paths (default 262144, as the eval_N262144.json files) -- GPU on iqcb (fast) or HIMSTER3 CPU.
Uses nhmc_code/eval_generic.paths (same W convention as eval_N262144.json: W = log q0 + sum(log qF - log qR) + S(x_T),
log w = -W). Targets: the kappa=0.27 K-scan / stage-scan models (kscan_cpu/L*_k27_K*_s*, kscan_base, *_phi4_buffer/L*_k0.27_buf_s0;
fig_stage_scan, fig_scaling_var (c), App. D line 8) and the CMCD-free NHMC rows of tab:udb_compare (L8 buffer models).
  PYTHONPATH=$MQ/revision_code/nhmc_code python eval_W_dump.py --run-dir <model dir> --kappa 0.27 --n 262144 --device cuda \
      --out <dir>/evalW_N262144.json
Writes <out> (var_W, its 64-block jackknife error, Kish ESS/N and error, n_ESS) and <out>.npz (W, m per path, float32)."""
import argparse
import json
import os

import numpy as np
import torch

from cond_kernels import create_cond_kernel
from eval_generic import paths


def kish(lw):
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


def jk(blk, stat):
    B = len(blk)
    reps = np.array([stat(np.concatenate([b for j, b in enumerate(blk) if j != i])) for i in range(B)])
    return float(np.sqrt((B - 1) / B * ((reps - reps.mean()) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="kernel_final.pt")
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--n", type=int, default=262144)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7, help="7 = same stream as eval_generic (reproduces eval_N262144.json); change for an independent sample")
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    k = create_cond_kernel(cfg["kernel"], embed_dim=cfg.get("embed_dim", 128), channels=tuple(cfg.get("channels", [32, 32])),
                           n_leapfrog=cfg["n_leapfrog"], lam=cfg["lam"], num_steps=cfg["num_steps"],
                           use_z2=cfg.get("use_z2", True), odd_nets=cfg.get("odd_nets", False))
    k.load_state_dict(torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu")); k.eval().to(a.device)
    torch.manual_seed(a.seed)
    Ws, Ms = [], []
    with torch.no_grad():
        for _ in range(a.n // a.batch):
            x, W = paths(k, a.batch, cfg["L"], a.kappa, cfg["lam"], cfg["num_steps"], a.device)
            Ws.append(W.double().cpu().numpy()); Ms.append(x.squeeze(1).mean((-1, -2)).double().cpu().numpy())
    W, M = np.concatenate(Ws), np.concatenate(Ms)
    ok = np.isfinite(W)
    lw = -W[ok]
    n = len(lw) // a.blocks * a.blocks
    blk = np.split(lw[:n], a.blocks)
    e = kish(lw[:n])
    r = dict(run_dir=a.run_dir, kappa=a.kappa, L=cfg["L"], T=cfg["num_steps"], n=int(n), n_nonfinite=int((~ok).sum()), seed=a.seed,
             var_W=float(lw[:n].var()), var_W_jk=jk(blk, np.var), kish_ess_frac=e, kish_jk=jk(blk, kish), n_ess=e * n,
             note="64-block delete-one jackknife; Kish jackknife unreliable when n_ess < ~50 (heavy tail)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out[:-5] + ".npz", W=W.astype(np.float32), m=M.astype(np.float32))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
