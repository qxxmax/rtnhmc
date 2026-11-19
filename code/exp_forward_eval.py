#!/usr/bin/env python3
"""Path-SNIS evaluation of any conditional NHMC kernel (Z2 wrapper) at fixed kappa values.

Base law from kernels.base_draw (N(0, I) or the kernel's learned N(0, s^2)), then the
kernel's own forward_step for every stage, so W = log q0 + sum(log qF - log qR) + S(x_T).
Output fields: free_energy.summarize (ESS fraction, log Z, Pareto k-hat, reweighted observables).
"""
import argparse
import json
import math
import os
import time

import torch

from kernels import create_cond_kernel
from paths import paths
from free_energy import pareto_khat, summarize  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="kernel_final.pt")
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.20, 0.27, 0.30])
    ap.add_argument("--n", type=int, default=102400)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    k = create_cond_kernel(cfg["kernel"], embed_dim=cfg.get("embed_dim", 128),
                           channels=tuple(cfg.get("channels", [32, 32])), n_leapfrog=cfg["n_leapfrog"],
                           lam=cfg["lam"], num_steps=cfg["num_steps"], use_z2=cfg.get("use_z2", True),
                           odd_nets=cfg.get("odd_nets", False))
    k.load_state_dict(torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu"))
    k.eval().to(dev)
    T, L = cfg["num_steps"], cfg["L"]
    pf = torch.sigmoid(k.flip_logits[1:T + 1]).mean().item() if hasattr(k, "flip_logits") else None
    res = []
    for kap in a.kappas:
        t0 = time.time(); torch.manual_seed(a.seed)
        Ws, Ms = [], []
        for _ in range(a.n // a.batch):
            x, W = paths(k, a.batch, L, kap, cfg["lam"], T, dev)
            Ws.append(W.cpu()); Ms.append(x.mean(dim=(1, 2, 3)).double().cpu())
        out = summarize(torch.cat(Ws), torch.cat(Ms), L * L)
        out.update(kernel=cfg["kernel"], kappa=kap, L=L, stages=T, substeps=cfg["n_leapfrog"], p_flip=pf,
                   force_evals_per_path=T * (cfg["n_leapfrog"] + 1), n=a.n, ckpt=a.ckpt, seed=a.seed, wall=time.time() - t0)
        res.append(out); print(json.dumps(out), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
