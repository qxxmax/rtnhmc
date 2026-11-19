#!/usr/bin/env python3
"""Inference-time path lengthening for a frozen nhmc_resid2 (Z2) checkpoint (2026-11-13; refine_eval2.py with the
sub-stage time grid fixed: tau_j = (t m + i - 1) / (T m), so for every m the first sub-stage uses the trained forward
law at t/T and the last sub-stage the trained reverse law at (t+1)/T; refine_eval2 used ((t-1) m + i) / (T m)).

Trained stage t = 1..T: Z2 flip (rho_t), p ~ q_F(.|x, t/T), n_lf leapfrog substeps of size eps_t with the force
    F_t = (beta_t + g_t) (-grad S) - (1 - beta_t) phi / s^2 + f_theta(phi, t/T, kappa),
then the reverse law q_R(-p|x', (t+1)/T).
Config "m x r x c": every trained stage becomes m stages j (network time tau_j = (t m + i - 1) / (T m), i = 1..m,
reverse law at tau_j + 1/(T m)), each with n_lf * r substeps of size c * eps_t / (m r) and the trained beta_t, g_t.
  c = 1: same total integration time, finer discretisation (refinement);
  c = m: m times longer integration time per trained stage (stretching).
Z2 flips only at the first sub-stage of each trained stage, with (rho_t, rho_{t+1}) as trained.
Every sub-stage is a volume-preserving reversible map between normalised momentum laws, so W is still the exact
path ratio: SNIS / log Z / IMH stay consistent for any config, only the variance changes.
1x1x1 reproduces eval_generic.py draw for draw (same seed, same RNG order)."""
import argparse
import json
import os
import time

import torch
from torch.distributions import Normal

from action import phi4_action
from cond_kernels import _broadcast_kappa, base_draw, create_cond_kernel
from eval_generic import summarize


@torch.no_grad()
def paths(k, n, L, kappa, lam, T, m, r, c, device):
    b = k.base
    x, logP = base_draw(k, (n, 1, L, L), device, kappa)
    kap = _broadcast_kappa(kappa, n, device, x.dtype)
    inv_s2 = 1.0 / b.base_std() ** 2
    nlf = b.n_leapfrog * r
    Tp = T * m
    for t in range(1, T + 1):
        beta, g = b.beta(t), b.gate[t - 1]
        eps = b.eps_at(t) * c / (m * r)
        for i in range(1, m + 1):
            j = t * m + i - 1          # sub-stage 1 at the trained t/T, reverse law of sub-stage m at (t+1)/T
            if i == 1:
                pf = torch.sigmoid(k.flip_logits[t]); pb = torch.sigmoid(k.flip_logits[min(t + 1, T)])
                flip = torch.rand(n, device=device) < pf
                x = x * torch.where(flip, -1.0, 1.0).to(x.dtype).view(-1, 1, 1, 1)
                lf_rho = torch.where(flip, torch.log(pf + 1e-8), torch.log(1 - pf + 1e-8))
                lb_rho = torch.where(flip, torch.log(pb + 1e-8), torch.log(1 - pb + 1e-8))
            tn = torch.full((n,), j / Tp, device=device, dtype=x.dtype)
            tn1 = torch.full((n,), (j + 1) / Tp, device=device, dtype=x.dtype)
            qf = Normal(b.fwd_mu(x, tn, kap), torch.exp(b.fwd_sigma(x, tn, kap)))
            p = qf.rsample()
            lf = qf.log_prob(p).sum(dim=(1, 2, 3))
            F = lambda z: b._force2(z, beta, g, inv_s2, tn, kap)
            p = p + 0.5 * eps * F(x)
            for s in range(nlf - 1):
                x = x + eps * p
                p = p + eps * F(x)
            x = x + eps * p
            p = p + 0.5 * eps * F(x)
            qb = Normal(b.bwd_mu(x, tn1, kap), torch.exp(b.bwd_sigma(x, tn1, kap)))
            lb = qb.log_prob(-p).sum(dim=(1, 2, 3))
            if i == 1:
                lf, lb = lf + lf_rho, lb + lb_rho
            logP = logP + lf - lb
    return x, logP + phi4_action(x.squeeze(1), lam=lam, kappa=kappa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="kernel_final.pt")
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--configs", nargs="+", default=["1x1x1", "1x2x1", "2x1x1", "4x1x1", "2x1x2", "4x1x4"])
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    assert cfg["kernel"] == "nhmc_resid2", cfg["kernel"]
    k = create_cond_kernel(cfg["kernel"], embed_dim=cfg.get("embed_dim", 128),
                           channels=tuple(cfg.get("channels", [32, 32])), n_leapfrog=cfg["n_leapfrog"],
                           lam=cfg["lam"], num_steps=cfg["num_steps"], use_z2=cfg.get("use_z2", True),
                           odd_nets=cfg.get("odd_nets", False))
    k.load_state_dict(torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu"))
    k.eval().to(a.device)
    T, L = cfg["num_steps"], cfg["L"]
    res = []
    for cf in a.configs:
        m, r, c = (int(v) for v in cf.split("x"))
        t0 = time.time(); torch.manual_seed(a.seed)
        Ws, Ms = [], []
        for _ in range(a.n // a.batch):
            x, W = paths(k, a.batch, L, a.kappa, cfg["lam"], T, m, r, c, a.device)
            Ws.append(W.cpu()); Ms.append(x.mean(dim=(1, 2, 3)).double().cpu())
        out = summarize(torch.cat(Ws), torch.cat(Ms), L * L)
        out.update(config=cf, kappa=a.kappa, L=L, stages=T * m, substeps=cfg["n_leapfrog"] * r, time_scale=c,
                   force_evals_per_path=T * m * (cfg["n_leapfrog"] * r + 1), n=a.n, ckpt=a.ckpt, seed=a.seed,
                   run_dir=a.run_dir, wall=time.time() - t0)
        res.append(out); print(json.dumps(out), flush=True)
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
