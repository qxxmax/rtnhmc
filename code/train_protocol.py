#!/usr/bin/env python3
"""'Protocol-only' NE-HMC: the NHMC path of nhmc_resid2 with every neural part switched off, training only
the scalar protocol: beta_t = cumsum(softmax(beta_logits))[t-1] (beta_K = 1, as resid2), per-stage gate g_t on the
physical force, per-stage leapfrog step eps_t, base width s. Momentum N(0, I) both ways, no learned force, no flips.
Stage t: p ~ N(0, I), n_lf leapfrog substeps of size eps_t with F_t = (beta_t + g_t)(-grad S) - (1 - beta_t) phi / s^2,
reverse law N(0, I). W = log q0 + sum_t [log N(p) - log N(-p')] + S(x_K) (exact path ratio, exp_forward_eval convention).
Loss = E_q[W] (reverse KL up to log Z, reparametrised), Adam; checkpoint every 20 min (exact resume), then the final
evaluation (free_energy.summarize) with N = 262144 paths (seed 7) and three 65536-path seeds (11/12/13).
Output dir: model.pt (params + args), train_log.csv, final_eval.json."""
import argparse
import json
import math
import os
import time

import torch

from phi4 import phi4_action, phi4_force
from free_energy import summarize

LOG2PI = math.log(2 * math.pi)


class Proto(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.beta_logits = torch.nn.Parameter(torch.zeros(K))
        self.gate = torch.nn.Parameter(torch.zeros(K))
        self.log_eps = torch.nn.Parameter(torch.full((K,), -2.0))
        self.log_s = torch.nn.Parameter(torch.zeros(()))

    def paths(self, n, L, kappa, lam, nlf, dtype=torch.float32):
        s = torch.exp(self.log_s).clamp(0.05, 20.0)
        beta = torch.cumsum(torch.softmax(self.beta_logits, 0), 0)
        eps_t = torch.exp(self.log_eps).clamp(1e-4, 0.5)
        x = s * torch.randn(n, L, L, dtype=dtype)
        d = L * L
        logP = -0.5 * (x * x).sum(dim=(1, 2)) / s ** 2 - d * (torch.log(s) + 0.5 * LOG2PI)
        for t in range(self.K):
            b, g, e = beta[t], self.gate[t], eps_t[t]
            F = lambda z: (b + g) * phi4_force(z, lam=lam, kappa=kappa) - (1 - b) * z / s ** 2
            p = torch.randn_like(x)
            logP = logP - 0.5 * (p * p).sum(dim=(1, 2))
            p = p + 0.5 * e * F(x)
            for _ in range(nlf - 1):
                x = x + e * p
                p = p + e * F(x)
            x = x + e * p
            p = p + 0.5 * e * F(x)
            logP = logP + 0.5 * (p * p).sum(dim=(1, 2))           # the d/2 log 2pi terms cancel per stage
        return x, logP + phi4_action(x, lam=lam, kappa=kappa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.022)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n-leapfrog", type=int, default=5)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt-min", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(f"{a.out}/final_eval.json"):
        return
    torch.manual_seed(a.seed)
    m = Proto(a.K)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    it0, t_train = 0, 0.0
    ck = f"{a.out}/ckpt.pt"
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); it0, t_train = st["it"], st["t_train"]
        torch.set_rng_state(st["rng"]); print(f"resumed at {it0}", flush=True)
    log = open(f"{a.out}/train_log.csv", "a")
    if it0 == 0:
        log.write("it,loss,var_W,sec\n")
    t_ck = time.time()
    for it in range(it0, a.iters):
        t0 = time.time()
        _, W = m.paths(a.batch, a.L, a.kappa, a.lam, a.n_leapfrog)
        loss = W.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step(); t_train += time.time() - t0
        if it % 50 == 0:
            log.write(f"{it},{loss.item():.6f},{W.var().item():.6f},{t_train:.1f}\n"); log.flush()
        if time.time() - t_ck > 60 * a.ckpt_min:
            torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), it=it + 1, t_train=t_train,
                            rng=torch.get_rng_state()), ck + ".tmp"); os.replace(ck + ".tmp", ck); t_ck = time.time()
    torch.save(dict(model=m.state_dict(), args=vars(a), t_train=t_train), f"{a.out}/model.pt")
    res = []
    with torch.no_grad():
        for n, seed in ((262144, 7), (65536, 11), (65536, 12), (65536, 13)):
            torch.manual_seed(seed); Ws, Ms = [], []
            for _ in range(n // 4096):
                x, W = m.paths(4096, a.L, a.kappa, a.lam, a.n_leapfrog, dtype=torch.float64)
                Ws.append(W); Ms.append(x.mean(dim=(1, 2)))
            out = summarize(torch.cat(Ws), torch.cat(Ms), a.L * a.L)
            out.update(kernel="proto_nehmc", kappa=a.kappa, L=a.L, stages=a.K, substeps=a.n_leapfrog, n=n, seed=seed,
                       s=float(torch.exp(m.log_s)), beta=torch.cumsum(torch.softmax(m.beta_logits, 0), 0).tolist(),
                       eps=torch.exp(m.log_eps).tolist(), gate=m.gate.tolist(), t_train=t_train,
                       force_evals_per_path=a.K * (a.n_leapfrog + 1))
            res.append(out); print(json.dumps({k: out[k] for k in ("n", "seed", "ess_frac", "var_W", "logZ")}), flush=True)
    json.dump(res, open(f"{a.out}/final_eval.json.tmp", "w"), indent=0); os.replace(f"{a.out}/final_eval.json.tmp", f"{a.out}/final_eval.json")


if __name__ == "__main__":
    main()
