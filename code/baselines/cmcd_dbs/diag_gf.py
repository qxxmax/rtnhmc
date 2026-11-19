#!/usr/bin/env python3
"""Stage-resolved diagnostics of a frozen CMCD / DBS phi^4 sampler under inference-time refinement (2026-11-15).
Question (Appendix K): does DBS lose under refinement because its separately trained reverse control g absorbed a
correction tied to the training step eta, while CMCD (g = -f) cannot?
For each refinement factor m (networks held at the trained step, step eta/m, total time unchanged) and each TRAINED
stage k (its m substeps summed) we record over N forward paths:
  diss_k   = mean of (log F - log B) over the stage's substeps   (forward-reverse mismatch of that stage)
  var_k    = variance of the same per-path stage increment
and, at m = 1 only, the size of the learned deviation from Nelson pairing along forward paths,
  dev_k    = RMS over paths and sites of (g + f)(x_{k+1}, k+1)    (identically zero for CMCD),
  eta_k    = mean learned step of stage k (the step size is learned, so the realized eta is reported).
Kernels and coefficients exactly as refine_cmcd.refined_coeffs / cmcd_phi4.sample_paths."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get('CMCD_DIR', os.path.dirname(os.path.abspath(__file__))))
import cmcd_phi4 as C  # noqa: E402
from refine_cmcd import refined_coeffs  # noqa: E402


def run(m, mult, n, chunk, need_dev):
    T0 = m.T0
    m.T = T0 * mult
    c = refined_coeffs(m, mult, 1, True)
    diss = torch.zeros(T0, dtype=torch.float64); s1 = torch.zeros(T0, dtype=torch.float64)
    s2 = torch.zeros(T0, dtype=torch.float64); dev = torch.zeros(T0, dtype=torch.float64)
    lw_all = []
    done = 0
    while done < n:
        b = min(chunk, n - done)
        x = m.prior_sample(b, c, torch.device('cpu'), torch.float32)
        lw = -m.prior_logpdf(x, c).double()
        drift, f, _ = m.point(x, 0, c, need_b=False)
        inc = torch.zeros(b, T0, dtype=torch.float64)
        for k in range(m.T):
            e, sc = c['eta'][k], c['scale'][k]
            fm = e * drift + e * f
            xn = x + fm + sc * torch.randn_like(x)
            drift_n, f_n, g_n = m.point(xn, k + 1, c, need_f=True)
            bm = e * (drift_n + g_n)
            dx = xn - x
            d = (m.lognormal(dx, fm, sc) - m.lognormal(-dx, bm, sc)).double()   # log F - log B
            inc[:, k // mult] += d
            lw = lw - d
            if need_dev and f_n is not None:
                dev[k // mult] += ((g_n + f_n).double() ** 2).mean(-1).sum()
            x, drift, f = xn, drift_n, f_n
        lw = lw + m.log_gamma(x).double()
        lw_all.append(lw)
        s1 += inc.sum(0); s2 += (inc ** 2).sum(0)
        done += b
    mean = s1 / n
    var = s2 / n - mean ** 2
    lw = torch.cat(lw_all)
    out = dict(mult=mult, var_logw=float(lw.var()), diss=mean.tolist(), stage_var=var.tolist())
    if need_dev:
        out['dev_rms'] = (dev / n).sqrt().tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--kappa', type=float, required=True)
    ap.add_argument('--mults', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    ap.add_argument('--n', type=int, default=16384)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    st = torch.load(a.model, map_location='cpu', weights_only=False)
    ar = argparse.Namespace(**st['args']); ar.kappa = a.kappa
    m = C.build(ar); m.load_state_dict(st['model'])
    m = m.to(dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.T0 = m.T
    c0 = C.UDBPhi4.coeffs(m)
    res = dict(model=a.model, alg=ar.alg, kappa=a.kappa, L=ar.L, T=m.T0, dt_arg=getattr(ar, 'dt', None),
               eta_stage=c0['eta'].double().mean(-1).tolist(), total_time=float(c0['eta'].double().mean(-1).sum()),
               runs=[])
    t0 = time.time()
    with torch.no_grad():
        for mult in a.mults:
            torch.manual_seed(a.seed)
            r = run(m, mult, a.n, a.chunk, need_dev=(mult == 1 and ar.alg == 'dbs'))
            res['runs'].append(r)
            print(f"m={mult} var_logw={r['var_logw']:.4f} sum diss={sum(r['diss']):.4f}", flush=True)
            m.T = m.T0
    res['wall'] = time.time() - t0
    json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()
