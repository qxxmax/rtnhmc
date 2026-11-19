#!/usr/bin/env python3
"""Inference-time refinement of a frozen overdamped CMCD / DBS phi^4 sampler (cmcd_phi4.py port of the official UDB
code), 2026-11-14. Question: does CMCD, as an Euler-Maruyama discretisation of a controlled SDE, improve when the
trained T-step path is sampled with m*T steps (as SPS: train N = 250, sample N = 2500)?

Refined path, factor m (and time-stretch c, default 1): T' = m T steps j' = 0..T', continuous step index s = j'/m;
  b'(s)        linear interpolation of the trained b_j (j = 0..T, last value clamped as in coeffs());
  eta'_{k'}    = c * eta_{floor(k'/m)} / m     (per dim; total integration time unchanged for c = 1);
  nets         evaluated at the float step s (PISGRADNet's Fourier features are continuous in the step, but with
               frequencies up to 100 rad/step they are far off the trained grid between integers), or with 'h'
               held at the trained step floor(s).
Kernels exactly as cmcd_phi4.sample_paths with these coefficients: F = N(x + eta'(drift + f)(x, s), 2 eta'),
B = N(x' + eta'(drift + g)(x', s + 1/m), 2 eta'). Both are normalised Gaussians, so log w is still the exact path
ratio for every (m, c): SNIS / log Z stay consistent, only the variance changes. m = c = 1 reproduces
cmcd_phi4.final_eval's sampler draw for draw (same seed)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get('CMCD_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)))))
import cmcd_phi4 as C  # noqa: E402


def refined_coeffs(m, mult, stretch, hold):
    T = m.T0
    c = C.UDBPhi4.coeffs(m)                       # trained coefficients, T0 steps
    s = torch.arange(T * mult + 1, device=c['b'].device, dtype=c['b'].dtype) / mult
    lo = s.floor().long().clamp(max=T); hi = (lo + 1).clamp(max=T); fr = s - lo.to(s.dtype)
    b = c['b'][lo] * (1 - fr) + c['b'][hi] * fr
    k = (torch.arange(T * mult, device=s.device) // mult)
    eta = c['eta'][k] * (stretch / mult)
    sn = s.floor() if hold else s                # hold: nets at the trained integer step floor(s)
    out = dict(b=b, eta=eta, scale=torch.sqrt(2 * eta), sig=c['sig'], mu=c['mu'], frow=m.fwd.time_rows(sn))
    if m.bwd is not None:
        out['brow'] = m.bwd.time_rows(sn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='model.pt of cmcd_phi4.py')
    ap.add_argument('--kappa', type=float, required=True)
    ap.add_argument('--configs', nargs='+', default=['1x1', '2x1', '4x1', '8x1', '2x1h', '4x1h', '8x1h', '2x2'],
                    help='mult x stretch [h = nets held at the trained step floor(s)]')
    ap.add_argument('--n', type=int, default=102400)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    st = torch.load(a.model, map_location='cpu', weights_only=False)
    ar = argparse.Namespace(**st['args'])
    ar.kappa = a.kappa
    m = C.build(ar)
    m.load_state_dict(st['model'])
    m = m.to(dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.T0 = m.T
    res = []
    for cf in a.configs:
        hold = cf.endswith('h')
        mult, stretch = (int(v) for v in cf.rstrip('h').split('x'))
        m.T = m.T0 * mult
        m.coeffs = (lambda mm=m, u=mult, v=stretch, h=hold: refined_coeffs(mm, u, v, h))
        torch.manual_seed(a.seed); t0 = time.time()
        lws, Ms = [], []
        with torch.no_grad():
            for i in range(0, a.n, a.chunk):
                lw, x, _ = m.sample_paths(min(a.chunk, a.n - i), torch.device('cpu'), torch.float32)
                lws.append(lw.double()); Ms.append(x.mean(-1).double())
        out = C.summarize(torch.cat(lws).numpy(), torch.cat(Ms).numpy(), m.dim)
        out.update(config=cf, alg=ar.alg, kappa=a.kappa, L=ar.L, T_trained=m.T0, T=m.T, seed=a.seed, model=a.model,
                   wall=time.time() - t0)
        res.append(out); print(json.dumps(out), flush=True)
        json.dump(res, open(a.out + '.tmp', 'w'), indent=1); os.replace(a.out + '.tmp', a.out)


if __name__ == '__main__':
    main()
