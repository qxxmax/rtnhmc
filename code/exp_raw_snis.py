#!/usr/bin/env python3
"""Raw and SNIS observables from one batch of forward paths.

The same n forward paths of one checkpoint give both uses of the path model:
  raw   equal-weight average over the path endpoints (how close the learned generative law is to the target);
  SNIS  endpoints weighted by w = exp(-W) (work convention of paths.py: E_Q[e^{-W}] = Z), self-normalised.
Observables per configuration: m = mean of phi over the sites; <|m|>, chi = V (<m^2> - <|m|>^2),
U4 = 1 - <m^4> / (3 <m^2>^2). Errors: jackknife over B blocks of the (independent) paths; for SNIS every jackknife
replicate re-normalises the weights. Also Kish ESS fraction (sum w)^2 / (n sum w^2) and the largest normalised weight.
Output json (estimates) + npz (m and W of every path)."""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_chains as LC  # noqa: E402


def moments(m, w, V):
    w = w / w.sum()
    a, b, c = (w * np.abs(m)).sum(), (w * m ** 2).sum(), (w * m ** 4).sum()
    return np.array([a, V * (b - a * a), 1 - c / (3 * b * b)])


def estimate(m, lw, V, B):
    n = len(m) // B * B
    m, lw = m[:n], lw[:n]
    out = {}
    for name, l in (('raw', np.zeros_like(lw)), ('snis', lw)):
        w = np.exp(l - l.max())
        full = moments(m, w, V)
        idx = np.arange(n).reshape(B, -1)
        jk = np.array([moments(np.delete(m, idx[i]), np.delete(w, idx[i]), V) for i in range(B)])
        err = np.sqrt((B - 1) / B * ((jk - jk.mean(0)) ** 2).sum(0))
        out[name] = {k: [float(full[j]), float(err[j])] for j, k in enumerate(('abs_m', 'chi', 'binder'))}
    w = np.exp(lw - lw.max())
    out['kish_ess_frac'] = float(w.sum() ** 2 / (n * (w ** 2).sum()))
    out['max_weight'] = float(w.max() / w.sum())
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--ckpt', default='kernel_final.pt')
    ap.add_argument('--L', type=int, default=None)
    ap.add_argument('--L2', type=int, default=None, help='second lattice extent (rectangular lattices)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--kappa', type=float, required=True)
    ap.add_argument('--n', type=int, default=65536)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--blocks', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20261111)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    t0 = time.time()
    st = LC.PhiStepper(a.run_dir, a.ckpt, a.kappa, torch.device(a.device), a.L, a.L2)
    V = st.L * st.L2
    # checkpoint after every batch: paths so far + torch RNG state, so a resumed run gives exactly the
    # paths of an uninterrupted one; removed once the json is written
    part = pathlib.Path(a.out[:-5] + '.part.npz')
    ms, Ws, i0, t_prev = [], [], 0, 0.0
    if part.exists():
        P = np.load(part)
        ms, Ws, i0, t_prev = [P['m']], [P['W']], int(P['i']), float(P['seconds'])
        torch.set_rng_state(torch.from_numpy(P['rng']))
        print(f'resume at {i0} paths', flush=True)
    for i in range(i0, a.n, a.batch):
        x, W = st.start(min(a.batch, a.n - i))
        ms.append(x.reshape(x.shape[0], -1).double().mean(1).cpu().numpy())
        Ws.append(W.double().cpu().numpy())
        tmp = part.with_name(part.name + '.tmp.npz')
        np.savez(tmp, m=np.concatenate(ms), W=np.concatenate(Ws), i=i + x.shape[0],
                 rng=torch.get_rng_state().numpy(), seconds=t_prev + time.time() - t0)
        tmp.replace(part)
    m, W = np.concatenate(ms), np.concatenate(Ws)
    ok = np.isfinite(W)
    res = estimate(m[ok], -W[ok], V, a.blocks)
    res.update(n=int(ok.sum()), n_nonfinite=int((~ok).sum()), L=st.L, L2=st.L2, kappa=a.kappa, model=st.id, args=vars(a),
               seconds=t_prev + time.time() - t0)
    np.savez_compressed(a.out[:-5] + '.npz', m=m.astype(np.float32), W=W)
    json.dump(res, open(a.out, 'w'), indent=1)
    part.unlink(missing_ok=True)
    print(json.dumps({k: res[k] for k in ('raw', 'snis', 'kish_ess_frac', 'n', 'seconds')}))


if __name__ == '__main__':
    main()
