#!/usr/bin/env python3
"""Same per-file jackknife as ../jk_cells.py (acceptance block jackknife on accept flags, tau delete-one-chain and
16-segment jackknife), for chain jsons whose *_traces.npz exist only on the cluster (2026-11-17). Light (numpy only,
seconds per file); it is 'cluster' only because the traces are not synced to the Mac. Alternative: rsync the traces.
Files concerned (see jackknife_summary.md section 7):
  HIM  revision_runs/longchain_min/E2_L10_*, E2_L6_* (64 files; jobs 92853-92858, 93060/93061),
       revision_runs/longchain_min/T16_* (119 files), revision_runs/pilot_bigL/T16_L{48,64,96}_* (24 files)
  iqcb /media/scratch/mqian/revision_runs/longchain, .../longchain_udb_20261114 (CMCD/DBS rtp:60 chains of tab:udb_compare)
Usage: python jk_cells_remote.py OUT.json DIR [DIR ...]   (all *.json with a sibling *_traces.npz; 'chains' key required)"""
import glob
import json
import os
import sys

import numpy as np

NSEG, NBLK = 4, 20


def jk_err(t):
    t = np.asarray(t, float); n = len(t)
    return float(np.sqrt((n - 1) / n * ((t - t.mean()) ** 2).sum())) if n > 1 else float('nan')


def dl(x, stat):
    x = np.asarray(x)
    return float(stat(x)), jk_err([stat(np.delete(x, i, 0)) for i in range(len(x))])


def sokal_g(x, c=5.0):
    x = np.asarray(x, np.float64); x = x - x.mean(); n = x.size
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 1.0
    g = 1.0 + 2.0 * np.cumsum(ac[1:] / ac[0]); m = np.arange(1, n)
    ok = np.nonzero(m >= c * g)[0]
    return float(g[ok[0]] if len(ok) else g[-1])


def bm_g(x, nb=50):
    x = np.asarray(x, np.float64); b = x.size // nb
    if b < 2 or x.var() <= 0:
        return 0.0
    return float(b * x[:b * nb].reshape(nb, b).mean(1).var(ddof=1) / x.var())


def analyze(jf):
    d = json.load(open(jf)); cs = d.get('chains')
    if not cs:
        return None
    N, burn = int(d['n_transitions']), int(d.get('burn', 0))
    r = dict(file=jf, n_transitions=N, burn=burn, n_chains=len(cs), converged=bool(d.get('criteria_met')))
    ac = np.array([c['acceptance'] for c in cs]); r['acc'], r['acc_jk_chain'] = dl(ac, np.mean)
    for o in ('absM', 'sign'):
        if all(o in c for c in cs):
            t = np.array([max(c[o]['tau_sokal'], c[o].get('tau_batch_means') or 0) / 2 for c in cs])
            r[f'tau_{o}'] = float(np.median(t)); r[f'tau_{o}_jk_chain_mean'] = dl(t, np.mean)[1]; r[f'tau_{o}_chains'] = t.tolist()
    tp = jf[:-5] + '_traces.npz'
    if os.path.exists(tp):
        z = np.load(tp)
        a = z['acc'][burn:N].astype(float); r['n_acc'] = int(a.sum())
        n = a.shape[0] // NBLK * NBLK
        B = a[:n].reshape(NBLK, -1, a.shape[1]).mean(1).T.ravel()
        r['acc_jk_block'] = jk_err([(B.sum() - B[i]) / (len(B) - 1) for i in range(len(B))])
        for o in ('absM', 'sign'):
            if o in z.files:
                x = z[o][burn:N].astype(np.float64); ls = x.shape[0] // NSEG
                seg = np.array([max(sokal_g(x[i * ls:(i + 1) * ls, c]), bm_g(x[i * ls:(i + 1) * ls, c])) / 2
                                for c in range(x.shape[1]) for i in range(NSEG)])
                r[f'tau_{o}_seg_median'], r[f'tau_{o}_seg_median_jk'] = dl(seg, np.median)
                r[f'tau_{o}_seg_mean'], r[f'tau_{o}_seg_mean_jk'] = dl(seg, np.mean)
    return r


if __name__ == '__main__':
    out = {}
    for d in sys.argv[2:]:
        for f in sorted(glob.glob(os.path.join(d, '*.json'))):
            if f.endswith('_traces.json'):
                continue
            try:
                r = analyze(f)
            except Exception as e:  # noqa: BLE001
                r = dict(file=f, error=repr(e))
            if r:
                out[os.path.basename(f)[:-5]] = r
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
    print(len(out), 'files ->', sys.argv[1])
