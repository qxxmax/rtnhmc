"""Per-chain-file jackknife errors for acceptance and tau_int(|m|), tau_int(sign) (2026-11-17).
Usage: python jk_cells.py GROUP   with GROUP in {E2, buf, T16, T4, T8, T12, E2tr16}; writes cells_<GROUP>.json.

For every chain json (selection as fig_scaling_20261116.files()):
  acc            mean of per-chain acceptance (= paper)            ; acc_jk_chain  delete-one-chain jackknife
  acc_jk_block   block jackknife on the accept flags acc[burn:N] (20 blocks per chain, all chains pooled) if traces exist
  n_acc          number of accepted transitions after burn-in (from traces, else acc * n_post * n_chains)
  tau_<o>        median over chains of max(Sokal, BM)/2 (= paper)   ; tau_<o>_jk_chain delete-one-chain jackknife of the median
  tau_<o>_jk_chain_mean   same for the mean over chains (better-behaved statistic, reported for comparison)
  tau_<o>_ms     Madras-Sokal error of the chain mean: sqrt(sum_c (tau_c^2 2(2W_c+1)/n_post))/n_chain
  tau_<o>_seg    (traces) every chain cut into 4 equal post-burn segments; 16 segment values of max(Sokal,BM)/2;
                 median / mean and their delete-one-segment jackknife errors (segment length >= n_post/4)
  converged      criteria_met from the json (False -> tau is a lower bound)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jk_lib import files, jk_delete_one, jk_err, sokal_g, bm_g, traces_for, load_json, dump  # noqa: E402

NSEG, NBLK = 4, 20


def tau_seg(x):
    g, _ = sokal_g(x)
    b = bm_g(x)
    return max(g, b or 0.0) / 2


def analyze(jf):
    d = load_json(jf)
    cs = d.get('chains')
    if not cs:
        return None
    N, burn = int(d['n_transitions']), int(d.get('burn', 0))
    npost = N - burn
    r = dict(file=jf, n_transitions=N, burn=burn, n_chains=len(cs), converged=bool(d.get('criteria_met')))
    acc_c = np.array([c['acceptance'] for c in cs], float)
    r['acc'], r['acc_jk_chain'], _ = jk_delete_one(acc_c, np.mean)
    r['acc_chains'] = acc_c.tolist()
    for o in ('absM', 'sign'):
        if not all(o in c for c in cs):
            continue
        t = np.array([max(c[o]['tau_sokal'], c[o].get('tau_batch_means') or 0) / 2 for c in cs], float)
        r[f'tau_{o}'], r[f'tau_{o}_jk_chain'], _ = jk_delete_one(t, np.median)
        r[f'tau_{o}_mean'], r[f'tau_{o}_jk_chain_mean'], _ = jk_delete_one(t, np.mean)
        r[f'tau_{o}_chains'] = t.tolist()
        ts = np.array([c[o]['tau_sokal'] / 2 for c in cs])
        W = np.array([c[o]['window'] for c in cs], float)
        r[f'tau_{o}_ms'] = float(np.sqrt((ts ** 2 * 2 * (2 * W + 1) / npost).sum()) / len(cs))
    tp = traces_for(jf)
    r['traces'] = tp
    if tp is not None:
        z = np.load(tp)
        a = z['acc'][burn:N].astype(float)
        r['acc_traces'] = float(a.mean())
        r['n_acc'] = int(a.sum())
        n = a.shape[0] // NBLK * NBLK
        B = a[:n].reshape(NBLK, -1, a.shape[1]).mean(1).T.ravel()          # chains x blocks, pooled
        r['acc_jk_block'] = jk_err([(B.sum() - B[i]) / (len(B) - 1) for i in range(len(B))])
        for o in ('absM', 'sign'):
            if o not in z.files:
                continue
            x = z[o][burn:N].astype(np.float64)
            ls = x.shape[0] // NSEG
            segs = np.array([tau_seg(x[i * ls:(i + 1) * ls, c]) for c in range(x.shape[1]) for i in range(NSEG)])
            r[f'tau_{o}_seg_median'], r[f'tau_{o}_seg_median_jk'], _ = jk_delete_one(segs, np.median)
            r[f'tau_{o}_seg_mean'], r[f'tau_{o}_seg_mean_jk'], _ = jk_delete_one(segs, np.mean)
            r[f'tau_{o}_seg_len_over_tau'] = float(ls / max(r.get(f'tau_{o}', 1.0) * 2, 1.0))
    else:
        r['n_acc'] = int(round(r['acc'] * npost * len(cs)))
    return r


if __name__ == '__main__':
    grp = sys.argv[1]
    t0 = time.time()
    F = files()
    out = {}
    for b, f in sorted(F.items()):
        if b.startswith(grp + '_') and not b.endswith('_traces.json'):
            try:
                r = analyze(f)
            except Exception as e:                      # noqa: BLE001
                r = dict(file=f, error=repr(e))
            if r is not None:
                out[b[:-5]] = r
    dump(out, f'cells_{grp}.json')
    print(grp, len(out), 'files', round(time.time() - t0, 1), 's')
