"""Jackknife error proxies for tab:udb_compare (App. G; figs/table_udb_compare.json) (2026-11-17).
The table's chains (longchain_udb_20261114, rtp:60) and N=102400 evals are cluster-only; locally the logz_udb_20261116/*.npz
hold, for the same CMCD/DBS models, 4 RT chains x 20000 (acc, |m|) and 65536 fresh forward works W_F. From these:
  acc       block jackknife (20 blocks per chain, 10 % burn-in as logz_estimators)
  tau(|m|)  median over chains of max(Sokal, BM)/2; jackknife over chains of the chain mean (as jk_cells)
  Var(log w), Kish ESS/N   64-block jackknife of W_F
NHMC rows: chain errors from cells_E2.json / cells_buf.json (L8), forward-path errors from snis_jk.json (raw_snis, 65536).
Writes udb_jk.json."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jk_lib import REV, OUT, jk_err, sokal_g, bm_g, dump  # noqa: E402


def kish(lw):
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


TAB = {(r['kappa'], r['sampler']): r for r in json.load(open(REV + '/figs/table_udb_compare.json'))}
out = {}
for f in sorted(glob.glob(REV + '/logz_udb_20261116/*_rt.npz')):
    alg, L, k = os.path.basename(f).split('_')[:3]
    z = np.load(f)
    n = z['acc'].shape[0]; b = int(0.1 * n)
    a = z['acc'][b:].astype(float)
    m = a.shape[0] // 20 * 20
    B = a[:m].reshape(20, -1, a.shape[1]).mean(1).T.ravel()
    M = z['absM'][b:].astype(float)
    tc = np.array([max(sokal_g(M[:, c])[0], bm_g(M[:, c]) or 0) / 2 for c in range(M.shape[1])])
    lw = -z['WF'].astype(float); lw = lw[np.isfinite(lw)]; nn = len(lw) // 64 * 64; blk = lw[:nn].reshape(64, -1)
    r = dict(acc=float(a.mean()), acc_jk=jk_err([(B.sum() - B[i]) / (len(B) - 1) for i in range(len(B))]),
             tau=float(np.median(tc)), tau_jk_chain_mean=jk_err([np.delete(tc, i).mean() for i in range(len(tc))]),
             var_logw=float(lw.var()), var_logw_jk=jk_err([np.delete(blk, i, 0).var() for i in range(64)]),
             kish=kish(lw), kish_jk=jk_err([kish(np.delete(blk, i, 0).ravel()) for i in range(64)]),
             n_chain=int(a.size), n_fwd=int(nn))
    kap = '0.' + k[2:]
    t = TAB.get((kap, alg.upper())) if L == 'L8' else None
    if t:
        r['table'] = dict(var=t['var'], ess=t['ess'], rt_acc=t['rt'] and t['rt'][0], rt_tau=t['rt'] and t['rt'][1])
    out[f'{alg}_{L}_{k}'] = r
cells = {**json.load(open(OUT + '/cells_E2.json')), **json.load(open(OUT + '/cells_buf.json'))}
sn = json.load(open(OUT + '/snis_jk.json'))
for k in ['020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030']:
    t = TAB.get(('0.' + k[1:], 'NHMC'))
    r = dict(table=t)
    for m in ('imhG', 'rt', 'rtp12'):
        c = cells.get(f'E2_L8_k{k}_{m}') or cells.get(f'buf_L8_k{k}_{m}')
        if c:
            r[m] = dict(acc=c['acc'], acc_jk=c.get('acc_jk_block', c['acc_jk_chain']), tau=c['tau_absM'],
                        tau_jk=c['tau_absM_jk_chain_mean'], converged=c['converged'])
    s = sn.get(f'E2_L8_k{k}')
    if s:
        r['fwd_raw_snis'] = dict(var_logw=s['var_logw'], var_logw_jk=s['var_logw_jk'], kish=s['kish'], kish_jk=s['kish_jk'])
    out[f'nhmc_L8_k{k}'] = r
dump(out, 'udb_jk.json')
for k, r in out.items():
    if k.startswith('nhmc'):
        continue
    print(k, 'acc %.3f+-%.3f tau %.2f+-%.2f Var %.3f+-%.3f ESS %.3f+-%.3f' % (r['acc'], r['acc_jk'], r['tau'], r['tau_jk_chain_mean'],
                                                                          r['var_logw'], r['var_logw_jk'], r['kish'], r['kish_jk']),
          'table', {kk: (round(v, 3) if v else v) for kk, v in r.get('table', {}).items()})
