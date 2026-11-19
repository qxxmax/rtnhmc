"""Kish ESS/N and Var(log w) of the raw forward paths (raw_snis/*.npz: 65536 i.i.d. paths per cell, per-path W) with a
delete-one-block jackknife over 64 blocks of 1024 paths (2026-11-17). Writes snis_jk.json.
log w = -W + const, so Var(log w) = Var(W). Kish ESS/N = (sum w)^2 / (N sum w^2).
Caveat: when ESS*N is O(10) the estimator is dominated by a few weights and the jackknife error is itself unreliable
(fig_rawsnis uses ESS < 1e-3 as the cut); such cells are flagged."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jk_lib import REV, jk_err, dump  # noqa: E402

B = 64


def kish(lw):
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


out = {}
for f in sorted(glob.glob(REV + '/raw_snis/*.npz')):
    name = os.path.basename(f)[:-4]
    z = np.load(f)
    W = z['W'].astype(np.float64)
    W = W[np.isfinite(W)]
    n = len(W) // B * B
    lw = -W[:n]
    blk = lw.reshape(B, -1)
    ess_rep = [kish(np.delete(blk, i, 0).ravel()) for i in range(B)]
    var_rep = [np.delete(blk, i, 0).var() for i in range(B)]
    j = f[:-4] + '.json'
    stored = json.load(open(j)).get('kish_ess_frac') if os.path.exists(j) else None
    e = kish(lw)
    v = float(lw.var())
    out[name] = dict(n=n, kish=e, kish_jk=jk_err(ess_rep), kish_stored=stored, kish_nESS=e * n,
                     var_logw=v, var_logw_jk=jk_err(var_rep), exp_minus_var=float(np.exp(-v)),
                     unreliable=bool(e < 1e-3))
dump(out, 'snis_jk.json')
for k in ('E2_L16_k027', 'E2_L8_k027', 'E2_L4_k027', 'E2_L16_k026', 'E2_L16_k030', 'E2_L16_k024'):
    if k in out:
        r = out[k]
        print(k, 'kish %.4f +- %.4f (stored %s)  Var(log w) %.3f +- %.3f' % (r['kish'], r['kish_jk'], r['kish_stored'], r['var_logw'], r['var_logw_jk']))
print(len(out), 'cells')
