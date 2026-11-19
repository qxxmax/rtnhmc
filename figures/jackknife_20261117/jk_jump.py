"""Accepted-jump size of |m| (section_S_zh.tex:82,90; section_IV_zh.tex:44; fig new_fig_acceptance (b)) with a block jackknife
(20 contiguous blocks per chain, all chains pooled; 2026-11-17). Same definition as phi4_figs_20261116.accept_quality:
  J_indep = mean |d|m|| over accepted transitions / E|X - X'|  (independent pair; here the exact Gini mean difference of the
            pooled post-burn |m| instead of a random permutation; computed on every 4th record to stay light)
  J_sd    = mean |d|m|| over accepted transitions / sd(|m|)   (text: 'about 1.1 sd, cf. 2/sqrt(pi) = 1.13')
Writes jump_jk.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jk_lib import OUT, jk_err, dump  # noqa: E402

CELLS = {**json.load(open(OUT + '/cells_E2.json')), **json.load(open(OUT + '/cells_buf.json'))}
NB = 20


def gini(y):
    y = np.sort(y)
    n = len(y)
    return float(2 * ((2 * np.arange(1, n + 1) - n - 1) * y).sum() / (n * (n - 1)))


out = {}
for L in (8, 16):
    for k in ('24', '27', '30'):
        for m in ('rt', 'rtp12', 'imhG'):
            c = CELLS.get(f'E2_L{L}_k0{k}_{m}') or CELLS.get(f'buf_L{L}_k0{k}_{m}')
            if not c or not c.get('traces'):
                continue
            z = np.load(c['traces'])
            b, N = c['burn'], c['n_transitions']
            M = z['absM'][b:N].astype(float)
            acc = z['acc'][b + 1:N].astype(bool)
            dM = np.abs(np.diff(M, axis=0))
            n = (M.shape[0] - 1) // NB * NB
            bl = n // NB
            # per block, per chain: sum of accepted jumps, count, and the |m| values
            S = np.array([[dM[i * bl:(i + 1) * bl, ch][acc[i * bl:(i + 1) * bl, ch]].sum() for i in range(NB)] for ch in range(M.shape[1])]).ravel()
            K = np.array([[acc[i * bl:(i + 1) * bl, ch].sum() for i in range(NB)] for ch in range(M.shape[1])]).ravel()
            Y = [M[i * bl:(i + 1) * bl, ch][::4] for ch in range(M.shape[1]) for i in range(NB)]

            def stat(keep):
                y = np.concatenate([Y[i] for i in keep])
                j = S[keep].sum() / max(K[keep].sum(), 1)
                return j / gini(y), j / y.std()
            allb = np.arange(len(S))
            full = stat(allb)
            reps = np.array([stat(np.delete(allb, i)) for i in range(len(S))])
            out[f'L{L}_k{k}_{m}'] = dict(J_indep=full[0], J_indep_jk=jk_err(reps[:, 0]), J_sd=full[1], J_sd_jk=jk_err(reps[:, 1]))
dump(out, 'jump_jk.json')
for k, v in out.items():
    print(k, 'J/indep %.3f +- %.3f   J/sd %.3f +- %.3f' % (v['J_indep'], v['J_indep_jk'], v['J_sd'], v['J_sd_jk']))
