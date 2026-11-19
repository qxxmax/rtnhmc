#!/usr/bin/env python3
"""Summary of improve/cmcd_torch/runs/*/rt_chains.json: accuracy (z vs the HMC+Z2 reference of rt_cmcd_phi4.py,
within-run: combined within-run SE + ref SE; between-run: sd(run means)/sqrt(R) + ref SE) and cost
(tau per transition = max(per-chain-centred, globally centred) where the global one is stored (sign), C_eff per path = tau x paths/transition, C_eff per force eval = x force evals/path)."""
import glob
import json
import math
import os

import numpy as np

KEYS = ('absM', 'M2', 'action', 'sign')
import sys
# repo copy: python summ_cmcd.py <dir> (default ./runs); globs <dir>/*z2_k*_it10000/rt_chains.json
ROOT = sys.argv[1] if len(sys.argv) > 1 else 'runs'
for f in sorted(glob.glob(os.path.join(ROOT, '*z2_k*_it10000', 'rt_chains.json'))):
    d = json.load(open(f))
    ref, rse, fpp = d['ref'], d['ref_se'], d['force_evals_per_path']
    print(f'\n== {os.path.basename(os.path.dirname(f))}  ref {os.path.basename(d["reference"])}  force evals/path {fpp}')
    for m, e in d['methods'].items():
        runs = e['runs']
        if m == 'snis':
            ess = [r.get('kish_ess_frac', float('nan')) for r in runs]
            row = f'{m:14s} ESS/N ' + ' '.join(f'{x:.3f}' for x in ess)
            for k in KEYS:
                if k in runs[0]:
                    mu = np.array([r[k]['mean'] for r in runs])
                    sb = mu.std(ddof=1) / math.sqrt(len(mu))
                    row += f' | {k} {mu.mean():.5f} z_b {(mu.mean() - ref[k][0]) / math.sqrt(sb ** 2 + rse[k] ** 2):+.1f}'
            print(row)
            continue
        ppt = runs[0]['paths_per_transition']
        row = f'{m:14s} acc {np.mean([r["acceptance"] for r in runs]):.3f} paths/tr {ppt:.3f} maxhold ' + \
              ','.join(str(r['holding']['max_run']) for r in runs)
        for k in KEYS:
            mu = np.array([r[k]['mean'] for r in runs])
            # Table II convention: tau = max(per-chain-centred tau, globally centred tau) where the latter is stored
            taus = np.array([max(r[k]['tau_int'], r[k].get('tau_global', 0.0)) for r in runs])
            se = np.array([math.sqrt(r[k]['var'] * t / (r[k]['n_eff'] * r[k]['tau_int'])) for r, t in zip(runs, taus)])
            sw = math.sqrt((se ** 2).sum()) / len(runs)
            sb = mu.std(ddof=1) / math.sqrt(len(runs))
            tau = float(taus.mean())
            zw = (mu.mean() - ref[k][0]) / math.sqrt(sw ** 2 + rse[k] ** 2)
            zb = (mu.mean() - ref[k][0]) / math.sqrt(sb ** 2 + rse[k] ** 2)
            row += f' | {k} {mu.mean():.5f} z_w {zw:+.1f} z_b {zb:+.1f} tau {tau:.2f} Cpath {tau * ppt:.2f} Cfe {tau * ppt * fpp:.0f}'
        print(row)
