"""App. F (MW / LJ13) numbers with a jackknife over independent runs / seeds (2026-11-17; same files as
figs/dw_lj_figs_20261116.py). Writes dwlj_jk.json.
  MW mode-change rate per MC step: delete-one-run jackknife of the mean over runs (fig (c) shows np.std, a spread).
  MW acceptance: same.
  LJ13 stationary energy W2 (n = 2000 vs PT reference): per seed the stored mean over 20 reference subsamples; jackknife
  over the 3 seeds of the seed mean; the reference floor (floor20_lj13.json) is reported with its sd as stored; also
  dist W2 and geometry (eq) W2."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jk_lib import REV, jk_err, dump  # noqa: E402


def mjk(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), jk=jk_err([np.delete(v, i).mean() for i in range(len(v))]) if len(v) > 1 else None,
                n=len(v), min=float(v.min()), max=float(v.max()))


out = {'MW': {}, 'LJ13': {}}
for sysn in ('dw4', 'dw8'):
    for f in sorted(glob.glob(f'{REV}/dw_noHMC/{sysn}_A_*.json')):
        for m, v in json.load(open(f))['methods'].items():
            if v.get('runs') and 'mode_change_rate' in v['runs'][0]:
                out['MW'][f'{sysn}_{m}'] = dict(mode_change_rate=mjk([r['mode_change_rate'] for r in v['runs']]),
                                                acceptance=mjk([r['acceptance'] for r in v['runs']]) if 'acceptance' in v['runs'][0] else None)
    for fl in (0, 1):
        h = json.load(open(f'{REV}/dw/{sysn}_hmc_flip{fl}.json'))
        out['MW'][f'{sysn}_hmc_flip{fl}'] = dict(mode_change_rate=mjk([float(r['mode_change_rate']) for r in h['runs']]),
                                               acceptance=mjk([float(r['acceptance']) for r in h['runs']]) if 'acceptance' in h['runs'][0] else None)
NH = f'{REV}/lj13_noHMC_93492_20261116/lj13_noHMC'
for lab, pre in (('IMH', 'A_imh_r'), ('RT', 'A_rt_r'), ('pRT', 'A_rt_partial8_r')):
    fs = sorted(glob.glob(f'{NH}/{pre}[0-9]/res/*lit2k.json'))
    S = [json.load(open(f))['summary'] for f in fs]
    out['LJ13'][lab] = {q: mjk([s[q]['mean'] for s in S]) for q in ('energy_w2', 'dist_w2', 'eq_w2')}
    out['LJ13'][lab]['per_seed_ref_sd'] = {q: [s[q]['sd'] for s in S] for q in ('energy_w2', 'dist_w2', 'eq_w2')}
fl = f'{REV.rsplit("/", 1)[0]}/particle_202610/lit/stat2/floor20_lj13.json'
if os.path.exists(fl):
    out['LJ13']['floor'] = json.load(open(fl))['summary']
dump(out, 'dwlj_jk.json')
for k, v in out['MW'].items():
    r = v['mode_change_rate']
    print(k, 'mode-change %.4g +- %s (n=%d, range %.3g-%.3g)' % (r['mean'], r['jk'] and '%.2g' % r['jk'], r['n'], r['min'], r['max']),
          v['acceptance'] and 'acc %.3f +- %.3f' % (v['acceptance']['mean'], v['acceptance']['jk'] or 0))
for k, v in out['LJ13'].items():
    if k == 'floor':
        print('floor', {q: v[q] for q in ('energy_w2', 'dist_w2', 'eq_w2') if q in v})
    else:
        print(k, {q: '%.4f +- %.4f (seeds %.3f-%.3f)' % (v[q]['mean'], v[q]['jk'] or 0, v[q]['min'], v[q]['max']) for q in ('energy_w2', 'dist_w2', 'eq_w2')})
