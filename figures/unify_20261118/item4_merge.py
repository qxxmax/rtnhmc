"""Item 4: merge 1D (Mac) and eq_w2 (HIM) results -> per seed and per method ranges, old vs common reference."""
import os
import json, numpy as np
H = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/'
d1 = json.load(open(H + 'item4_lj13_1d.json')); eq = json.load(open(H + 'item4_eq_w2_common.json')); fl = json.load(open(H + 'item4_lj13_floor.json'))
meth = {'path-IMH': 'A_imh', 'RT': 'A_rt', 'pRT': 'A_rt_partial8'}
out = dict(per_seed={}, per_method={})
for m, p in meth.items():
    rows = {}
    for r in range(3):
        k = f'{p}_r{r}'; q = d1['runs'][k]
        e_new = float(np.mean([eq[f'{k}_rep{i}_gen']['eq_w2'] for i in range(3)]))
        rows[f'r{r}'] = dict(energy_w2_old=q['energy_w2_old'], energy_w2_common=q['energy_w2_common'], energy_w2_fullref=q['energy_w2_fullref'],
                             dist_w2_old=q['dist_w2_old'], dist_w2_common=q['dist_w2_common'], dist_w2_fullref=q['dist_w2_fullref'],
                             eq_w2_old=q['eq_w2_old'], eq_w2_common=e_new)
    out['per_seed'][m] = rows
    out['per_method'][m] = {f: [min(v[f] for v in rows.values()), max(v[f] for v in rows.values())] for f in rows['r0']}
    out['per_method'][m]['energy_w2_common_mean'] = float(np.mean([v['energy_w2_common'] for v in rows.values()]))
rt = [v['energy_w2_common'] for m in ('RT', 'pRT') for v in out['per_seed'][m].values()]
im = [v['energy_w2_common'] for v in out['per_seed']['path-IMH'].values()]
diff = np.mean(im) - np.mean(rt); se = np.sqrt(np.var(im, ddof=1) / 3 + np.var(rt, ddof=1) / 6)
out['gap_common'] = dict(imh_mean=float(np.mean(im)), roundtrip_mean=float(np.mean(rt)), diff=float(diff), se_seed=float(se), z=float(diff / se),
                         min_imh_minus_max_roundtrip=float(min(im) - max(rt)))
for key in ('energy_w2_fullref',):
    a = [v[key] for v in out['per_seed']['path-IMH'].values()]; b = [v[key] for m in ('RT', 'pRT') for v in out['per_seed'][m].values()]
    out['gap_fullref'] = dict(imh=[min(a), max(a)], roundtrip=[min(b), max(b)], diff=float(np.mean(a) - np.mean(b)),
                              se_seed=float(np.sqrt(np.var(a, ddof=1) / 3 + np.var(b, ddof=1) / 6)))
out['paired_refboot'] = d1['paired_refboot']; out['floor'] = fl
out['common_ref_energy_mean'] = d1['common_ref_energy_mean']
print(json.dumps({k: out[k] for k in ('per_method', 'gap_common', 'gap_fullref')}, indent=1))
json.dump(out, open(H + 'item4_lj13_summary.json', 'w'), indent=1)
