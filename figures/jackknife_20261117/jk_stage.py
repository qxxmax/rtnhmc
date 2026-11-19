"""Jackknife over training seeds / (L,T) cells for the path-number scaling of Var(log w) at kappa=0.27 (2026-11-17).
Inputs (read only): the seed values hard-coded in figs_20261116/make_stage_scan_20261117.py (variant 'fig'), and the
eval_N262144 values collected by fig_scaling_20261116.py into scaling_summary_20261116.json['varW_k27'] (variant 'files';
T=64 is not in the synced files, so the App. D text values L4 T64 = 0.139/0.143/0.144 and L8 T64 = 0.382 are added and
labelled). Writes stage_jk.json.
  slope_T          polyfit(log T, log mean_seeds Var) for L=8 (fig_stage_scan)       jackknife: delete-one seed value
                   (a T with one seed is dropped when that seed is deleted) and delete-one T
  global fit       log Var = a + p log V + b log T over all (L,T) cell means          jackknife: delete-one seed, delete-one cell
  per-L T slope    L = 4, 6, 8                                                         same
  slopes 0.0165    (Var_T16(L8)-Var_T16(L4))/48, (Var_T16(L16)-Var_T16(L8))/192 and the intercept (0.16); seed jackknife,
                   L16 single model: sampling error 0.020 from the 65536-path raw_snis block jackknife (snis_jk.json)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jk_lib import REV, OUT, jk_err, dump  # noqa: E402

FIG = {4: [3.517, 3.510, 3.502], 8: [2.156, 2.147, 2.120], 16: [1.231, 1.206, 1.222], 32: [0.674, 0.682, 0.681], 64: [0.382]}
VW = json.load(open(REV + '/figs_20261116/scaling_summary_20261116.json'))['varW_k27']
TEXT_T64 = {(4, 64): [0.139, 0.143, 0.144], (8, 64): [0.382]}


def slope(seeds):
    Ts = np.array(sorted(t for t in seeds if seeds[t]))
    m = np.array([np.mean(seeds[t]) for t in Ts])
    return float(np.polyfit(np.log(Ts), np.log(m), 1)[0])


def seed_jk(seeds, stat):
    reps = []
    for t in seeds:
        for i in range(len(seeds[t])):
            s = {u: (list(v) if u != t else v[:i] + v[i + 1:]) for u, v in seeds.items()}
            reps.append(stat(s))
    return jk_err(reps)


def cell_jk(seeds, stat):
    return jk_err([stat({u: v for u, v in seeds.items() if u != t}) for t in seeds])


res = {}
for tag, s in (('fig', FIG), ('files_L8_T4-32', {int(k.split('_')[1]): v for k, v in VW.items() if k.startswith('8_')}),
               ('files_L8_T4-32+textT64', {**{int(k.split('_')[1]): v for k, v in VW.items() if k.startswith('8_')}, 64: [0.382]})):
    res[f'slope_T_L8_{tag}'] = dict(value=slope(s), jk_seed=seed_jk(s, slope), jk_T=cell_jk(s, slope),
                                    means={t: float(np.mean(v)) for t, v in sorted(s.items())})

cells = {}
for k, v in VW.items():
    L, T = map(int, k.split('_'))
    if L in (4, 6, 8):
        cells[(L, T)] = list(v)
cells.update(TEXT_T64)


def gfit(c):
    keys = [k for k in sorted(c) if c[k]]
    X = np.array([[1, np.log(L * L), np.log(T)] for L, T in keys])
    y = np.log([np.mean(c[k]) for k in keys])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, float(np.sqrt(np.mean((y - X @ beta) ** 2)))


b0, rms0 = gfit(cells)
reps_s, reps_c = [], []
for k in cells:
    for i in range(len(cells[k])):
        c = dict(cells); c[k] = cells[k][:i] + cells[k][i + 1:]
        reps_s.append(gfit(c)[0])
    reps_c.append(gfit({u: v for u, v in cells.items() if u != k})[0])
reps_s, reps_c = np.array(reps_s), np.array(reps_c)
res['global_fit'] = dict(V_exp=b0[1], V_exp_jk_seed=jk_err(reps_s[:, 1]), V_exp_jk_cell=jk_err(reps_c[:, 1]),
                         T_exp=b0[2], T_exp_jk_seed=jk_err(reps_s[:, 2]), T_exp_jk_cell=jk_err(reps_c[:, 2]),
                         log_rms=rms0, n_cells=len(cells), note='cells L=4,6,8 x T=4..32 (files) + T=64 text values')
for L in (4, 6, 8):
    s = {T: v for (LL, T), v in cells.items() if LL == L}
    res[f'slope_T_L{L}'] = dict(value=slope(s), jk_seed=seed_jk(s, slope), jk_T=cell_jk(s, slope), Ts=sorted(s))
    s2 = {T: v for T, v in s.items() if T != 64}
    res[f'slope_T_L{L}_noT64'] = dict(value=slope(s2), jk_seed=seed_jk(s2, slope), jk_T=cell_jk(s2, slope))
# T^{-b}  ->  T propto V^{1/b}
b = -res['slope_T_L8_fig']['value']; eb = np.hypot(res['slope_T_L8_fig']['jk_seed'], res['slope_T_L8_fig']['jk_T'])
res['T_propto_V_exp'] = dict(value=1 / b, err=eb / b ** 2, from_='fig slope, seed+T jackknife in quadrature')
# slopes 0.0165 and the constant 0.16 (T = 16)
v4, v8 = VW['4_16'], VW['8_16']
v16, e16 = VW['16_16'][0], json.load(open(OUT + '/snis_jk.json'))['E2_L16_k027']['var_logw_jk']


def lin(v4m, v8m, v16m):
    s1 = (v8m - v4m) / 48; s2 = (v16m - v8m) / 192
    return s1, s2, v4m - 16 * s1
f0 = lin(np.mean(v4), np.mean(v8), v16)
r4 = [lin(np.mean(np.delete(v4, i)), np.mean(v8), v16) for i in range(3)]
r8 = [lin(np.mean(v4), np.mean(np.delete(v8, i)), v16) for i in range(3)]
err = [np.sqrt(jk_err([r[j] for r in r4]) ** 2 + jk_err([r[j] for r in r8]) ** 2
               + ((lin(np.mean(v4), np.mean(v8), v16 + e16)[j] - lin(np.mean(v4), np.mean(v8), v16 - e16)[j]) / 2) ** 2) for j in range(3)]
res['slopes_T16'] = dict(slope_4_8=f0[0], slope_4_8_err=err[0], slope_8_16=f0[1], slope_8_16_err=err[1],
                         intercept=f0[2], intercept_err=err[2], var_T16={'4': np.mean(v4), '8': np.mean(v8), '16': v16})
res = {k: ({kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
       for k, v in res.items()}
dump(res, 'stage_jk.json')
for k, v in res.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'means'})
