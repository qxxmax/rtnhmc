"""2026-11-17 u3 of figs_20261117u2/make_stage_scan_u2.py: data and fits unchanged; only the legend handle of L=16 is
drawn open (all L=16 points are single-seed models and open in the plot; u2 showed a filled handle).
u2 docstring: 2026-11-17 u2 of figs_20261117u/make_stage_scan_20261117u.py (fig_stage_scan).
u hard-coded L=8 values (T=8 seed 0 = 2.156, T=64 = 0.382, ...); u2 reads every value from eval_N262144.json with the
seed rule of u2_common.kscan_points (seed 0 iqcb phi4_buffer GPU pipeline, seeds 1, 2 HIM kscan_cpu) -- the same rule as
fig_scaling_var (c).  The figure now shows all sizes with a T scan at kappa = 0.27 (L = 4, 6, 8, 16; u: L = 8 only),
mean over seeds +- standard error (single-seed points open, no bar), dashed: per-L power-law fit Var = A T^b.
Fits (-> derived_stage_u2.json):
  per-L exponent b_L: polyfit(log T, log seed-mean) over all T of that L; jackknife error = delete-one-seed (cells with one
                      seed are dropped in that replica) and delete-one-T, combined in quadrature;
  global: log Var = a + p log V + b log T, over L = 4, 6, 8 (as in the text) and over L = 4, 6, 8, 16; same jackknife;
  T = 16 line: (Var(L8) - Var(L4))/48, (Var(L16) - Var(L8))/192 and intercept Var(L4) - 16 slope (seed jackknife;
               L16 is a single model: sampling error from jackknife_20261117/snis_jk.json as in stage_jk.py)."""
import os
import json
import sys

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261117u3')
import u3_common as U  # noqa: E402
NS = U.NS
NS.apply()
pr = U.Prov('fig_stage_scan')
KS = U.kscan_points()
CELLS = {k: [c['var'] for c in v] for k, v in KS['cells'].items()}
for (L, T), v in sorted(KS['cells'].items()):
    for c in v:
        pr.add(f'L={L}', c['path'], KS['rule'], f'T={T} seed={c["seed"]} ({c["src"]}) Var={c["var"]:.4f}' + (' NEW vs u' if c['new'] else ''))
for c in KS['excluded']:
    pr.add('excluded', c['path'], KS['rule'], c['why'])
SCAN_L = [L for L in (4, 6, 8, 16) if len({T for (LL, T) in CELLS if LL == L}) >= 3]


def jk(reps):
    t = np.asarray(reps, float)
    n = len(t)
    return float(np.sqrt((n - 1) / n * ((t - t.mean()) ** 2).sum())) if n > 1 else float('nan')


def slope(c):
    Ts = np.array(sorted(c))
    m = np.array([np.mean(c[t]) for t in Ts])
    return float(np.polyfit(np.log(Ts), np.log(m), 1)[0])


def errs(c, stat):
    rs = []
    for t in c:
        for i in range(len(c[t])):
            d = {u: (list(v) if u != t else v[:i] + v[i + 1:]) for u, v in c.items()}
            d = {u: v for u, v in d.items() if v}
            rs.append(stat(d))
    rt = [stat({u: v for u, v in c.items() if u != t}) for t in c]
    return jk(rs), jk(rt)


def gfit(c):
    keys = sorted(c)
    X = np.array([[1, np.log(L * L), np.log(T)] for L, T in keys])
    y = np.log([np.mean(c[k]) for k in keys])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, float(np.sqrt(np.mean((y - X @ beta) ** 2)))


R = {}
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for L in SCAN_L:
    c = {T: v for (LL, T), v in CELLS.items() if LL == L}
    Ts = np.array(sorted(c))
    m = np.array([np.mean(c[t]) for t in Ts])
    se = np.array([np.std(c[t], ddof=1) / np.sqrt(len(c[t])) if len(c[t]) > 1 else 0 for t in Ts])
    one = np.array([len(c[t]) == 1 for t in Ts])
    col = NS.LCOL[L]
    ax.errorbar(Ts[~one], m[~one], yerr=se[~one], fmt='o', color=col, ms=3.5, capsize=1.5, elinewidth=0.7)
    ax.plot(Ts[one], m[one], 'o', color=col, mfc='white', ms=3.5)
    b, a = np.polyfit(np.log(Ts), np.log(m), 1)
    tt = np.geomspace(Ts.min() * 0.85, Ts.max() * 1.18)
    ax.plot(tt, np.exp(a) * tt ** b, color=col, lw=0.8, ls=NS.FIT_LS)
    es, et = errs(c, slope)
    R[f'T_exp_L{L}'] = dict(value=float(b), jk_seed=es, jk_T=et, total=float(np.hypot(np.nan_to_num(es), et)), Ts=[int(t) for t in Ts],
                            n_seeds={int(t): len(c[t]) for t in Ts}, means={int(t): float(np.mean(c[t])) for t in Ts},
                            se={int(t): (float(np.std(c[t], ddof=1) / np.sqrt(len(c[t]))) if len(c[t]) > 1 else None) for t in Ts})
    if 64 in c:
        c2 = {t: v for t, v in c.items() if t != 64}
        es2, et2 = errs(c2, slope)
        R[f'T_exp_L{L}_noT64'] = dict(value=slope(c2), jk_seed=es2, jk_T=et2)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
TT = sorted({T for (L, T) in CELLS if L in SCAN_L})
ax.set_xticks(TT)
ax.set_xticklabels([str(t) for t in TT])
ax.minorticks_off()
ax.set_yticks([0.2, 0.5, 1, 2, 5])
ax.set_yticklabels(['0.2', '0.5', '1', '2', '5'])
ax.set_xlabel('$T$')
ax.set_ylabel(r'Var$(\log w)$')
NS.tag(ax, r'$\kappa=0.27$', loc='upper right')
h = [plt.Line2D([], [], color=NS.LCOL[L], marker='o', ms=3.5, ls=NS.FIT_LS, lw=0.8, label=f'$L={L}$',
                mfc='white' if all(len(CELLS[(L, T)]) == 1 for (LL, T) in CELLS if LL == L) else NS.LCOL[L]) for L in SCAN_L]  # u3
ax.legend(handles=h, loc='lower left', ncol=2, columnspacing=1.0)
fig.tight_layout()
NS.save(fig, U.pdf('fig_stage_scan'))

for tag, Lset in (('L4-8', (4, 6, 8)), ('L4-16', (4, 6, 8, 16))):
    c = {k: v for k, v in CELLS.items() if k[0] in Lset and k[1] in (4, 8, 16, 32, 64) and
         len({T for (LL, T) in CELLS if LL == k[0]}) >= 3}
    beta, rms = gfit(c)
    rs, rc = [], []
    for k in c:
        for i in range(len(c[k])):
            d = dict(c)
            d[k] = c[k][:i] + c[k][i + 1:]
            d = {u: v for u, v in d.items() if v}
            rs.append(gfit(d)[0])
        rc.append(gfit({u: v for u, v in c.items() if u != k})[0])
    rs, rc = np.array(rs), np.array(rc)
    R[f'global_{tag}'] = dict(V_exp=float(beta[1]), V_exp_jk_seed=jk(rs[:, 1]), V_exp_jk_cell=jk(rc[:, 1]),
                              T_exp=float(beta[2]), T_exp_jk_seed=jk(rs[:, 2]), T_exp_jk_cell=jk(rc[:, 2]),
                              log_rms=rms, n_cells=len(c), cells=sorted(f'{L}_{T}' for L, T in c))
    R[f'global_{tag}']['T_at_fixed_Var_propto_V_exp_powerlaw'] = float(beta[1] / -beta[2])
    R[f'global_{tag}']['one_over_abs_T_exp'] = float(1 / -beta[2])
    R[f'global_{tag}']['one_over_abs_T_exp_err'] = float(np.hypot(jk(rs[:, 2]), jk(rc[:, 2])) / beta[2] ** 2)

# T = 16 straight line in V
v4, v8, v16 = CELLS[(4, 16)], CELLS[(8, 16)], CELLS[(16, 16)]
e16 = json.load(open(U.JK + '/snis_jk.json'))['E2_L16_k027']['var_logw_jk']


def lin(a4, a8, a16):
    s1 = (a8 - a4) / 48
    s2 = (a16 - a8) / 192
    return s1, s2, a4 - 16 * s1


f0 = lin(np.mean(v4), np.mean(v8), np.mean(v16))
r4 = [lin(np.mean(np.delete(v4, i)), np.mean(v8), np.mean(v16)) for i in range(len(v4))]
r8 = [lin(np.mean(v4), np.mean(np.delete(v8, i)), np.mean(v16)) for i in range(len(v8))]
E = [float(np.sqrt(jk([r[j] for r in r4]) ** 2 + jk([r[j] for r in r8]) ** 2 +
                   ((lin(np.mean(v4), np.mean(v8), v16[0] + e16)[j] - lin(np.mean(v4), np.mean(v8), v16[0] - e16)[j]) / 2) ** 2)) for j in range(3)]
R['T16_line'] = dict(slope_4_8=f0[0], slope_4_8_err=E[0], slope_8_16=f0[1], slope_8_16_err=E[1], intercept=f0[2], intercept_err=E[2],
                     var={'4': float(np.mean(v4)), '8': float(np.mean(v8)), '16': float(np.mean(v16))})
# deviations used in the text: same T/V for L4 and L8; prediction for T=4
dev = {}
for (La, Ta), (Lb, Tb) in (((4, 4), (8, 16)), ((4, 8), (8, 32)), ((4, 16), (8, 64))):
    if (La, Ta) in CELLS and (Lb, Tb) in CELLS:
        dev[f'L{La}T{Ta}_vs_L{Lb}T{Tb}'] = float(np.mean(CELLS[(Lb, Tb)]) / np.mean(CELLS[(La, Ta)]) - 1)
R['same_T_over_V_rel_diff'] = dev
p = R['global_L4-8']['T_exp']
pred = {}
for L in (4, 6, 8):
    v16m = np.mean(CELLS[(L, 16)])
    pred[f'L{L}'] = float(v16m * (4 / 16) ** p / np.mean(CELLS[(L, 4)]) - 1)
R['T4_prediction_rel_dev_globalT'] = pred
pred2 = {}
for L in (4, 6, 8):
    lv = (0.0165 * L * L + 0.16)
    pred2[f'L{L}'] = float(lv * (4 / 16) ** -0.81 / np.mean(CELLS[(L, 4)]) - 1)
R['T4_prediction_rel_dev_textformula'] = pred2
rel = [np.std(v, ddof=1) / np.mean(v) for v in CELLS.values() if len(v) > 1]
R['seed_rel_sd'] = dict(median=float(np.median(rel)), max=float(np.max(rel)))
R['missing'] = ('L8 T64 seeds 1, 2 (HIM 93231/93232/93718/93719) still training on 2026-11-17 (latest epochs 8600-17800 of 20000), '
                'no eval_N262144.json; L6 T64 seed 1 (HIM 93867) trained to 20000 but not evaluated; L6 T64 seed 0 absent')
R['rule'] = KS['rule']
json.dump(R, open(U.OUT + '/derived_stage_u3.json', 'w'), indent=1, default=float)
pr.note('figure now shows L = 4, 6, 8, 16 (u: L = 8 only, hard-coded values)')
pr.note(R['missing'])
pr.dump(extra=R)
for k, v in R.items():
    print(k, v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ('means', 'se', 'cells')})
