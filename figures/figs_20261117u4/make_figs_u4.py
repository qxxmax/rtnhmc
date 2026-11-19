"""2026-11-17 u4 of figs_20261117u2/make_figs_u2.py: fig_udb_bar only (rect_transfer is at u3, refine_K unchanged at u2).
Change (drawing only, data unchanged): DBS drawn as open squares (u2: filled circles in a blue close to NHMC), the DBS
convention of fig_refine_K.
u2 docstring: 2026-11-17 u2 of figs_20261117u/make_figs_20261117u.py: fig_rect_transfer, fig_udb_bar, fig_refine_K (style unchanged).
Changes against u:
  - rect_transfer: the N_x >= 128 estimates are recomputed here from the npz files (logz_rect2_20261115/R_*_rtp12_s*.npz +
    raw_snis/R_*x8_k0*.npz, logz_estimators.estimates) instead of being read from rect2_bar_summary.json (whose residuals
    use the superseded TI); the estimates are identical to the summary to all digits (checked), so the plot is unchanged.
  - udb_bar: TI error = hypot(se_comb, quad_err) (u: se_comb only, REVIEW_code B3); L16 kappa=0.30 has no quad_err in
    ti_logz.csv (step-halving segment returns None, REVIEW_code B4) -> se_comb only there.
  - refine_K: unchanged.  Every file read is recorded in prov/."""
import os
import sys, json, glob, math, csv, pathlib
import numpy as np, matplotlib
matplotlib.use('pdf'); import matplotlib.pyplot as plt
REV = pathlib.Path(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'); OUT = REV / 'figs_20261117u4'
sys.path.insert(0, str(REV / 'ti_wolff_hmc')); import ti_logz as T
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/revision/longchain'); import logz_estimators as E
sys.path.insert(0, str(OUT)); import u4_common as U; NS = U.NS
NS.apply()
C = {'NHMC': NS.COL['NHMC'], 'CMCD': NS.COL['CMCD'], 'DBS': NS.COL['DBS']}
MK = {'NHMC': 'o', 'CMCD': 'o', 'DBS': 's'}                                  # u4: DBS open square
def save(fig, name):
    NS.save(fig, U.pdf(name))
# ---------------------------------------------------------------- F3 BAR of NHMC / CMCD / DBS
PU = U.Prov('fig_udb_bar')
TIC = {(int(r['L']), round(float(r['kappa']), 4)): r for r in csv.DictReader(open(REV / 'ti_wolff_hmc/ti_logz.csv'))}
tic = {k: (float(r['logZ_comb']), math.hypot(float(r['se_comb']), float(r['quad_err'] or 0))) for k, r in TIC.items()}
udb = json.load(open(REV / 'logz_udb_20261116/udb_bar_summary.json'))
PU.add('a, b', str(REV / 'ti_wolff_hmc/ti_logz.csv'), 'TI reference logZ_comb; error hypot(se_comb, quad_err) (u2; u: se_comb)')
PU.add('a, b', str(REV / 'logz_udb_20261116/udb_bar_summary.json'), 'CMCD / DBS BAR estimates est.bar (residual recomputed with the u2 TI error)')


def udb_resid(key, L, k):
    d = udb[key]
    t = tic[(L, round(k / 100, 4))]
    return d['est']['bar'][0] - t[0], math.hypot(d['est']['bar'][1], t[1])
def nh(L, k, meth):
    f = REV / f'logz_chains/E2_L{L}_k0{k:02d}_{meth}.est.json'
    PU.add('a' if L == 8 else 'b', str(f), 'NHMC BAR estimate (cached est.json, as u)')
    e = json.load(open(f)); t = tic[(L, round(k / 100, 4))]
    return e['bar'][0] - t[0], math.hypot(e['bar'][1], t[1]), e['acc']
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.35), gridspec_kw={'width_ratios': [2.2, 1]})
ks8 = list(range(20, 31))
for i, s in enumerate(('NHMC', 'CMCD', 'DBS')):
    xs, ys, es = [], [], []
    for k in ks8:
        if s == 'NHMC': r, e, _ = nh(8, k, 'rt')
        else: r, e = udb_resid(f'{s.lower()}_L8_k0{k}_rt', 8, k)
        xs.append(k / 100 + (i - 1) * 0.0018); ys.append(r); es.append(e)
    axs[0].errorbar(xs, ys, yerr=es, fmt=MK[s], ms=2.8, color=C[s], mfc='white' if s == 'DBS' else C[s], elinewidth=0.7, label=s)
axs[0].axhline(0, color=NS.ZERO_COLOR, lw=0.6); axs[0].set_xlabel(r'$\kappa$'); axs[0].set_ylabel(r'$\log\hat Z-\log Z_{\rm TI}$')
NS.tag(axs[0], '$L=8$')
ks16 = [24, 27, 30]
for i, s in enumerate(('NHMC', 'CMCD', 'DBS')):
    xs, ys, es = [], [], []
    for jj, k in enumerate(ks16):
        if s == 'NHMC': r, e, _ = nh(16, k, 'rtp12')
        else: r, e = udb_resid(f'{s.lower()}_L16_k0{k}_rt', 16, k)
        xs.append(jj + (i - 1) * 0.18); ys.append(r); es.append(e)
    axs[1].errorbar(xs, ys, yerr=es, fmt=MK[s], ms=2.8, color=C[s], mfc='white' if s == 'DBS' else C[s], elinewidth=0.7)
axs[1].axhline(0, color=NS.ZERO_COLOR, lw=0.6); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(['0.24', '0.27', '0.30']); axs[1].set_xlabel(r'$\kappa$')
NS.tag(axs[1], '$L=16$')
h, _ = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
NS.fig_legend(fig, h, ncol=3)
for a_, l in zip(axs, 'ab'): NS.panel_letter(a_, l)
save(fig, 'fig_udb_bar')
PU.note('u4: DBS open squares')
PU.note('L16 kappa=0.30: quad_err empty in ti_logz.csv -> se_comb only')
PU.dump()
