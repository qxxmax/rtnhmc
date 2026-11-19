"""2026-11-17 u2 of figs_20261117u/make_figs_20261117u.py: fig_rect_transfer, fig_udb_bar, fig_refine_K (style unchanged).
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
REV = pathlib.Path(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'); OUT = REV / 'figs_20261117u2'
sys.path.insert(0, str(REV / 'ti_wolff_hmc')); import ti_logz as T
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/revision/longchain'); import logz_estimators as E
sys.path.insert(0, str(OUT)); import u2_common as U; NS = U.NS
NS.apply()
C = {'NHMC': NS.COL['NHMC'], 'CMCD': NS.COL['CMCD'], 'DBS': NS.COL['DBS']}
def save(fig, name):
    NS.save(fig, U.pdf(name))
# ---------------------------------------------------------------- F2 rectangular transfer, new TI
def load(pat):
    s = {}
    for f in glob.glob(pat):
        d = json.load(open(f)); s[round(d['params']['kappa'] * 1000)] = (d['observables']['hop']['mean'], d['observables']['hop']['err'])
    return s
def ti(s, kmax):
    def integ(stride):
        tot, var = 0, 0
        for seg in (list(range(0, 251, 5)), list(range(250, kmax + 1))):
            seg = [i for i in seg if i <= kmax][::stride]
            if len(seg) < 2: continue
            w = T.weights(len(seg) - 1, (seg[1] - seg[0]) * 1e-3)
            tot += float(w @ np.array([s[i][0] for i in seg])); var += float(((w * np.array([s[i][1] for i in seg])) ** 2).sum())
        return tot, math.sqrt(var)
    x, e = integ(1); y, _ = integ(2)
    return T.LOGZ1 + 2 * x, math.hypot(2 * e, 2 * abs(x - y) / 15)
PR = U.Prov('fig_rect_transfer')
_EST = {}


def rect_est(n, k):
    if (n, k) not in _EST:
        fs = sorted(glob.glob(str(REV / f'logz_rect2_20261115/R_{n}x8_k0{k}_rtp12_s*.npz')))
        zs = [np.load(f) for f in fs]
        wf = REV / f'raw_snis/R_{n}x8_k0{k}.npz'
        for f in fs:
            PR.add(f'k0.{k}', f, 'reverse records: 4 seeds x 4 chains x 2e4 (rtp12)')
        PR.add(f'k0.{k}', str(wf), 'forward work, 65536 paths (estimator uses 65520 = 20 blocks)')
        _EST[(n, k)] = E.estimates(dict(WR=np.concatenate([z['WR'] for z in zs], 1), WF=np.load(wf)['W'],
                                        absM=np.concatenate([z['absM'] for z in zs], 1), acc=np.concatenate([z['acc'] for z in zs], 1)))
    return _EST[(n, k)]


NX = [64, 128, 256, 512]
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.35), sharex=True)
B = NS.BAR_STYLE
for j, k in enumerate((20, 30)):
    ax = axs[j]; tiv = {n: ti(load(str(REV / f'ti_rect2_20261116/*R{n}x8_k*_wolff2.json')), k * 10) for n in NX}
    for n in NX:
        PR.add(f'k0.{k} TI', str(REV / 'ti_rect2_20261116'), f'TI of R{n}x8: *_wolff2.json files (101 per N_x), Simpson + 3/8, step-halving /15')
    band = np.array([tiv[n][1] for n in NX]) * 1e4
    ax.fill_between(NX, -band, band, color=NS.BAND, lw=0); ax.axhline(0, color=NS.ZERO_COLOR, lw=0.6)
    for i, (key, lab) in enumerate((('snis', 'forward'), ('rev', 'reverse'), ('bar', 'BAR'))):
        st = B[lab]
        xs, ys, es = [], [], []
        for n in NX:
            V = n * 8
            if n == 64:
                e = json.load(open(REV / f'logz_chains/R_64x8_k0{k}_rtp12.est.json')); v = e[key]
                PR.add(f'k0.{k}', str(REV / f'logz_chains/R_64x8_k0{k}_rtp12.est.json'), '64x8: cached estimate (as u)')
            else: v = rect_est(n, k)[key]
            xs.append(n * 2 ** ((i - 1) * 0.06)); ys.append((v[0] / V - tiv[n][0]) * 1e4); es.append(v[1] / V * 1e4)
        ax.errorbar(xs, ys, yerr=es, fmt=st['marker'], ms=3.4, color=st['color'], mfc='white' if st['open'] else st['color'],
                    elinewidth=0.7, capsize=0, label=lab)
    ax.set_xscale('log', base=2); ax.set_xticks(NX); ax.set_xticklabels([str(n) for n in NX]); ax.minorticks_off()
    ax.set_xlabel('$N_x$'); NS.tag(ax, f'$\\kappa={k / 100:.2f}$')
    ax.set_ylim(-7, 7); ax.set_xlim(52, 630)
axs[0].set_ylabel(r'$(\log\hat Z-\log Z_{\rm TI})/V\ [10^{-4}]$')
h, _ = axs[1].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
NS.fig_legend(fig, h, ncol=3)
for a_, l in zip(axs, 'ab'): NS.panel_letter(a_, l)
save(fig, 'fig_rect_transfer')
PR.dump()
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
    axs[0].errorbar(xs, ys, yerr=es, fmt='o', ms=2.8, color=C[s], elinewidth=0.7, label=s)
axs[0].axhline(0, color=NS.ZERO_COLOR, lw=0.6); axs[0].set_xlabel(r'$\kappa$'); axs[0].set_ylabel(r'$\log\hat Z-\log Z_{\rm TI}$')
NS.tag(axs[0], '$L=8$')
ks16 = [24, 27, 30]
for i, s in enumerate(('NHMC', 'CMCD', 'DBS')):
    xs, ys, es = [], [], []
    for jj, k in enumerate(ks16):
        if s == 'NHMC': r, e, _ = nh(16, k, 'rtp12')
        else: r, e = udb_resid(f'{s.lower()}_L16_k0{k}_rt', 16, k)
        xs.append(jj + (i - 1) * 0.18); ys.append(r); es.append(e)
    axs[1].errorbar(xs, ys, yerr=es, fmt='o', ms=2.8, color=C[s], elinewidth=0.7)
axs[1].axhline(0, color=NS.ZERO_COLOR, lw=0.6); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(['0.24', '0.27', '0.30']); axs[1].set_xlabel(r'$\kappa$')
NS.tag(axs[1], '$L=16$')
h, _ = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
NS.fig_legend(fig, h, ncol=3)
for a_, l in zip(axs, 'ab'): NS.panel_letter(a_, l)
save(fig, 'fig_udb_bar')
PU.note('L16 kappa=0.30: quad_err empty in ti_logz.csv -> se_comb only')
PU.dump()
# ---------------------------------------------------------------- F4 refinement of CMCD / DBS at three training steps
PF = U.Prov('fig_refine_K')
SH = {'cmcd': {80: '#f3c97a', 160: '#E69F00', 320: '#8a5f00'}, 'dbs': {80: '#a9d8f4', 160: '#56B4E9', 320: '#1f6f9e'}}
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.35), sharex=True)
for j, k in enumerate(('0.27', '0.30')):
    for alg, ls, mk in (('cmcd', '-', 'o'), ('dbs', '--', 's')):
        for Tt in (80, 160, 320):
            d = json.load(open(REV / f'dbs_eta_20261115/diag_{alg}_k{k}_T{Tt}.json'))
            PF.add(f'k={k}', str(REV / f'dbs_eta_20261115/diag_{alg}_k{k}_T{Tt}.json'), 'runs[].mult, runs[].var_logw (unchanged)')
            n = [r['mult'] for r in d['runs']]; v = [r['var_logw'] for r in d['runs']]
            col = SH[alg][Tt]
            axs[j].plot(n, v, ls=ls, marker=mk, ms=2.8, color=col, lw=0.9, mfc=col if alg == 'cmcd' else 'white')
    axs[j].set_xscale('log', base=2); axs[j].set_yscale('log'); axs[j].set_xticks([1, 2, 4, 8, 16]); axs[j].set_xticklabels(['1', '2', '4', '8', '16']); axs[j].minorticks_off()
    axs[j].set_xlabel('$n$'); NS.tag(axs[j], f'$\\kappa={k}$', loc='lower left' if j == 1 else 'upper left')
for ax in axs:
    ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 3, 5]); ax.set_yticklabels(['0.2', '0.3', '0.5', '1', '2', '3', '5']); ax.set_ylim(0.17, 5.5)
axs[0].set_ylabel(r'Var$(\log w)$')
h = []
for alg, lab, ls, mk in (('cmcd', 'CMCD', '-', 'o'), ('dbs', 'DBS', '--', 's')):
    for Tt in (80, 160, 320):
        col = SH[alg][Tt]
        h.append(plt.Line2D([], [], color=col, ls=ls, marker=mk, ms=3, mfc=col if alg == 'cmcd' else 'white', label=f'{lab}, $T={Tt}$'))
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
NS.fig_legend(fig, h, ncol=6, columnspacing=1.0)
for a_, l in zip(axs, 'ab'): NS.panel_letter(a_, l)
save(fig, 'fig_refine_K')
PF.dump()
