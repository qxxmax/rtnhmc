"""2026-11-17 u3 of figs_20261117u2/make_figs_u2.py, fig_rect_transfer only (udb_bar, refine_K unchanged, not redrawn).
Drawing changes (data unchanged): TI band extended at constant width to the x limits so that the offset BAR / forward
points at N_x = 64, 512 lie inside it; band drawn above a thinner zero line (lw 0.4) so that the narrow kappa = 0.20 band
is not hidden by the line.
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
REV = pathlib.Path(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'); OUT = REV / 'figs_20261117u3'
sys.path.insert(0, str(REV / 'ti_wolff_hmc')); import ti_logz as T
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/revision/longchain'); import logz_estimators as E
sys.path.insert(0, str(OUT)); import u3_common as U; NS = U.NS
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
BANDS = {}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.35), sharex=True)
B = NS.BAR_STYLE
for j, k in enumerate((20, 30)):
    ax = axs[j]; tiv = {n: ti(load(str(REV / f'ti_rect2_20261116/*R{n}x8_k*_wolff2.json')), k * 10) for n in NX}
    for n in NX:
        PR.add(f'k0.{k} TI', str(REV / 'ti_rect2_20261116'), f'TI of R{n}x8: *_wolff2.json files (101 per N_x), Simpson + 3/8, step-halving /15')
    band = np.array([tiv[n][1] for n in NX]) * 1e4
    bx = [52] + NX + [630]; bb = np.r_[band[0], band, band[-1]]                         # u3: band to the x limits
    ax.axhline(0, color=NS.ZERO_COLOR, lw=0.4, zorder=0.5); ax.fill_between(bx, -bb, bb, color=NS.BAND, lw=0, zorder=1)
    BANDS[k] = band.tolist()
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
                    elinewidth=0.7, capsize=0, label=lab, zorder=3)
    ax.set_xscale('log', base=2); ax.set_xticks(NX); ax.set_xticklabels([str(n) for n in NX]); ax.minorticks_off()
    ax.set_xlabel('$N_x$'); NS.tag(ax, f'$\\kappa={k / 100:.2f}$')
    ax.set_ylim(-7, 7); ax.set_xlim(52, 630)
axs[0].set_ylabel(r'$(\log\hat Z-\log Z_{\rm TI})/V\ [10^{-4}]$')
h, _ = axs[1].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
NS.fig_legend(fig, h, ncol=3)
for a_, l in zip(axs, 'ab'): NS.panel_letter(a_, l)
save(fig, 'fig_rect_transfer')
PR.note('u3: TI band half-widths [1e-4 per site] ' + json.dumps(BANDS))
PR.dump()
print(BANDS)
