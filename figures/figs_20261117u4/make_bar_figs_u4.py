"""2026-11-17 u4 of figs_20261117u2/make_bar_figs_u2.py (data, estimators, binning unchanged; drawing only):
  - workdist: condition tags drawn on a white box (in (c) the x = 0 line ran through the tag).
  - prefix: the three N_x series are staggered horizontally by x 2^(0.05 (i - 1)), i = 0, 1, 2 for 128, 256, 512
    (gid 'u4xmul:<f>', undone in compare_u4.py; points at 10k and 20k overlapped); x label "prefix length per chain".
u2 docstring: 2026-11-17 u2 copy of figs_20261117u/make_bar_figs_20261117u.py: data, estimators and numbers unchanged; only output
names (*_20261117u2.pdf) and provenance recording (prov/) added.

u docstring: restyled copy of figs_20261116/make_bar_figs_20261116b.py (unified style, nhmc_style.py): only styling and
outputs (figs_20261117u/fig_bar_{workdist,prefix}_20261117u.pdf) changed; data, estimators and numbers unchanged. Titles
became condition tags inside the panels; panel letters added; reverse colour #E69F00 (BAR-figure convention).

Appendix E (bidirectional free energy) figures and table with the 2026-11-15/16 data (read only):
forward work W_F of 65536 fresh paths (raw_snis/R_{Nx}x8_k0XX.npz) and reverse records W_R of the truncated round-trip chains
(logz_rect2_20261115/*.npz, 4 seeds x 4 chains x 2e4; 64x8 from logz_chains/R_64x8_k0XX_rtp12.npz), reference = recomputed TI
(ti_rect2_20261116). Work convention of the code: W = log Q - log P~, so E_Q[e^{-W}] = Z; both W_F and W_R are W(Gamma).
Outputs: fig_bar_workdist_20261116b.{pdf,png}, fig_bar_prefix_20261116b.{pdf,png}, bar_table_20261116.json."""
import os
import sys, json, glob, math, numpy as np, matplotlib
matplotlib.use('pdf'); import matplotlib.pyplot as plt
REV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'
sys.path.insert(0, REV + '/ti_wolff_hmc'); import ti_logz as T
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/revision/longchain'); import logz_estimators as E
OUT = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261117u4'
sys.path.insert(0, OUT); import u4_common as U; NS = U.NS
PW, PP = U.Prov('fig_bar_workdist'), U.Prov('fig_bar_prefix')
NS.apply()
def load_ti(n):
    s = {}
    for f in glob.glob(f'{REV}/ti_rect2_20261116/*R{n}x8_k*_wolff2.json'):
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
    a, e = integ(1); b, _ = integ(2)
    return T.LOGZ1 + 2 * a, math.hypot(2 * e, 2 * abs(a - b) / 15)
def chains(n, k, pr=None):
    if pr is not None:
        for f in ([f'{REV}/logz_chains/R_64x8_k0{k}_rtp12.npz'] if n == 64 else sorted(glob.glob(f'{REV}/logz_rect2_20261115/R_{n}x8_k0{k}_rtp12_s*.npz'))):
            pr.add(f'{n}x8 k0.{k}', f, 'reverse records of the rtp12 chains (first 10% dropped in the histograms)')
        pr.add(f'{n}x8 k0.{k}', f'{REV}/raw_snis/R_{n}x8_k0{k}.npz', 'forward work, 65536 paths')
    if n == 64:
        z = np.load(f'{REV}/logz_chains/R_64x8_k0{k}_rtp12.npz'); return z['WR'], z['absM'], z['acc']
    zs = [np.load(f) for f in sorted(glob.glob(f'{REV}/logz_rect2_20261115/R_{n}x8_k0{k}_rtp12_s*.npz'))]
    return (np.concatenate([z['WR'] for z in zs], 1), np.concatenate([z['absM'] for z in zs], 1), np.concatenate([z['acc'] for z in zs], 1))
NX = (64, 128, 256, 512); table = {}
for k in ('20', '30'):
    for n in NX:
        V = n * 8; F = -ti(load_ti(n), int(k) * 10)[0] * V; Fe = ti(load_ti(n), int(k) * 10)[1] * V
        WF = np.load(f'{REV}/raw_snis/R_{n}x8_k0{k}.npz')['W']; WR, M, A = chains(n, k)
        r = E.estimates(dict(WR=WR, WF=WF, absM=M, acc=A))
        res = {key: ((r[key][0] + F) / V * 1e4, math.hypot(r[key][1], Fe) / V * 1e4) for key in ('snis', 'rev', 'bar')}
        table[f'{n}x8_k0{k}'] = dict(resid_1e4=res, ti_err_1e4=Fe / V * 1e4, acc=r['acc'], tau_M=r['tau_M'] / 2, n_chain=r['n_chain'], n_fwd=r['n_fwd'],
                                     varWF_perV=float(np.var(WF) / V), varWR_perV=float(np.var(WR[WR.shape[0] // 10:]) / V))

for key, v in table.items(): print(key, {kk: (round(vv[0], 2), round(vv[1], 2)) for kk, vv in v['resid_1e4'].items()}, 'acc %.3f tauM %.1f' % (v['acc'], v['tau_M']))
# ---- work distributions (per site, relative to F_TI/V)
fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.2), sharey=False)
for ax, (n, k) in zip(axs, ((64, '20'), (512, '20'), (512, '30'))):
    V = n * 8; F = -ti(load_ti(n), int(k) * 10)[0] * V
    WF = np.load(f'{REV}/raw_snis/R_{n}x8_k0{k}.npz')['W']; WR = chains(n, k, PW)[0]; WR = WR[WR.shape[0] // 10:].ravel()
    xf, xr = (WF - F) / V * 1e3, (WR - F) / V * 1e3
    lo, hi = np.percentile(np.r_[xf, xr], [0.5, 99.5]); bins = np.linspace(lo, hi, 60)
    ax.hist(xf, bins, density=True, histtype='step', color=NS.BAR_STYLE['forward']['color'], lw=1.1, label='forward')
    ax.hist(xr, bins, density=True, histtype='step', color=NS.BAR_STYLE['reverse']['color'], lw=1.1, label='reverse (chain)')
    ax.axvline(0, color=NS.ZERO_COLOR, lw=0.6); ax.set_xlabel(r'$(W-F_{\rm TI})/V\ [10^{-3}]$'); ax.set_yticks([])
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    NS.tag(ax, f'${n}\\times8$, $\\kappa=0.{k}$', bbox=dict(fc='white', ec='none', pad=0.6), zorder=5)   # u4: white box
h, _ = axs[0].get_legend_handles_labels()
h = [plt.Line2D([], [], color=NS.BAR_STYLE[k_]['color'], lw=1.1, label=l_) for k_, l_ in (('forward', 'forward'), ('reverse', 'reverse (chain)'))]
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.5)
NS.fig_legend(fig, h, ncol=2)
for a_, l_ in zip(axs, 'abc'): NS.panel_letter(a_, l_, x=0.12)
NS.save(fig, U.pdf('fig_bar_workdist'))
PW.add('all', f'{REV}/ti_rect2_20261116', 'TI F_TI of each N_x (wolff2 json)')
PW.note('u4: tags on white box')
PW.dump()
# ---- finite-sample convergence: estimates on chain prefixes (all chains), forward pool fixed
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.35), sharey=True)
cols = NS.NXCOL
for j, k in enumerate(('20', '30')):
    for i_, n in enumerate((128, 256, 512)):
        f_ = 2 ** (0.05 * (i_ - 1))                                                   # u4: horizontal stagger
        V = n * 8; F = -ti(load_ti(n), int(k) * 10)[0] * V
        WF = np.load(f'{REV}/raw_snis/R_{n}x8_k0{k}.npz')['W']; WR, M, A = chains(n, k, PP)
        xs, ys, es = [], [], []
        for m in (2500, 5000, 10000, 20000):
            r = E.estimates(dict(WR=WR[:m], WF=WF, absM=M[:m], acc=A[:m]), B=10)
            xs.append(m); ys.append((r['bar'][0] + F) / V * 1e4); es.append(r['bar'][1] / V * 1e4)
        eb = axs[j].errorbar([x_ * f_ for x_ in xs], ys, yerr=es, fmt='o-', ms=3, lw=0.9, color=cols[n], label=f'${n}\\times8$')
        eb.lines[0].set_gid(f'u4xmul:{f_!r}')
        for c_ in eb.lines[2]:
            c_.set_gid(f'u4xmul:{f_!r}')
    axs[j].axhline(0, color=NS.ZERO_COLOR, lw=0.6); axs[j].set_xscale('log'); axs[j].set_xticks([2500,5000,10000,20000]); axs[j].set_xticklabels(['2.5k','5k','10k','20k']); axs[j].minorticks_off(); axs[j].set_xlabel('prefix length per chain')
    NS.tag(axs[j], f'$\\kappa=0.{k}$', loc='upper right')
axs[0].set_ylabel(r'BAR $(\log\hat Z-\log Z_{\rm TI})/V\ [10^{-4}]$')
h, _ = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
NS.fig_legend(fig, h, ncol=3)
for a_, l_ in zip(axs, 'ab'): NS.panel_letter(a_, l_)
NS.save(fig, U.pdf('fig_bar_prefix'))
PP.add('all', f'{REV}/ti_rect2_20261116', 'TI F_TI of each N_x (wolff2 json)')
PP.note('u4: series staggered x 2^(0.05(i-1)); x label prefix length per chain')
PP.dump(extra=dict(table={k: dict(resid_1e4=v['resid_1e4'], acc=v['acc'], tau_M=v['tau_M'], n_fwd=v['n_fwd']) for k, v in table.items()}))
