"""2026-11-17 u2 copy of figs_20261117u/make_fig14_acf_20261117u.py: data and estimator unchanged; output name
(fig14_acf_L8_20261117u2.pdf) and provenance (prov/) added; chain lengths printed for the caption (k0.27 pRT and
k0.30 path-IMH have 2e4 transitions per chain, the others 1e4).

u docstring: restyled copy of figs_20261116/make_fig14_acf_20261116.py (unified style, nhmc_style.py): only styling and output
(figs_20261117u/fig14_acf_L8_20261117u.pdf) changed; data and estimators unchanged; the summary json is only printed."""
import os
import sys, json, numpy as np, matplotlib
matplotlib.use('pdf'); import matplotlib.pyplot as plt
R = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/longchain_min'
OUT = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261117u2'
sys.path.insert(0, OUT); import u2_common as U; NS = U.NS
PR = U.Prov('fig14_acf_L8')
NS.apply()
C = {'imhG': ('path-IMH', NS.COL['path-IMH'], '-'), 'rt': ('RT', NS.COL['RT'], '-'), 'rtp12': ('pRT ($s=12$)', NS.COL['pRT'], '--')}
def acf(x, lag):
    x = x - x.mean(0)
    v = (x * x).mean()
    return np.array([1.0] + [float((x[:-j] * x[j:]).mean() / v) for j in range(1, lag + 1)])
def tau_g2(rho):
    for m in range(1, len(rho)):
        t = 0.5 + rho[1:m + 1].sum()
        if m >= 5 * 2 * t: return t
    return 0.5 + rho[1:].sum()
out = {}
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.3), sharey=True)
for j, k in enumerate(('027', '030')):
    for m, (lab, col, ls) in C.items():
        z = np.load(f'{R}/E2_L8_k{k}_{m}_traces.npz'); a = z['absM']; b = a.shape[0] // 10
        PR.add('ab'[j], f'{R}/E2_L8_k{k}_{m}.json', 'chain json (n_transitions, criteria_met)')
        PR.add('ab'[j], f'{R}/E2_L8_k{k}_{m}_traces.npz', 'traces; first 10% dropped; ACF of the 4 chains pooled after per-chain centering',
               f'{a.shape[0]} transitions x {a.shape[1]} chains')
        rho = acf(a[b:], 400); t = tau_g2(rho); acc = float(z['acc'][b:].mean())
        out[f'k{k}_{m}'] = dict(tau=round(t, 2), acc=round(acc, 3))
        axs[j].plot(np.arange(41), rho[:41], ls=ls, color=col, lw=1.2, label=lab)
    axs[j].axhline(0, color=NS.ZERO_COLOR, lw=0.5); axs[j].set_xlabel('$j$'); axs[j].set_xlim(0, 40)
    NS.tag(axs[j], f'$L=8$, $\\kappa=0.{k[1:]}$', loc='upper right')
axs[0].set_ylabel(r'$\rho_{|m|}(j)$')
h, _ = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
NS.fig_legend(fig, h, ncol=3)
for a_, l in zip(axs, 'ab'): NS.panel_letter(a_, l)
NS.save(fig, U.pdf('fig14_acf_L8'))
PR.dump(extra=out)
print(out)
