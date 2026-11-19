"""2026-11-18 u5 of figs_20261117u4/make_bar_figs_u4.py, fig_bar_prefix only.
Single change: error bars are jackknife errors over 20 joint blocks of the forward sample and the reverse records (the block
count of the appendix E table, logz_estimators.estimates default B=20; forward sample 65520 paths), instead of 10 blocks.
Values are read from unify_20261118/item12_bar.json (item2: bar20 / err20, made by unify_20261118/item12_bar.py with the
same chains, forward pool, TI reference and estimator as u4); nothing is recomputed here.  Central values are the B=20
estimates (forward pool truncated to 65520, shifts below 0.011 x 1e-4 against the u4 B=10 values).
Drawing identical to u4 (stagger x 2^(0.05 (i - 1)), gid 'u4xmul:<f>')."""
import os
import sys, json, matplotlib
matplotlib.use('pdf'); import matplotlib.pyplot as plt
REV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'
OUT = REV + '/figs_20261118u5'
sys.path.insert(0, OUT); import u5_common as U; NS = U.NS
SRC = REV + '/unify_20261118/item12_bar.json'
D = json.load(open(SRC))['item2']
PP = U.Prov('fig_bar_prefix')
NS.apply()
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.35), sharey=True)
cols = NS.NXCOL
for j, k in enumerate(('20', '30')):
    for i_, n in enumerate((128, 256, 512)):
        f_ = 2 ** (0.05 * (i_ - 1))                                                   # u4: horizontal stagger
        rows = D[f'{n}x8_k0{k}']
        assert [q['prefix'] for q in rows] == [2500, 5000, 10000, 20000] and all(q['n_fwd20'] == 65520 for q in rows)
        xs = [q['prefix'] for q in rows]; ys = [q['bar20'] for q in rows]; es = [q['err20'] for q in rows]
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
PP.add('all', SRC, 'item2 bar20 / err20 (20 joint blocks, 65520 forward paths)')
PP.add('all', REV + '/unify_20261118/item12_bar.py', 'script that produced item2')
PP.note('u5: 20 blocks (was 10); values from item12_bar.json')
PP.dump()
