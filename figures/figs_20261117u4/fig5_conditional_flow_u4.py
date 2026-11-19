"""2026-11-17 u4 of figs_20261117u2/fig5_conditional_flow_u2.py (data unchanged; symbols only, NOTATION_PLAN R8 / R30):
legend "accepted flow F^RT_{+-}" -> "accepted flow Omega^RT_{+-}", x label of (b) (s_f, s_r) -> (zeta_F, zeta_R).  The u2 block
that parsed the old manuscript PDF is not repeated (result in figs_20261117u2/fig5_compare_u2.json).
u2 docstring: 2026-11-17 u2: fig5_nhmc_conditional_flow redrawn from a script (the manuscript PDF of 2026-11-04 had none).
Data: local_20261103/v6_data_phi4/phi4_transport.csv (generator scripts/build_transport_udb.py, no sampling).
  (a) law 'fixed Gaussian Z2 (T=20, n_lf=5, eps=0.135) FROZEN', K = 512, p_flip = 0, 0.05, 0.10, 0.15:
      Omega +- Omega_se (ideal flow), F_RT +- F_se (accepted flow); source field
      'results_sync/pflip_kconv.json + pflip_kconv_errors.json (FROZEN)'.
  (b) laws 'controlled Gaussian et1 arm s11 / sf1.02 / sr1.02 / s1.02', K = 512, p_flip = 0; source field
      'improve/exp2/et1/results/et1_tables.json'; SE = chain-clustered jackknife.
Same content as the old PDF (two panels, ideal and accepted flow, SE bars); style from figs_20261117u/nhmc_style.py.
The old PDF is parsed with PyMuPDF (marker centres and error-bar ends, mapped to data through the tick labels) and compared
point by point -> fig5_compare_u2.json."""
import os
import csv
import json
import sys

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261117u4')
import u4_common as U  # noqa: E402
NS = U.NS
NS.apply()
CSV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/local_20261103/v6_data_phi4/phi4_transport.csv'
OLD = 'MANUSCRIPT_FIGURE_NOT_INCLUDED'
pr = U.Prov('fig5_nhmc_conditional_flow')
rows = list(csv.DictReader(open(CSV)))
pr.add('a, b', CSV, 'K = 512 rows; (a) FROZEN p_flip scan, (b) et1 arms s11/sf1.02/sr1.02/s1.02')
A = sorted([r for r in rows if 'FROZEN' in r['law'] and r['K'] == '512'], key=lambda r: float(r['p_flip']))
ARMS = (('s11', '(1, 1)'), ('sf1.02', '(1.02, 1)'), ('sr1.02', '(1, 1.02)'), ('s1.02', '(1.02, 1.02)'))
B = [next(r for r in rows if r['law'] == f'controlled Gaussian et1 arm {a}' and r['K'] == '512') for a, _ in ARMS]
f = lambda r, k: float(r[k])
CO, CF = NS.COL['NHMC'], NS.COL['CMCD']          # ideal flow blue, accepted flow orange (as the old figure)

fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.4))
x = [f(r, 'p_flip') for r in A]
ax[0].errorbar(x, [f(r, 'Omega') for r in A], yerr=[f(r, 'Omega_se') for r in A], fmt='o-', color=CO, ms=3.5, capsize=1.5,
               elinewidth=0.7, label=r'ideal flow $\Omega_{+-}$')
ax[0].errorbar(x, [f(r, 'F_RT') for r in A], yerr=[f(r, 'F_se') for r in A], fmt='s--', color=CF, ms=3.5, capsize=1.5,
               elinewidth=0.7, label=r'accepted flow $\Omega^{\rm RT}_{+-}$')
ax[0].set_xticks([0, 0.05, 0.10, 0.15])
ax[0].set_ylim(0, 0.16)
ax[0].set_xlabel(r'$p_{\rm flip}$')
ax[0].set_ylabel('inter-sector flow')
NS.tag(ax[0], r'$L=8$, $\kappa=0.27$', loc='lower right')
xb = np.arange(4)
ax[1].errorbar(xb - 0.07, [f(r, 'Omega') for r in B], yerr=[f(r, 'Omega_se') for r in B], fmt='o', color=CO, ms=3.5, capsize=1.5,
               elinewidth=0.7)
ax[1].errorbar(xb + 0.07, [f(r, 'F_RT') for r in B], yerr=[f(r, 'F_se') for r in B], fmt='s', color=CF, ms=3.5, capsize=1.5,
               elinewidth=0.7)
ax[1].set_xticks(xb)
ax[1].set_xticklabels([l for _, l in ARMS])
ax[1].set_xlim(-0.5, 3.5)
ax[1].set_ylim(0, 0.05)
ax[1].set_xlabel(r'$(\zeta_{\rm F},\,\zeta_{\rm R})$')
NS.tag(ax[1], r'$p_{\rm flip}=0$', loc='lower right')
h, _ = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
NS.fig_legend(fig, h, ncol=2)
NS.panel_letters(ax)
NS.save(fig, U.pdf('fig5_nhmc_conditional_flow'))

pr.note('u4: labels Omega^RT_{+-} (R8) and (zeta_F, zeta_R) (R30); comparison with the 2026-11-04 PDF is in figs_20261117u2/fig5_compare_u2.json')
pr.dump()
