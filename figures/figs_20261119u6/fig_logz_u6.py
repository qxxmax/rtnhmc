#!/usr/bin/env python3
"""2026-11-19 u6 of figs_20261117u3/phi4_figs_u3.py fig_logz (imported; data, estimators, TI/AIS references unchanged).
Drawing only:
  - y = (log Z - log Z_TI) / V (per site), own range per panel; no inset.
  - main figure (fig_logz): SNIS, X + SNIS (X = path-IMH, RT, pRT) and AIS; the y range is set by the bridge estimates,
    AIS and the TI band; a SNIS point outside it is drawn as an arrow at the panel edge with its value written next to it.
  - appendix figure (fig_logz_rev): the one-way reverse estimates X-rev with SNIS, same axes convention.
Usage: python3 fig_logz_u6.py"""
import os
import math
import pathlib
import sys

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261117u3')
import phi4_figs_u3 as P  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
NS = P.NS
P.CUR[0] = P.prov('fig_logz_u6')
P.out = lambda name: str(OUT / f'{name}_20261119u6')


def ti_series(D, L):
    import ti_logz as T
    C = T.combine(T.series(D, L, 'wolff'), T.series(D, L, 'hmc'))
    ks = sorted(set(range(200, 301, 10)) | {P.kmil(k) for k in P.KAPPAS[L]})
    ti, qs = {}, {}
    for k in ks:
        a = T.ti(C, L, k, 1)
        if a is None:
            continue
        ti[k] = a
        if k % 10 == 0:
            b = T.ti(C, L, k, 2)
            qs[k] = abs(a[0] - b[0]) / 15 if b else 0.0
    for k in ti:
        q = qs.get(k, max(qs.get(k - k % 10, 0.0), qs.get(k - k % 10 + 10, 0.0)))
        ti[k] = (ti[k][0], math.hypot(ti[k][1], q))
    return ti


def make(name, keys):
    import ti_logz as T
    D = T.load()
    A = T.ais_refs()
    lab = [t for t in P.LZ_LAB if t[0] in keys]
    mk_open = {'P': 'P', 's': 's', 'o': 'o', '^': '^', 'S': 'D', 'O': 'h', 'A': 'v'}
    KG = sorted(P.kval(k) for k in P.KAPPAS[P.LS[0]])
    n = len(lab) + 1

    def shift(kap, i):
        j = int(np.argmin([abs(kap - g) for g in KG]))
        sp = min(abs(KG[j] - KG[jj]) for jj in (j - 1, j + 1) if 0 <= jj < len(KG))
        return (i - (n - 1) / 2) * (0.7 / (n - 1)) * sp

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    outliers = []
    for j, L in enumerate(P.LS):
        a_ = axs.flat[j]
        V = L * L / 1e4          # plotted in units of 1e-4 per site
        ti = ti_series(D, L)
        kk = sorted(ti)
        x, s = np.array(kk) / 1000, np.array([ti[k][1] for k in kk]) / V
        a_.fill_between(x, -s, s, color=NS.BAND, lw=0, zorder=0)
        a_.axhline(0, color=NS.ZERO_COLOR, lw=0.6, zorder=0)
        series, ext = [], [float(np.max(s))]
        px = [k for k in ti if (L, round(k / 1000, 3)) in A]
        ra = [(A[(L, round(k / 1000, 3))][0] - ti[k][0]) / V for k in px]
        ea = [A[(L, round(k / 1000, 3))][1] / V for k in px]
        ext += [abs(r) + e for r, e in zip(ra, ea)]
        for i, (key, lb, col, mk) in enumerate(lab):
            xs, ys, es = [], [], []
            for k in P.KAPPAS[L]:
                v = P.chain_logz(f'E2_L{L}_k0{k}').get(key)
                if v is not None and P.kmil(k) in ti:
                    xs.append(P.kval(k) + shift(P.kval(k), i))
                    ys.append((v[0] - ti[P.kmil(k)][0]) / V)
                    es.append(v[1] / V)
            series.append((key, col, mk, xs, ys, es))
            if key != 'snis':
                ext += [abs(y) + e for y, e in zip(ys, es)]
        lim = 1.25 * max(ext)
        snis_all = [abs(y) + e for key, _, _, _, ys, es in series if key == 'snis' for y, e in zip(ys, es)]
        if snis_all and max(snis_all) <= 2.0 * lim:      # SNIS inside a modest range: include it
            lim = max(lim, 1.1 * max(snis_all))
        a_.errorbar([k / 1000 + shift(k / 1000, n - 1) for k in px], ra, yerr=ea, fmt='x', ms=3, color='0.4',
                    elinewidth=0.6, zorder=3)
        for key, col, mk, xs, ys, es in series:
            m_ = mk if mk.islower() or mk == 'P' else mk_open[mk]
            filled = key == 'snis' or key.endswith('rev')
            inn = [q for q in range(len(xs)) if abs(ys[q]) <= lim]
            a_.errorbar([xs[q] for q in inn], [ys[q] for q in inn], yerr=[es[q] for q in inn], fmt=m_, ms=2.8,
                        mfc=col if filled else 'white', mec=col, ecolor=col, elinewidth=0.6, zorder=3)
            for q in range(len(xs)):
                if abs(ys[q]) > lim:
                    sg = 1 if ys[q] > 0 else -1
                    a_.annotate('', xy=(xs[q], sg * 0.98 * lim), xytext=(xs[q], sg * 0.80 * lim),
                                arrowprops=dict(arrowstyle='->', color=col, lw=0.8), zorder=4)
                    a_.text(xs[q] + 0.0025, sg * 0.89 * lim, f'{ys[q]:.1f}', ha='left', va='center', fontsize=6.5,
                            color=col)
                    outliers.append(dict(L=L, key=key, value_1e4_per_site=ys[q]))
        a_.set_ylim(-lim, lim)
        P.kaxis(a_, label=j >= 2)
        if j % 2 == 0:
            a_.set_ylabel(r'$10^4\,(\log Z - \log Z_{\rm TI})/V$')
        NS.tag(a_, f'$L={L}$', loc='upper left')
        NS.panel_letter(a_, 'abcd'[j])
    h = [plt.Line2D([], [], ls='', marker=mk if (key == 'snis' or key.endswith('rev')) else mk_open[mk], ms=4, mec=col,
                    mfc=col if (key == 'snis' or key.endswith('rev')) else 'white', label=lb) for key, lb, col, mk in lab]
    h += [plt.Line2D([], [], ls='', marker='x', color='0.4', label='AIS'),
          matplotlib.patches.Patch(color=NS.BAND, label='TI (Wolff / HMC)')]
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    NS.fig_legend(fig, h, ncol=len(h) if len(h) <= 6 else 4)
    NS.save(fig, P.out(name))
    print(name, 'outliers drawn as arrows:', outliers)


make('fig_logz', ('snis', 'IMH-bar', 'RT-bar', 'pRT-bar'))
make('fig_logz_rev', ('snis', 'IMH-rev', 'RT-rev', 'pRT-rev'))
P.CUR[0].dump()
