"""2026-11-17 u3 of figs_20261117u2/fig_phi4_main_u2.py: data unchanged; (c) points coloured by L (method by shape,
as in fig_observables u3), legend method entries grey shapes; z clip at +-4.3 counted (0 points).
u2 docstring: 2026-11-17 u2 of figs_20261117u/fig_phi4_main_20261117u.py: data and selection unchanged (data functions from
phi4_figs_u2, whose selection equals u); only output name (fig_phi4_main_20261117u2.pdf) and provenance recording added."""
import os
import json, math, sys
import numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261117u3')
import phi4_figs_u3 as P
NS = P.NS
P.CUR[0] = P.prov('fig_phi4_main')

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
for L in P.LS:
    xs, ess, raw, sn = [], [], [], []
    for k in P.KAPPAS[L]:
        f = P.RSN / f'{P.cell_name(L, k)}.json'
        if not f.exists(): continue
        P.CUR[0].add('a, b', str(f), 'raw_snis/E2_L{L}_k0{k}.json (65536 forward paths)')
        d = json.load(open(f)); ref = P.wh_comb(L, P.kmil(k), 'abs_m')
        xs.append(P.kval(k)); ess.append(d['kish_ess_frac'])
        raw.append((d['raw']['abs_m'][0] - ref[0]) / ref[0]); sn.append((d['snis']['abs_m'][0] - ref[0]) / ref[0])
    ax[0].plot(xs, ess, 'o-', ms=2.8, lw=0.8, color=P.LCOL[L])
    ax[1].plot(xs, raw, 'o:', ms=2.8, lw=0.6, mfc='white', color=P.LCOL[L])
    ax[1].plot(xs, sn, 's-', ms=2.8, lw=0.6, color=P.LCOL[L])
stats = {m: {'abs_m': [], 'chi': [], 'binder': []} for m in P.MRK}
OFF = {'IMH': -0.0012, 'RT': 0.0, 'pRT': 0.0012}
for L in P.LS:
    for k in P.KAPPAS[L]:
        mf = P.method_files(P.cell_name(L, k))
        tot = [t for t in (P.post_burn_total(jf) for jf in mf.values()) if t]
        if not tot: continue
        N = min(tot)
        for m, jf in mf.items():
            r = P.obs_from_traces(jf, L, nper=N // 4)
            if r is None: continue
            for j, key in enumerate(('abs_m', 'chi', 'binder')):
                ref = P.wh_comb(L, P.kmil(k), key)
                if ref:
                    z = (r[0][j] - ref[0]) / math.hypot(r[1][j], ref[1]); stats[m][key].append(z)
                    if key == 'abs_m':
                        ax[2].plot(P.kval(k) + OFF[m], P.zclip(z), P.MRK[m], ms=2.6, mfc=P.LCOL[L], mec=P.LCOL[L])  # u3: colour = L
ax[0].set_yscale('log'); ax[0].set_ylabel(r'ESS$_w/N$')
ax[1].axhline(0, color=NS.ZERO_COLOR, lw=0.6); ax[1].set_ylabel(r'$\delta\langle|m|\rangle$')
P.zpanel(ax[2]); ax[2].set_ylabel(r'$z(\langle|m|\rangle)$')
for j, a in enumerate(ax):
    P.kaxis(a); NS.panel_letter(a, 'abc'[j])
h = [NS.lhandle(L) for L in P.LS]
h += [plt.Line2D([], [], ls=':', marker='o', mfc='white', color='0.3', ms=4, label='raw'), plt.Line2D([], [], ls='-', marker='s', color='0.3', ms=4, label='SNIS')]
h += [plt.Line2D([], [], ls='', marker=P.MRK[m], mfc='0.4', mec='0.4', ms=4, label=P.LAB[m]) for m in P.MRK]  # u3
fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.5)
NS.fig_legend(fig, h, ncol=9, columnspacing=1.0)
NS.save(fig, P.out('fig_phi4_main'))
S = {m: {k: dict(n=len(v), frac_within2=float(np.mean(np.abs(v) < 2)), max_abs=float(np.max(np.abs(v)))) for k, v in d.items() if v} for m, d in stats.items()}
P.CUR[0].note(f'u3: (c) coloured by L; z clipped at +-4.3: {P.NCLIP[0]} points')
P.CUR[0].dump(extra=S)
print(json.dumps(S))
