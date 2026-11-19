#!/usr/bin/env python3
"""2026-11-17 u4 of figs_20261117u3/fig_scaling_accept_u3.py (data unchanged, drawing only):
  - 1000-step pilots: edge in the kappa colour (u3: grey), open, same shape; a small grey centre dot (gid 'u4deco', extra
    artist) marks them as pilots; zero-accept 95% upper-limit arrows of the pilots in the kappa colour; legend entries
    "1000-step pilot" (open circle + grey dot) and "95% upper limit" (grey arrow) stay.
  - x label erfc(sigma_s/2) (sigma_s^2 measured conditional work variance, wtail2 cond_var; NOTATION_PLAN R15).
u3 docstring: 2026-11-17 u3: fig_scaling_accept only, cut out of figs_20261117u2/fig_scaling_u2.py (data functions copied unchanged;
the jackknife / fit machinery of the u2 script is not needed for this figure and is not run here).
Change against u2 (drawing only, data unchanged): pilot points whose predicted acceptance lies below the main x range
(1e-4) were clipped by the axis in u2 (6 zero-accept pilots, drawn as 95% upper limits, and 2 pilots with one accept);
u3 extends the x axis down to 1e-6, which brings 4 of these 8 into the plot (2 zero-accept upper limits, 2 one-accept
points), so that the legend entry "95% upper limit" has a counterpart; the remaining 4 zero-accept pilots (predicted
acceptance 9e-18 to 1e-8) are listed in prov and in the caption.  Figure 0.55 in wider to keep the main range readable."""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from scipy.optimize import brentq, least_squares  # noqa: E402
from scipy.special import erfc, erfcinv  # noqa: E402

sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261117u4')
import u4_common as U  # noqa: E402
NS = U.NS
NS.apply()

REV = U.REV
KAP = ['20', '21', '22', '23', '24', '25', '26', '265', '27', '275', '28', '29', '30']
kv = lambda k: int(k) / (100 if len(k) == 2 else 1000)
kcol = lambda k: NS.kcol(kv(k))
LS = (4, 6, 8, 10, 12, 16)
RTM, PRTM, IMHM, HMCM = NS.MRK['RT'], NS.MRK['pRT'], NS.MRK['path-IMH'], NS.MRK['HMC']
PILOT_EC = '0.55'
PILOT_DOT = '0.45'                                      # u4: centre dot of pilot symbols
P = {f: U.Prov(f) for f in ('fig_scaling_accept',)}


# ------------------------------------------------------------------ data
def native_file(L, k, m):
    if m == 'imh':
        c = [f'E2_L{L}_k0{k}_imhG', f'E2_L{L}_k0{k}_imh', f'buf_L{L}_k0{k}_imhG', f'buf_L{L}_k0{k}_imh']
    else:
        c = [f'E2_L{L}_k0{k}_{m}', f'buf_L{L}_k0{k}_{m}']
    for n in c:
        f = U.find(n + '.json')
        if f:
            return f
    return None


def transfer_file(L, k, m):
    return native_file(16, k, m) if L == 16 else U.find(f'T16_L{L}_k0{k}_{m}.json')


_C = {}


def get(f):
    if f is None:
        return None
    if f not in _C:
        _C[f] = U.chain_summary(f)
    return _C[f]


nat = lambda L, k, m: get(native_file(L, k, m))
tra = lambda L, k, m: get(transfer_file(L, k, m))
U_RULE_NATIVE = 'u2: E2_* then buf_*; path-IMH E2 imhG > E2 imh > buf imhG > buf imh'


def u_native_file(L, k, m):          # what u read (for the difference log)
    return U.find(f'E2_L{L}_k0{k}_{"imhG" if m == "imh" else m}.json')


CV, CVF = {}, {}
for f in sorted(glob.glob(U.H17 + '/*/revision_runs/wtail2_20261114/*.json')):
    L, k = re.match(r'L(\d+)_k0(\d+)', os.path.basename(f)).groups()
    CV[(int(L), k)] = {int(s): v['cond_var'] for s, v in json.load(open(f))['stages'].items()}
    CVF[(int(L), k)] = f

G = json.load(open(REV + '/figs/data/wh_tau_grid.json'))
REFJK = json.load(open(U.JK + '/fits_jk_buf.json'))['ref_tau']


def ref(alg, L, kmil=270):
    v = [x for x in G.values() if x['alg'] == alg and x['L'] == L and round(x['kappa'] * 1000) == kmil][0]
    e = REFJK.get(f'{alg}_L{L}')
    return v['tau_absm'] / 2, (e['err'] if e else None)


HMC = {L: ref('hmc', L) for L in (8, 12, 16)}
WOLFF = {L: ref('wolff', L) for L in (8, 12, 16)}

# ------------------------------------------------------------------ quantities (one function, re-run for jackknife replicas)
OVR = {}



def errbar(ax, x, y, yerr=None, xerr=None, **kw):
    kw.setdefault('elinewidth', 0.6)
    kw.setdefault('capsize', 0)
    return ax.errorbar(x, y, yerr=yerr, xerr=xerr, **kw)


def arrow_down(ax, x, y, color, ms=6):
    ax.plot([x], [y], marker=r'$\downarrow$', ms=ms, color=color, ls='')


def arrow_up(ax, x, y, color, ms=6):
    ax.plot([x], [y * 1.5], marker=r'$\uparrow$', ms=ms, color=color, ls='')



# ================================================================== fig_scaling_accept
pr = P['fig_scaling_accept']
fig, ax = plt.subplots(figsize=(3.3, 2.6))                       # u3: wider (x axis 1e-6..1)
XMIN = 1e-6
OFFR = []
dev_nat = []
for (L, k), cv in sorted(CV.items()):
    pr.add('native', CVF[(L, k)], 'wtail2_20261114 cond_var[s], s=0 (RT) and s=12 (pRT)')
    for m, s, mk in (('rt', 0, RTM), ('rtp12', 12, PRTM)):
        r = nat(L, k, m)
        if r and s in cv:
            p = erfc(np.sqrt(cv[s]) / 2)
            errbar(ax, p, r['acc'], yerr=r['acc_err'], fmt=mk, color=kcol(k), ms=3.5)
            dev_nat.append(abs(p - r['acc']))
            pr.add('native', r['path'], U_RULE_NATIVE,
                   'new in u2 (buf_*; u read E2_* only)' if u_native_file(L, k, m) is None else '')
for k in ('24', '27', '30'):
    cv = CV[(16, k)]
    for m, s, mk in (('rt', 0, RTM), ('rtp12', 12, PRTM)):
        for L in (24, 32, 48, 64, 96):
            r = tra(L, k, m)
            if not r:
                continue
            p = erfc(np.sqrt(cv[s] * L * L / 256) / 2)
            ec = kcol(k)                                       # u4: pilots keep the kappa colour
            if p < XMIN:
                OFFR.append(f'k0.{k} {m} L{L} pred={p:.3g} acc={r["acc"]:.3g} n_acc={r["nacc"]}')
            if r['acc'] <= 0:
                arrow_down(ax, p, r['ul95'], ec)
            else:
                errbar(ax, p, r['acc'], yerr=r['acc_err'], fmt=mk, color=ec, mfc='white', mec=ec, ms=3.5)
                if r['pilot']:
                    ax.plot([p], [r['acc']], 'o', ms=1.9, color=PILOT_DOT, mew=0, gid='u4deco', zorder=4)
            pr.add('transfer', r['path'], 'T16_L{L}_*; pilot = pilot_bigL n=1000 (grey edge); acc=0 -> 3/n_post arrow',
                   f'pilot, n_acc={r["nacc"]}' if r['pilot'] else f'n_acc={r["nacc"]}')
ax.plot([1e-4, 1], [1e-4, 1], color=NS.REF_COLOR, ls=NS.REF_LS, lw=0.7)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(XMIN, 1)
ax.set_ylim(1e-4, 1)
ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
ax.set_yticks([1e-4, 1e-2, 1])
ax.set_xlabel(r'predicted erfc$(\sigma_s/2)$')
ax.set_ylabel('measured acceptance')
h = [plt.Line2D([], [], color='0.2', ls='', marker=m, ms=3.5, mfc=f, label=l) for m, f, l in
     ((RTM, '0.2', 'RT native'), (PRTM, '0.2', 'pRT native'), (RTM, 'white', 'RT transfer'), (PRTM, 'white', 'pRT transfer'))]
h += [(plt.Line2D([], [], color='0.2', ls='', marker='o', mfc='white', ms=3.5),
       plt.Line2D([], [], color=PILOT_DOT, ls='', marker='o', ms=1.9, mew=0)),
      plt.Line2D([], [], color=PILOT_EC, ls='', marker=r'$\downarrow$', ms=6, label='95% upper limit')]
h += [NS.khandle(kv(k), lw=2.5) for k in ('24', '27', '30')]
LABS = [x.get_label() if not isinstance(x, tuple) else '1000-step pilot' for x in h]
ax.legend(h, LABS, loc='upper left', handlelength=1.2, labelspacing=0.2, borderaxespad=0.1, fontsize=6,
          handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})
fig.tight_layout()
NS.save(fig, U.pdf('fig_scaling_accept'))
pr.note(f'native points: {len(dev_nat)} (u: 12; + L12 kappa=0.27 RT/pRT from buf_L12_k027_*); max |pred-meas| = {max(dev_nat):.4f}')
pr.note('u3: x range 1e-6..1; not shown (predicted < 1e-6): ' + '; '.join(OFFR))
pr.note('u4: pilots kappa-coloured open with grey centre dot; arrows kappa-coloured')
pr.dump()
print(OFFR)

# ================================================================== fig_scaling_var
