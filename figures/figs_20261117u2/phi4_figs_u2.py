#!/usr/bin/env python3
"""2026-11-17 u2 of figs_20261117u/phi4_figs_20261117u.py (style: figs_20261117u/nhmc_style.py, imported unchanged).
Data paths and chain selection (method_files) are those of u (verified with inventory_u2.json: the 2026-11-17 syncs add
no new or longer chain json for these figures).  Changes against u (each logged in prov/<fig>.json):
  - mixing, mixing_sign, sscan, transfer, transfer_matrix: tau error bars = delete-one-chain jackknife of the chain mean
    (u2_common.cellinfo, method of jackknife_20261117/jk_cells.py); HMC / Wolff references in mixing: Madras-Sokal error
    with window W = 5 g (as jackknife_20261117/jk_fits.ref_tau); sscan acceptance: block-jackknife error bar.
  - transfer_obs: observables of chains with criteria_met = False are drawn with open markers.
  - transfer_matrix (d-f): raw_snis lookup also in iqcb_figdata_20261117/raw_snis (adds T12_L32_k024, missing in u).
  - acceptance, acceptance_all: accept_quality uses an independent fixed seed per (L, kappa, method) (crc32 of the tag);
    u used one shared default_rng(0) whose state depended on the call order, so the jump ratios were not reproducible
    between figures.
  - every file read is recorded (U.Prov).
PDF only. Usage: python3 phi4_figs_u2.py <names>   (names: observables rawsnis sscan transfer_obs transfer
transfer_matrix acceptance acceptance_all logz convergence mixing mixing_sign)

g = 1 + 2 sum rho (= n / ESS) per MC step is what the json stores; the figures show tau_int = g / 2."""
import zlib
import glob
import json
import math
import os
import pathlib
import sys

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

OUTU = pathlib.Path(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261117u2')
sys.path.insert(0, str(OUTU))
import u2_common as U  # noqa: E402
NS = U.NS
PROV = {}


def prov(fig):
    if fig not in PROV:
        PROV[fig] = U.Prov(fig)
    return PROV[fig]

HERE = pathlib.Path(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs')      # data paths of the original script
REV = HERE.parent
LC = REV / 'longchain_min'
LC_OLD = REV / 'longchain'
HIMLC = sorted(pathlib.Path(p) for p in glob.glob(str(REV / 'him_figdata_20261116/himout/*/revision_runs/longchain_min'))) + [REV / 'iqcb_figdata_20261116' / 'longchain_min']
HIMRS = sorted(pathlib.Path(p) for p in glob.glob(str(REV / 'him_figdata_20261116/himout/*/revision_runs/raw_snis')))
EXTRA = HERE / 'data' / 'traces_extra'
RS17 = REV / 'iqcb_figdata_20261117' / 'raw_snis'
TI = REV / 'ti_wolff_hmc'
TIW = REV / 'ti_wh'
sys.path.insert(0, str(REV / 'audits' / 'chain_audit_20261106'))
sys.path.insert(0, str(TI))
import chain_audit as CA  # noqa: E402

LS = (4, 8, 12, 16)
KAPPAS = {L: tuple(sorted(list(range(20, 31)) + [265, 275], key=lambda k: k / 100 if k < 100 else k / 1000))
          for L in (4, 8, 12, 16)}


def kval(k):
    return k / 100 if k < 100 else k / 1000


def kmil(k):
    return k * 10 if k < 100 else k


NS.apply()
COL = NS.COL                                   # 'IMH' key -> path-IMH colour
MRK = {'IMH': NS.MRK['path-IMH'], 'RT': NS.MRK['RT'], 'pRT': NS.MRK['pRT']}
REFMK = {'HMC': NS.MRK['HMC'], 'Wolff': NS.MRK['Wolff']}
LAB = {'IMH': 'path-IMH', 'RT': 'RT', 'pRT': 'pRT'}
LCOL = NS.LCOL
MS = 3.2


def out(name):
    return OUTU / f'{name}_20261117u2.pdf'


def tau_err(jf, key='absM'):
    r = U.cellinfo(str(jf))
    return (r.get(f'tau_{key}_jk_chain_mean') if r else None) or 0.0


def ms_err(tau, n):
    """Madras-Sokal error of tau_int = g/2 from a single chain of n records, window W = 5 g."""
    g = 2 * tau
    return tau * math.sqrt(2 * (2 * 5 * g + 1) / n)


# ---------------------------------------------------------------- data (unchanged)
def cell_name(L, k, pre='E2'):
    return f'{pre}_L{L}_k0{k}'


def method_files(cell):
    out_ = {}
    for m, cands in (('IMH', ('imhGx', 'imhG', 'imh', 'imhCPU')), ('RT', ('rt',)), ('pRT', ('rtp12',))):
        for c in cands:
            cand = [(d, cell) for d in (LC, LC_OLD, *HIMLC)]
            if cell.startswith('E2_'):
                b = cell.replace('E2_', 'buf_', 1)
                cand += [(LC, b), (LC / ('local_' + b.split('_k')[0]), b)]
            f = next((d / f'{n}_{c}.json' for d, n in cand if (d / f'{n}_{c}.json').exists()), None)
            if f is not None:
                out_[m] = f
                if CUR[0] is not None:
                    CUR[0].add('chains', str(f), 'u method_files: E2 cell in longchain_min, longchain, him_figdata_20261116 '
                               'longchain_min, iqcb_figdata_20261116/longchain_min; then buf_*; path-IMH imhGx>imhG>imh>imhCPU')
                break
    return out_


CUR = [None]


def traces(jf):
    b = str(jf)[:-5] + '_traces.npz'
    for p in (b, str(EXTRA / os.path.basename(b))):
        if os.path.exists(p):
            if CUR[0] is not None:
                CUR[0].add('traces', p, 'trace file of the selected chain json')
            return np.load(p)
    return None


def chain_tau(jf, key):
    d = json.load(open(jf))
    t = [max(c[key]['tau_sokal'], c[key].get('tau_batch_means') or 0.0) / 2 for c in d['chains'] if key in c]
    return float(np.median(t)) if t else float('nan'), bool(d['criteria_met']), d


def obs_from_traces(jf, L, nblock=20, nper=None):
    z = traces(jf)
    if z is None:
        return None
    d = json.load(open(jf))
    burn = int(d.get('burn', 0))
    a = z['absM'][burn:].astype(float)
    if nper is not None:
        a = a[:nper]
    V = L * L
    B = []
    for c in range(a.shape[1]):
        x = a[:, c]
        n = len(x) // nblock * nblock
        x = x[:n].reshape(nblock, -1)
        B.append(np.stack([x.mean(1), (x ** 2).mean(1), (x ** 4).mean(1)], 1))
    B = np.concatenate(B)
    nb = len(B)

    def f(mu):
        m, m2, m4 = mu
        return np.array([m, V * (m2 - m * m), 1 - m4 / (3 * m2 * m2)])
    full = f(B.mean(0))
    jk = np.array([f((B.sum(0) - B[i]) / (nb - 1)) for i in range(nb)])
    err = np.sqrt((nb - 1) / nb * ((jk - jk.mean(0)) ** 2).sum(0))
    return full, err


def wh_ref(L, k_milli):
    out_ = {}
    for a in ('wolff', 'hmc'):
        fs = glob.glob(str(TI / f'*_L{L}_k0{k_milli:03d}_{a}.json')) + glob.glob(str(TIW / f'*_L{L}_k0{k_milli:03d}_{a}.json'))
        if fs:
            out_[a] = json.load(open(fs[0]))['observables']
            if CUR[0] is not None:
                CUR[0].add('reference', fs[0], 'Wolff / HMC reference (ti_wolff_hmc, else ti_wh), 1e6 records')
    return out_


def wh_comb(L, k_milli, key):
    r = wh_ref(L, k_milli)
    v = [(r[a][key]['mean'], r[a][key]['err']) for a in r]
    if not v:
        return None
    w = np.array([1 / e ** 2 for _, e in v])
    m = np.array([x for x, _ in v])
    return float((w * m).sum() / w.sum()), float(1 / math.sqrt(w.sum()))


def post_burn_total(jf):
    z = traces(jf)
    if z is None:
        return None
    return (z['absM'].shape[0] - int(json.load(open(jf)).get('burn', 0))) * z['absM'].shape[1]


def zpanel(a):
    """z-score panel decoration: |z| < 2 band, zero line, fixed range."""
    a.axhspan(-2, 2, color=NS.BAND, lw=0, zorder=0)
    a.axhline(0, color=NS.ZERO_COLOR, lw=0.6, zorder=0)
    a.set_ylim(-4.5, 4.5)
    a.set_yticks([-4, -2, 0, 2, 4])


def kaxis(a, label=True):
    a.set_xlim(0.195, 0.305)
    a.set_xticks([0.20, 0.25, 0.30])
    if label:
        a.set_xlabel(r'$\kappa$')


def method_handles(line=False):
    return [NS.mhandle('path-IMH', line=line), NS.mhandle('RT', line=line), NS.mhandle('pRT', line=line)]


# ---------------------------------------------------------------- figure: observables
def fig_observables():
    keys = (('abs_m', r'$\langle|m|\rangle$'), ('chi', r'$\chi$'), ('binder', r'$U_4$'))
    fig, ax = plt.subplots(2, 3, figsize=(6.85, 3.9), sharex=True, gridspec_kw=dict(height_ratios=(1.6, 1)))
    grid = np.arange(150, 301, 5)
    for L in LS:
        for j, (key, _) in enumerate(keys):
            xs, ys = [], []
            for km in grid:
                v = wh_comb(L, km, key)
                if v:
                    xs.append(km / 1000)
                    ys.append(v[0])
            ax[0, j].plot(xs, ys, color=LCOL[L], lw=0.9, ls=NS.REF_LS, zorder=1)
    OFF = {'IMH': -0.0012, 'RT': 0.0, 'pRT': 0.0012}
    for L in LS:
        for k in KAPPAS[L]:
            mf = method_files(cell_name(L, k))
            tot = [t for t in (post_burn_total(jf) for jf in mf.values()) if t]
            if not tot:
                continue
            N = min(tot)
            for m, jf in mf.items():
                r = obs_from_traces(jf, L, nper=N // 4)
                if r is None:
                    continue
                for j, (key, _) in enumerate(keys):
                    ref = wh_comb(L, kmil(k), key)
                    ax[0, j].errorbar(kval(k) + OFF[m], r[0][j], yerr=r[1][j], fmt=MRK[m], ms=2.8, mfc=COL[m], mec=COL[m],
                                      ecolor=COL[m], elinewidth=0.6, zorder=3)
                    if ref:
                        zz = (r[0][j] - ref[0]) / math.hypot(r[1][j], ref[1])
                        ax[1, j].plot(kval(k) + OFF[m], np.clip(zz, -4.3, 4.3), MRK[m], ms=2.8, mfc=COL[m], mec=COL[m],
                                      zorder=3)
    for j, (key, lab) in enumerate(keys):
        ax[0, j].set_ylabel(lab)
        zpanel(ax[1, j])
        kaxis(ax[1, j])
        ax[1, j].set_ylabel(r'$z$')
        NS.panel_letter(ax[0, j], 'abc'[j])
        NS.panel_letter(ax[1, j], 'def'[j])
    ax[0, 1].set_yscale('log')
    h = [NS.lhandle(L, ls=NS.REF_LS) for L in LS] + method_handles()
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=0.6)
    NS.fig_legend(fig, h, ncol=7)
    NS.save(fig, out('fig_observables'))


ESS_LOW = 1e-3


def fig_rawsnis():
    keys = (('abs_m', r'$\delta\langle|m|\rangle$', True), ('chi', r'$\delta\chi$', True), ('binder', r'$\Delta U_4$', False))
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.15))
    any_low = False
    for L in LS:
        for est, mk in (('raw', 'o'), ('snis', 's')):
            xs, ys, es, ess = [], [], [], []
            for k in KAPPAS[L]:
                f = RSN / f'{cell_name(L, k)}.json'
                if not f.exists():
                    continue
                CUR[0].add('a-d', str(f), 'raw_snis/E2_<cell>.json (65536 paths, 20-block jackknife)')
                d = json.load(open(f))
                row = []
                for key, _, rel in keys:
                    ref = wh_comb(L, kmil(k), key)
                    v, e = d[est][key]
                    row.append(((v - ref[0]) / ref[0], e / ref[0]) if rel else (v - ref[0], e))
                xs.append(kval(k))
                ys.append([r[0] for r in row])
                es.append([r[1] for r in row])
                ess.append(d['kish_ess_frac'])
            if not xs:
                continue
            ys, es, xs = np.array(ys), np.array(es), np.array(xs)
            dx = -0.0008 if est == 'raw' else 0.0008
            low = (np.array(ess) < ESS_LOW) if est == 'snis' else np.zeros(len(xs), bool)
            for j in range(3):
                ax[j].plot(xs + dx, ys[:, j], color=LCOL[L], lw=0.6, ls='-' if est == 'snis' else ':')
                ax[j].errorbar(xs[~low] + dx, ys[~low, j], yerr=es[~low, j], fmt=mk, ms=2.8, color=LCOL[L],
                               mfc=LCOL[L] if est == 'snis' else 'white', elinewidth=0.6, ls='')
                any_low = any_low or bool(low.any())
                if low.any():
                    ax[j].plot(xs[low] + dx, ys[low, j], mk, ms=3.2, mfc='white', mec=LCOL[L], mew=0.9, ls='')
            if est == 'snis':
                ax[3].plot(xs, ess, 'o-', ms=2.8, color=LCOL[L], lw=0.8)
    for j, (_, lab, _) in enumerate(keys):
        ax[j].axhline(0, color=NS.ZERO_COLOR, lw=0.6)
        ax[j].set_ylabel(lab)
    ax[3].set_yscale('log')
    if any_low:
        ax[3].axhline(ESS_LOW, color='0.6', lw=0.6, ls=':')
    ax[3].set_ylabel(r'ESS$_w/N$')
    for j in range(4):
        kaxis(ax[j])
        NS.panel_letter(ax[j], 'abcd'[j])
    h = [NS.lhandle(L) for L in LS]
    h += [plt.Line2D([], [], ls=':', marker='o', mfc='white', color='0.3', ms=4, label='raw'),
          plt.Line2D([], [], ls='-', marker='s', color='0.3', ms=4, label='SNIS')]
    if any_low:
        h.append(plt.Line2D([], [], ls='', marker='s', mfc='white', mec='0.3', ms=4, label=r'SNIS, ESS$_w/N<10^{-3}$'))
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.2)
    NS.fig_legend(fig, h, ncol=7)
    NS.save(fig, out('fig_rawsnis'))


def fig_sscan(k=27):
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.15))
    for L in LS:
        xs, gs, ac, caps, ge, ae = [], [], [], [], [], []
        for s_ in (0, 2, 4, 6, 8, 10, 12, 14):
            mf = method_files(cell_name(L, k))
            jf = mf.get('RT') if s_ == 0 else (mf.get('pRT') if s_ == 12 else LC / f'{cell_name(L, k)}_rtp{s_}.json')
            if jf is None or not os.path.exists(jf):
                continue
            CUR[0].add('chains', str(jf), 's=0: RT, s=12: pRT of method_files; other s: longchain_min/E2_L{L}_k027_rtp{s}.json')
            g, ok, d = chain_tau(jf, 'absM')
            r = U.cellinfo(str(jf))
            xs.append(s_)
            gs.append(g)
            ge.append(tau_err(jf))
            caps.append(not ok)
            ac.append(float(np.median([c['acceptance'] for c in d['chains']])))
            ae.append(r.get('acc_jk_block') or r.get('acc_jk_chain') or 0.0)
        if not xs:
            continue
        ax[0].plot(xs, gs, '-', color=LCOL[L], lw=1.0)
        ax[0].errorbar(xs, gs, yerr=ge, fmt='o', ms=3, color=LCOL[L], elinewidth=0.6, label=f'$L={L}$')
        xs, gs, caps = np.array(xs), np.array(gs), np.array(caps)
        if caps.any():
            ax[0].plot(xs[caps], gs[caps] * 1.4, marker=r'$\uparrow$', ms=6, color=LCOL[L], ls='')
        ax[1].plot(xs, ac, '-', color=LCOL[L], lw=1.0)
        ax[1].errorbar(xs, ac, yerr=ae, fmt='o', ms=3, color=LCOL[L], elinewidth=0.6)
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$\tau_{\rm int}(|m|)$')
    ax[1].set_ylabel('acceptance')
    for j in range(2):
        ax[j].axvline(12, color='0.75', lw=0.6, ls=':', zorder=0)
        ax[j].set_xlabel(r'$s$')
        NS.panel_letter(ax[j], 'ab'[j])
    NS.tag(ax[1], r'$\kappa=0.27$', loc='upper left')
    h = [plt.Line2D([], [], color=LCOL[L], marker='o', ms=3, label=f'$L={L}$') for L in LS]
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
    NS.fig_legend(fig, h, ncol=4)
    NS.save(fig, out('fig_sscan'))
    CUR[0].note('tau error bars: delete-one-chain jackknife of the chain mean; acceptance (median over chains, as u) error: '
                '20-block-per-chain jackknife of the accept flags')


def fig_transfer_obs():
    keys = (('abs_m', r'$\langle|m|\rangle$'), ('chi', r'$\chi$'), ('binder', r'$U_4$'))
    TCOL = {24: LCOL[24], 32: LCOL[32]}
    fig, ax = plt.subplots(2, 3, figsize=(6.85, 3.9), sharex=True, gridspec_kw=dict(height_ratios=(1.6, 1)))
    for L in (24, 32):
        for j, (key, _) in enumerate(keys):
            xs, ys = [], []
            for km in np.arange(150, 301, 5):
                v = wh_comb(L, km, key)
                if v:
                    xs.append(km / 1000)
                    ys.append(v[0])
            ax[0, j].plot(xs, ys, color=TCOL[L], lw=0.9, ls=NS.REF_LS, zorder=1)
    OFF = {'IMH': -0.0012, 'RT': 0.0, 'pRT': 0.0012}
    NOCONV = []
    for L in (24, 32):
        for k in range(20, 31):
            mf = method_files(cell_name(L, k, 'T16'))
            tot = [t for t in (post_burn_total(jf) for jf in mf.values()) if t]
            if not tot:
                continue
            N = min(tot)
            for m, jf in mf.items():
                r = obs_from_traces(jf, L, nper=N // 4)
                if r is None:
                    continue
                conv = bool(json.load(open(jf))['criteria_met'])
                fc = TCOL[L] if conv else 'white'
                if not conv:
                    NOCONV.append(f'L{L} k0.{k} {LAB[m]}')
                for j, (key, _) in enumerate(keys):
                    ref = wh_comb(L, kmil(k), key)
                    ax[0, j].errorbar(kval(k) + OFF[m], r[0][j], yerr=r[1][j], fmt=MRK[m], ms=2.8, mfc=fc,
                                      mec=TCOL[L], ecolor=TCOL[L], elinewidth=0.6, zorder=3)
                    if ref:
                        zz = (r[0][j] - ref[0]) / math.hypot(r[1][j], ref[1])
                        ax[1, j].plot(kval(k) + OFF[m], np.clip(zz, -4.3, 4.3), MRK[m], ms=2.8, mfc=fc,
                                      mec=TCOL[L], zorder=3)
    for j, (key, lab) in enumerate(keys):
        ax[0, j].set_ylabel(lab)
        zpanel(ax[1, j])
        kaxis(ax[1, j])
        ax[1, j].set_ylabel(r'$z$')
        NS.panel_letter(ax[0, j], 'abc'[j])
        NS.panel_letter(ax[1, j], 'def'[j])
    ax[0, 1].set_yscale('log')
    h = [NS.lhandle(L, ls=NS.REF_LS) for L in (24, 32)]
    h += [plt.Line2D([], [], ls='', marker=MRK[m], mfc='0.4', mec='0.4', ms=4, label=LAB[m]) for m in MRK]
    h += [plt.Line2D([], [], ls='', marker='o', mfc='white', mec='0.4', ms=4, label='not converged')]
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=0.6)
    NS.fig_legend(fig, h, ncol=6)
    CUR[0].note('open markers (criteria_met = False): ' + ', '.join(NOCONV))
    NS.save(fig, out('fig_transfer_obs'))


def fig_transfer():
    """native L = 16 (solid line, open markers) vs the L16 models run at L = 24 / 32 (transfer: dashed line, filled)."""
    fig, ax = plt.subplots(2, 2, figsize=(6.7, 3.5), sharex=True, gridspec_kw=dict(height_ratios=(1.6, 1)))
    for j, L in enumerate((24, 32)):
        a = ax[0, j]
        for m in ('IMH', 'RT', 'pRT'):
            for pre, LL, ls, fc in (('E2', 16, '-', 'white'), ('T16', L, '--', None)):
                xs, ys, caps, zs, es = [], [], [], [], []
                for k in range(20, 31):
                    jf = method_files(cell_name(LL, k, pre)).get(m)
                    if jf is None:
                        continue
                    g, ok, d = chain_tau(jf, 'absM')
                    xs.append(kval(k))
                    ys.append(g)
                    es.append(tau_err(jf))
                    caps.append(not ok)
                    if pre == 'T16':
                        mm = np.array([c['absM']['mean'] for c in d['chains']])
                        se = np.array([c['absM']['se'] for c in d['chains']])
                        w = 1 / se ** 2
                        ref = wh_comb(L, kmil(k), 'abs_m')
                        zs.append(((w * mm).sum() / w.sum() - ref[0]) / math.hypot(1 / math.sqrt(w.sum()), ref[1]))
                if not xs:
                    continue
                xs, ys, caps = np.array(xs), np.array(ys), np.array(caps)
                a.plot(xs, ys, ls=ls, color=COL[m], lw=0.8)
                es = np.array(es)
                a.errorbar(xs[~caps], ys[~caps], yerr=es[~caps], fmt=MRK[m], ms=3, mfc=fc or COL[m], mec=COL[m],
                           ecolor=COL[m], elinewidth=0.6, ls='')
                a.plot(xs[caps], ys[caps], MRK[m], ms=3, mfc=fc or COL[m], mec=COL[m], ls='')
                if caps.any():
                    a.plot(xs[caps], ys[caps] * 1.5, marker=r'$\uparrow$', ms=6, color=COL[m], ls='')
                if pre == 'T16':
                    zs = np.clip(zs, -4.3, 4.3)
                    ax[1, j].plot(xs[~caps], zs[~caps], MRK[m], ms=3, color=COL[m], ls='')
                    ax[1, j].plot(xs[caps], zs[caps], MRK[m], ms=3, mfc='white', mec=COL[m], ls='')
        a.set_yscale('log')
        NS.tag(a, f'$L={L}$')
        zpanel(ax[1, j])
        kaxis(ax[1, j])
        NS.panel_letter(a, 'ab'[j])
        NS.panel_letter(ax[1, j], 'cd'[j])
    ax[0, 0].set_ylabel(r'$\tau_{\rm int}(|m|)$')
    ax[1, 0].set_ylabel(r'$z$')
    h = [NS.mhandle('path-IMH', ls='--'), NS.mhandle('RT', ls='--'), NS.mhandle('pRT', ls='--')]
    h += [plt.Line2D([], [], color='0.3', marker='o', mfc='white', ls='-', ms=4, label=r'$L=16$ native')]
    h += [plt.Line2D([], [], color='0.3', marker=r'$\uparrow$', ls='', ms=6, label='not converged')]
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=0.6)
    NS.fig_legend(fig, h, ncol=5)
    CUR[0].note('tau error bars (converged chains): delete-one-chain jackknife of the chain mean; not-converged chains: '
                'arrow (lower bound), no bar; (c, d) z of not-converged chains drawn open (u2)')
    NS.save(fig, out('fig_transfer'))


def fig_transfer_matrix(kappas=(24, 27, 30)):
    LT = (4, 8, 12, 16, 24, 32)
    SCOL = {4: LCOL[4], 12: LCOL[12], 16: LCOL[16]}
    fig, ax = plt.subplots(2, len(kappas), figsize=(7.0, 3.9), sharex=True)
    HND = {}
    for j, k in enumerate(kappas):
        for src, lab, col, mk, ls in [(None, 'native', 'k', 'o', '-')] + [(s_, f'from $L={s_}$', SCOL[s_], 's', '--') for s_ in (4, 12, 16)]:
            xs, ts, caps, es, te = [], [], [], [], []
            for L in LT:
                pre = 'E2' if src is None or src == L else f'T{src}'
                if src is not None and src == L:
                    continue
                jf = method_files(cell_name(L, k, pre)).get('pRT')
                if jf is not None:
                    t, ok, _ = chain_tau(jf, 'absM')
                    xs.append(L)
                    ts.append(t)
                    te.append(tau_err(jf))
                    caps.append(not ok)
                rf = next((f for d_ in (REV / 'raw_snis', *HIMRS, RS17) for f in (d_ / f'{cell_name(L, k, pre)}.json',
                                       d_ / f'{cell_name(L, k, pre).replace("E2_", "buf_", 1)}.json') if f.exists()), None)
                if rf is not None:
                    es.append((L, json.load(open(rf))['kish_ess_frac']))
                    CUR[0].add('d-f ESS', str(rf), 'raw_snis: local, him_figdata_20261116 raw_snis, then iqcb_figdata_20261117/raw_snis (u2)',
                               'NEW in u2 (not found by u)' if str(rf).startswith(str(RS17)) else '')
            if xs:
                xs, ts, caps, te = np.array(xs), np.array(ts), np.array(caps), np.array(te)
                ax[0, j].plot(xs, ts, ls=ls, color=col, lw=0.9)
                ax[0, j].errorbar(xs[~caps], ts[~caps], yerr=te[~caps], fmt=mk, ms=3.5, color=col, elinewidth=0.6, ls='')
                ax[0, j].plot(xs[caps], ts[caps], mk, ms=3.5, color=col, ls='')
                HND.setdefault(lab, plt.Line2D([], [], ls=ls, marker=mk, ms=3.5, color=col, lw=0.9, label=lab))
                if caps.any():
                    ax[0, j].plot(xs[caps], ts[caps] * 1.5, marker=r'$\uparrow$', ms=6, color=col, ls='')
            if es:
                ax[1, j].plot([e[0] for e in es], [e[1] for e in es], ls=ls, marker=mk, ms=3.5, color=col, lw=0.9)
        ax[0, j].set_yscale('log')
        ax[0, j].yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax[0, j].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax[0, j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[1, j].set_yscale('log')
        NS.tag(ax[0, j], f'$\\kappa=0.{k}$')
        ax[1, j].set_xlabel(r'$L$')
        ax[1, j].set_xscale('log', base=2)
        ax[1, j].set_xticks(LT)
        ax[1, j].set_xticklabels([str(L) for L in LT])
        ax[1, j].minorticks_off()
        NS.panel_letter(ax[0, j], 'abc'[j])
        NS.panel_letter(ax[1, j], 'def'[j])
    ax[0, 0].set_ylabel(r'$\tau_{\rm int}(|m|)$, pRT')
    ax[1, 0].set_ylabel(r'ESS$_w/N$')
    h = list(HND.values()) + [plt.Line2D([], [], color='0.3', marker=r'$\uparrow$', ls='', ms=6, label='not converged')]
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.8)
    NS.fig_legend(fig, h, ncol=5)
    CUR[0].note('tau error bars (converged chains): delete-one-chain jackknife of the chain mean; not converged: arrow, no bar')
    NS.save(fig, out('fig_transfer_matrix'))


def accept_quality(jf, tag):
    """u2: independent fixed seed per (L, kappa, method): seed = crc32(tag), tag = 'L{L}_k{k}_{method}'.
    (u: one shared default_rng(0), so a value depended on how many cells had been processed before.)"""
    z = traces(jf)
    if z is None:
        return None
    rng = np.random.default_rng(zlib.crc32(tag.encode()))
    d = json.load(open(jf))
    b = int(d.get('burn', 0))
    a = z['acc'][b + 1:].astype(bool)
    M = z['absM'][b:].astype(float)
    dM = np.abs(np.diff(M, axis=0))
    y = M.ravel()
    q = (float(a.mean()), float(dM[a].mean() / np.mean(np.abs(y - rng.permutation(y)))))
    JUMPS[tag] = q
    return q


JUMPS = {}


def _acc_series(L, m):
    xs, acc, jump = [], [], []
    for k in KAPPAS[L]:
        jf = method_files(cell_name(L, k)).get(m)
        q = accept_quality(jf, f'L{L}_k{k}_{m}') if jf is not None else None
        if q is not None:
            xs.append(kval(k))
            acc.append(q[0])
            jump.append(q[1])
    return xs, acc, jump


JUMP_LAB = r'$\langle|\Delta|m||\rangle_{\rm acc}\,/\,\langle|\Delta|m||\rangle_{\rm indep}$'


def fig_acceptance(L=16, hist=(16, 27)):
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.2))
    for m in ('IMH', 'RT', 'pRT'):
        xs, acc, jump = _acc_series(L, m)
        ax[0].plot(xs, acc, MRK[m] + '-', ms=3, lw=0.8, color=COL[m], label=LAB[m])
        ax[1].plot(xs, jump, MRK[m] + '-', ms=3, lw=0.8, color=COL[m])
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1.1)
    ax[1].axhline(1, color='0.6', lw=0.6, ls=':')
    ax[0].set_ylabel('acceptance')
    ax[1].set_ylabel(JUMP_LAB)
    for c in range(2):
        kaxis(ax[c])
    NS.tag(ax[0], f'$L={L}$', loc='upper right')
    Lh, k = hist
    z = np.load(RSN / f'{cell_name(Lh, k)}.npz')
    CUR[0].add('c', str(RSN / f'{cell_name(Lh, k)}.npz'), '65536 forward paths (m, W)')
    CUR[0].add('c, d', str(HERE / 'data' / 'wh_m_signed_L16.npz'), 'Wolff signed m, 2e5 records')
    m_, W = z['m'].astype(float), z['W'].astype(float)
    ok = np.isfinite(W)
    m_, W = m_[ok], W[ok]
    w = np.exp(-(W - W.min()))
    ref = np.load(HERE / 'data' / 'wh_m_signed_L16.npz')[f'L{Lh}_k0{kmil(k)}_wolff'].astype(float)
    hi = np.quantile(np.abs(np.concatenate([ref, m_])), 0.998)
    bins = np.linspace(-hi, hi, 49)
    for a in ax[2:]:
        a.hist(ref, bins=bins, density=True, color=NS.BAND, label='Wolff')
    ax[2].hist(m_, bins=bins, density=True, histtype='step', color='k', ls=':', lw=1.1, label='raw')
    ax[2].hist(m_, bins=bins, weights=w, density=True, histtype='step', color='k', lw=0.9, label='SNIS')
    for meth, jf in method_files(cell_name(Lh, k)).items():
        tr = traces(jf)
        b = int(json.load(open(jf)).get('burn', 0))
        mc = (tr['sign'][b:] * tr['absM'][b:]).astype(float).ravel()
        ax[3].hist(mc, bins=bins, density=True, histtype='step', color=COL[meth], lw=0.9, label=LAB[meth])
    top = 1.45 * max(a.get_ylim()[1] for a in ax[2:])
    for a in ax[2:]:
        a.set_ylim(0, top)
        a.set_xlabel(r'$m$')
    NS.tag(ax[2], f'$L={Lh}$, $\\kappa={kval(k)}$')
    NS.tag(ax[3], f'$L={Lh}$, $\\kappa={kval(k)}$')
    ax[2].set_ylabel(r'$P(m)$')
    ax[3].set_yticklabels([])
    for c in range(4):
        NS.panel_letter(ax[c], 'abcd'[c])
    h, _ = ax[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color='k', ls=':', lw=1.1, label='raw'), plt.Line2D([], [], color='k', lw=0.9, label='SNIS'),
          matplotlib.patches.Patch(color=NS.BAND, label='Wolff')]
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)
    NS.fig_legend(fig, h, ncol=6)
    NS.save(fig, out('fig_acceptance'))


def fig_acceptance_all():
    fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.8), sharex=True, sharey='row')
    for j, L in enumerate(LS):
        for m in ('IMH', 'RT', 'pRT'):
            xs, acc, jump = _acc_series(L, m)
            ax[0, j].plot(xs, acc, MRK[m] + '-', ms=2.8, lw=0.8, color=COL[m], label=LAB[m])
            ax[1, j].plot(xs, jump, MRK[m] + '-', ms=2.8, lw=0.8, color=COL[m])
        ax[0, j].set_ylim(0, 1)
        ax[1, j].set_ylim(0, 1.1)
        ax[1, j].axhline(1, color='0.6', lw=0.6, ls=':')
        NS.tag(ax[0, j], f'$L={L}$', loc='lower left')
        kaxis(ax[1, j])
        NS.panel_letter(ax[0, j], 'abcd'[j])
        NS.panel_letter(ax[1, j], 'efgh'[j])
    ax[0, 0].set_ylabel('acceptance')
    ax[1, 0].set_ylabel(JUMP_LAB)
    h, _ = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.8)
    NS.fig_legend(fig, h, ncol=3)
    NS.save(fig, out('fig_acceptance_all'))


# ---------------------------------------------------------------- figure: log Z
LZC = REV / 'logz_chains'
RSN = REV / 'raw_snis'
LZ_LAB = (('snis', 'SNIS', '#CC79A7', 'P'), ('IMH-rev', 'path-IMH-rev', COL['IMH'], 's'), ('RT-rev', 'RT-rev', COL['RT'], 'o'),
          ('pRT-rev', 'pRT-rev', COL['pRT'], '^'), ('IMH-bar', 'path-IMH + SNIS', COL['IMH'], 'S'),
          ('RT-bar', 'RT + SNIS', COL['RT'], 'O'), ('pRT-bar', 'pRT + SNIS', COL['pRT'], 'A'))


def chain_logz(tag):
    """as in phi4_figs_20261116, but read-only: a valid cache is used, otherwise the estimate is computed in memory."""
    sys.path.insert(0, str(REV.parent.parent / 'revision' / 'longchain'))
    import logz_estimators as LE
    out_, sn, shared = {}, [], False
    for m, lab in (('imh', 'IMH'), ('rt', 'RT'), ('rtp12', 'pRT')):
        f = LZC / f'{tag}_{m}.npz'
        if not f.exists():
            continue
        if CUR[0] is not None:
            CUR[0].add('chains', str(f), 'logz_chains/<cell>_<method>.npz (WR, WF/shared raw_snis WF); est.json cache used if valid')
        cache = f.with_suffix('.est.json')
        r = json.load(open(cache)) if cache.exists() and cache.stat().st_mtime > f.stat().st_mtime else {}
        if r.get('est_version') != LE.EST_VERSION:
            z = dict(np.load(f))
            if len(z['WF']) < 1024:
                z['WF'] = np.load(RSN / f'{tag}.npz')['W'].astype(float)
            r = LE.estimates(z)
            r['shared_fwd'] = len(np.load(f)['WF']) < 1024
        if not (r.get('shared_fwd') and shared):
            sn.append(r['snis'])
        shared = shared or bool(r.get('shared_fwd'))
        out_[f'{lab}-rev'] = tuple(r['rev'])
        out_[f'{lab}-bar'] = tuple(r['bar'])
    if sn:
        w = np.array([1 / e ** 2 for _, e in sn])
        out_['snis'] = (float((w * np.array([v for v, _ in sn])).sum() / w.sum()), float(1 / math.sqrt(w.sum())))
    return out_


def fig_logz():
    """log Z - log Z_TI vs kappa, 2 x 2 (L = 4 / 8 / 12 / 16), own y range per panel; L = 16 with an inset around zero."""
    import ti_logz as T
    D = T.load()
    A = T.ais_refs()
    CUR[0].add('reference', str(TI), 'TI: ti_wolff_hmc/*.json (Wolff + HMC dS/dkappa, 1e6 records), Simpson + /15 step-halving error')
    for f in sorted(glob.glob(str(REV / 'logz_grid' / 'logz_L*.json'))) + sorted(glob.glob(str(REV.parent / 'logz_ref_L*.json'))):
        CUR[0].add('AIS', f, 'ti_logz.ais_refs()')
    mk_open = {'P': 'P', 's': 's', 'o': 'o', '^': '^', 'S': 'D', 'O': 'h', 'A': 'v'}
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    jobs = [(axs.flat[j], j, L) for j, L in enumerate(LS)]
    INSET = 0.06

    def draw(a_, L, ti, x, s):
        a_.fill_between(x, -s, s, color=NS.BAND, lw=0, zorder=0)
        a_.axhline(0, color=NS.ZERO_COLOR, lw=0.6, zorder=0)
        ext = [float(np.max(s))]
        px = [kk for kk in ti if (L, round(kk / 1000, 3)) in A]
        if px:
            ra = [(A[(L, round(kk / 1000, 3))][0] - ti[kk][0]) for kk in px]
            ea = [A[(L, round(kk / 1000, 3))][1] for kk in px]
            a_.errorbar(np.array(px) / 1000, ra, yerr=ea, fmt='x', ms=3, color='0.4', elinewidth=0.6, zorder=2)
            ext += [abs(r) + e for r, e in zip(ra, ea)]
        for i, (key, lab, col, mk) in enumerate(LZ_LAB):
            xs, ys, es = [], [], []
            for k in KAPPAS[L]:
                v = chain_logz(f'E2_L{L}_k0{k}').get(key)
                if v is not None and kmil(k) in ti:
                    xs.append(kval(k) + (i - 3) * 0.0011)
                    ys.append(v[0] - ti[kmil(k)][0])
                    es.append(v[1])
            if xs:
                m_ = mk if mk.islower() or mk == 'P' else mk_open[mk]
                filled = key == 'snis' or key.endswith('rev')
                a_.errorbar(xs, ys, yerr=es, fmt=m_, ms=2.8, mfc=col if filled else 'white', mec=col, ecolor=col,
                            elinewidth=0.6, zorder=3)
                ext += [abs(y) + e for y, e in zip(ys, es)]
        return max(ext)

    for a_, j, L in jobs:
        C = T.combine(T.series(D, L, 'wolff'), T.series(D, L, 'hmc'))
        ks = sorted(set(range(200, 301, 10)) | {kmil(k) for k in KAPPAS[L]})
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
        kk = sorted(ti)
        x, s = np.array(kk) / 1000, np.array([ti[k][1] for k in kk])
        lim = 1.2 * draw(a_, L, ti, x, s)
        a_.set_ylim(-lim, lim)
        kaxis(a_, label=j >= 2)
        if j % 2 == 0:
            a_.set_ylabel(r'$\log Z - \log Z_{\rm TI}$')
        big = L == LS[-1] and lim > 2.5 * INSET
        NS.tag(a_, f'$L={L}$', loc='upper right' if big else 'upper left')
        NS.panel_letter(a_, 'abcd'[j])
        if big:
            ins = a_.inset_axes([0.1, 0.58, 0.5, 0.34])
            draw(ins, L, ti, x, s)
            ins.set_xlim(0.195, 0.305)
            ins.set_ylim(-INSET, INSET)
            ins.tick_params(labelsize=6)
            ins.set_xticklabels([])
            a_.axhspan(-INSET, INSET, facecolor='none', edgecolor='0.6', lw=0.5, ls=':', zorder=1)
    h = [plt.Line2D([], [], ls='', marker=mk if (key == 'snis' or key.endswith('rev')) else mk_open[mk], ms=4, mec=col,
                    mfc=col if (key == 'snis' or key.endswith('rev')) else 'white', label=lab) for key, lab, col, mk in LZ_LAB]
    h += [plt.Line2D([], [], ls='', marker='x', color='0.4', label='AIS'),
          matplotlib.patches.Patch(color=NS.BAND, label='TI (Wolff / HMC)')]
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    NS.fig_legend(fig, h, ncol=5)
    NS.save(fig, out('fig_logz'))


# ---------------------------------------------------------------- figure: convergence
def prefix_curves(x, ts):
    cm = np.cumsum(x) / np.arange(1, len(x) + 1)
    tau = np.array([max(CA.tau_sokal(x[:t]), CA.tau_bm(x[:t])) for t in ts]) / 2
    return cm[np.array(ts) - 1], tau


def fig_convergence(cells=((16, 26), (16, 27))):
    W = np.load(HERE / 'data' / 'wh_absm_traces.npz')
    CUR[0].add('reference', str(HERE / 'data' / 'wh_absm_traces.npz'), 'HMC / Wolff |m| traces (1e6)')
    fig, ax = plt.subplots(2, len(cells), figsize=(6.6, 4.0), sharex='col')
    for j, (L, k) in enumerate(cells):
        ref = wh_comb(L, kmil(k), 'abs_m')
        for m, jf in method_files(cell_name(L, k)).items():
            z = traces(jf)
            d = json.load(open(jf))
            burn = int(d.get('burn', 0))
            a = z['absM'][burn:].astype(float)
            n = a.shape[0]
            ts = np.unique(np.geomspace(200, n, 40).astype(int))
            cms, taus = [], []
            for c in range(a.shape[1]):
                cm, tau = prefix_curves(a[:, c], ts)
                cms.append(cm)
                taus.append(tau)
            cms, taus = np.array(cms), np.array(taus)
            mean = cms.mean(0)
            se = np.sqrt(np.array([a[:t].var() for t in ts]) * 2 * np.median(taus, 0) / (a.shape[1] * ts))
            ax[0, j].plot(ts, mean, color=COL[m], label=LAB[m])
            ax[0, j].fill_between(ts, mean - se, mean + se, color=COL[m], alpha=0.2, lw=0)
            ax[1, j].plot(ts, np.median(taus, 0), color=COL[m], label=LAB[m])
        for a_name, lab in (('hmc', 'HMC'), ('wolff', 'Wolff')):
            x = W[f'L{L}_k0{k}0_{a_name}'].astype(float)
            ts = np.unique(np.geomspace(200, len(x), 40).astype(int))
            cm, tau = prefix_curves(x, ts)
            se = np.sqrt(np.array([x[:t].var() for t in ts]) * 2 * tau / ts)
            ax[0, j].plot(ts, cm, color=COL[lab], ls='--', label=lab)
            ax[0, j].fill_between(ts, cm - se, cm + se, color=COL[lab], alpha=0.15, lw=0)
            ax[1, j].plot(ts, tau, color=COL[lab], ls='--', label=lab)
        ax[0, j].axhline(ref[0], color=NS.REF_COLOR, lw=0.7, ls=NS.REF_LS, label='reference')
        ax[0, j].set_xscale('log')
        ax[1, j].set_xscale('log')
        ax[1, j].set_yscale('log')
        ax[0, j].set_ylabel(r'$\langle|m|\rangle$')
        ax[1, j].set_ylabel(r'$\tau_{\rm int}(|m|)$')
        ax[1, j].set_xlabel('MC step')
        lo, hi = ref[0] * 0.8, ref[0] * 1.2
        ax[0, j].set_ylim(lo, hi)
        NS.panel_letter(ax[0, j], 'ab'[j])
        NS.panel_letter(ax[1, j], 'cd'[j])
        NS.tag(ax[0, j], f'$L={L}$, $\\kappa=0.{k}$', loc='lower right')
    h, _ = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.8)
    NS.fig_legend(fig, h, ncol=6)
    NS.save(fig, out('fig_convergence'))


# ---------------------------------------------------------------- figure: mixing
def fig_mixing(keys=('absM',), name='fig_mixing'):
    S = json.load(open(HERE / 'data' / 'wh_sign_summary.json'))
    G = json.load(open(HERE / 'data' / 'wh_tau_grid.json'))
    any_cap = False
    any_frozen = False
    fig, ax = plt.subplots(len(keys), 4, figsize=(7.0, 2.0 * len(keys) + 0.35), sharex='col', squeeze=False)
    for j, L in enumerate(LS):
        for r, key in enumerate(keys):
            a = ax[r, j]
            for m in ('IMH', 'RT', 'pRT'):
                xs, ys, caps, es = [], [], [], []
                for k in KAPPAS[L]:
                    jf = method_files(cell_name(L, k)).get(m)
                    if jf is None:
                        continue
                    t, ok, d = chain_tau(jf, key)
                    if t != t:
                        continue
                    xs.append(kval(k))
                    ys.append(t)
                    es.append(tau_err(jf, key))
                    caps.append(not ok)
                if not xs:
                    continue
                xs, ys, caps, es = np.array(xs), np.array(ys), np.array(caps), np.array(es)
                a.plot(xs, ys, ls='-', color=COL[m], lw=0.8, zorder=2)
                a.errorbar(xs[~caps], ys[~caps], yerr=es[~caps], fmt=MRK[m], ms=3.5, mfc=COL[m], mec=COL[m], ecolor=COL[m],
                           elinewidth=0.6, ls='', zorder=3)
                a.plot(xs[caps], ys[caps], MRK[m], ms=3.5, mfc=COL[m], mec=COL[m], ls='', zorder=3)
                any_cap = any_cap or bool(caps.any())
                if caps.any():
                    a.plot(xs[caps], ys[caps] * 1.6, marker=r'$\uparrow$', ms=6, color=COL[m], ls='', zorder=3)
            for alg, lab in (('hmc', 'HMC'), ('wolff', 'Wolff')):
                src = G if key == 'absM' else S
                pts = sorted((v['kappa'], v) for v in src.values() if v['L'] == L and v['alg'] == alg)
                xs = np.array([p[0] for p in pts])
                CUR[0].add('reference', str(HERE / 'data' / ('wh_tau_grid.json' if key == 'absM' else 'wh_sign_summary.json')),
                           'HMC / Wolff tau (g/2); error Madras-Sokal, W = 5 g, n records of the file')
                if key == 'absM':
                    ys = np.array([p[1]['tau_absm'] for p in pts]) / 2
                else:
                    ys = np.array([p[1]['tau_sign'] if p[1]['tau_sign'] else np.nan for p in pts]) / 2
                ns = np.array([p[1]['n'] for p in pts], float)
                ye = np.array([ms_err(y_, n_) if np.isfinite(y_) else np.nan for y_, n_ in zip(ys, ns)])
                a.errorbar(xs, ys, yerr=ye, fmt=REFMK[lab], ms=3.2, mfc='white', mec=COL[lab], ecolor=COL[lab], mew=0.8,
                           elinewidth=0.6, ls='', zorder=1)
                if key != 'absM':
                    fz = np.array([p[1]['tau_sign'] is None for p in pts])
                    any_frozen = any_frozen or bool(fz.any())
                    if fz.any():
                        n = np.array([p[1]['n'] for p in pts])
                        a.plot(xs[fz], n[fz] / 2, marker=r'$\uparrow$', ms=6, color=COL[lab], ls='', zorder=1)
            a.set_yscale('log')
            kaxis(a, label=r == len(keys) - 1)
            if j == 0:
                a.set_ylabel(r'$\tau_{\rm int}(|m|)$' if key == 'absM' else r'$\tau_{\rm int}(\mathrm{sign})$')
            NS.tag(a, f'$L={L}$')
            NS.panel_letter(a, 'abcdefgh'[r * 4 + j])
    h = method_handles(line=True)
    h += [NS.mhandle('HMC', line=False), NS.mhandle('Wolff', line=False)]
    if any_cap:
        h += [plt.Line2D([], [], color='0.3', marker=r'$\uparrow$', ls='', ms=6, label='not converged (lower bound)')]
    if any_frozen:
        h += [plt.Line2D([], [], color=COL['HMC'], marker=r'$\uparrow$', ls='', ms=6, label='no sign flips (HMC)')]
    fig.tight_layout(rect=(0, 0, 1, 0.85 if len(keys) == 1 else 0.93), w_pad=1.2)
    NS.fig_legend(fig, h, ncol=7)
    NS.save(fig, out(name))
    CUR[0].note('NHMC tau error bars: delete-one-chain jackknife of the chain mean (not-converged chains: arrow, no bar); '
                'HMC / Wolff: Madras-Sokal with W = 5 g')


def fig_mixing_sign():
    fig_mixing(('sign',), 'fig_mixing_sign')


if __name__ == '__main__':
    for w in sys.argv[1:]:
        CUR[0] = prov(f'fig_{w}')
        globals()[f'fig_{w}']()
        CUR[0].dump(extra=dict(jump_ratio_and_acc={k: v for k, v in JUMPS.items()}) if w.startswith('acceptance') else None)
        CUR[0] = None
