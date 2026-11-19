#!/usr/bin/env python3
"""2026-11-18 u5 of figs_20261117u4/dw_lj_figs_u4.py: only panel (f) changed.
  (f) per-seed stationary energy W2 = energy_w2_common of unify_20261118/item4_lj13_summary.json (same gen subsamples as u4,
      reference = the same three 2000-config test-split subsamples for every method);
      grey band = mean +- 1 sd of val2000_vs_common_ref_mean3 (unify_20261118/item4_lj13_floor.json; 200 val subsamples vs the
      same three common reference subsamples), same band definition (mean +- sd) as u4.
  (a)-(e) unchanged (checked point by point against figdata/u4/fig_dw_lj13.json).
u4 docstring: 2026-11-17 u4 of figs_20261117u2/dw_lj_figs_u2.py (data unchanged, drawing only):
  - the shared legend entry "reference floor" stood for two different grey bands; it is removed and each band gets an
    entry in its own panel: (e) "PT subsample, 5-95%" (W2 of 200 equally large PT subsamples vs the full PT reference),
    (f) "val vs test, mean +- sd" (20 replicate comparisons of the two reference splits, floor20_lj13.json);
  - (f) y label "stationary W2(E)".
u2 docstring: 2026-11-17 u2 of figs_20261117u/dw_lj_figs_20261117u.py (fig_dw_lj13). Changes against u:
  (c) error bar = delete-one-run jackknife of the mean over the 3 runs (u: np.std of the 3 runs, a spread);
  (e) has its own in-panel legend (path-IMH, RT): no pRT curve exists because the pRT relaxation (relax_rtp8/res.json)
      stored only the ensemble mean / se, no per-chain energies (u: pRT appeared only in the shared legend);
  (f) pRT now 3 seeds: 93492 r0 + HIM 93490 r1, r2 (him_figdata_20261117_extra; same model lj13sc_a05T56_s0
      kernel_best.pt, same settings, seeds 20462029 / 20463029); mean +- delete-one-seed jackknife written to
      derived_lj13_u2.json;
  every file read is recorded in prov/fig_dw_lj13.json.

u docstring: 2026-11-17 restyled copy of figs/dw_lj_figs_20261116.py (unified style, nhmc_style.py): only styling, output dir and name
(figs_20261117u/fig_dw_lj13_20261117u.pdf) changed; data paths absolute, data and selection unchanged.

2026-11-16 copy of figs/dw_lj_figs.py: MW4/MW8 labels (many-well, not the particle DW4); pRT added to (d) from
lj13_noHMC_93492_20261116/lj13/relax_rtp8 (res.json relaxation mean/se, 3 runs x 256); (f) replaced by the chains without
equilibrium starts (lj13_noHMC_93492_20261116, lit2k n=2000 vs the Klein test split, one point per seed) with the 20-replicate
val-vs-test floor (particle_202610/lit/stat2/floor20_lj13.json); SNIS dropped. Output figs_20261116/fig_dw_lj13_20261116.pdf.

DW / LJ13 revision figure (2026-11-07): fig_dw_lj13.pdf, same checkpoint per system, IMH vs RT vs pRT, HMC reference.

(a, b) DW4 / DW8 (v4 many-well models dw4_anneal_seed0 / dw8_long_score_seed0): mode-population L1 error of the 256-chain
       ensemble vs MC step; chains start at forward-path endpoints in the least likely mode (--init proposal_worst,
       dw_noHMC/dw{4,8}_relax_worst.json; IMH / RT / pRT on the same model).
(c)    stationary mode-change rate per MC step (dw_noHMC/dw{4,8}_A_<method>.json, chains from forward endpoints;
       HMC without / with the exact sign flip dw*_hmc_flip{0,1}).
(d)    LJ13 (lj13sc_a05T56_s0): ensemble mean energy vs MC step from the forward-path endpoints (lj13/relax_*_res, 3 runs x 256).
(e)    energy W2 of the ensemble vs the parallel-tempering reference (lj13_pt_reference, 204800 configurations) vs MC step;
       grey band = W2 of an equally large reference subsample (finite-sample floor).
(f)    stationary energy W2 (AS protocol, n = 2000, vs the PT reference; lj13/A_pt), mean +- sd over 3 runs; PT floor dashed."""
import os
import glob
import json
import pathlib

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs')
import sys  # noqa: E402
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs_20261118u5')
import u5_common as U  # noqa: E402
NS = U.NS
PR = U.Prov('fig_dw_lj13')
EXTRA93490 = U.REV + '/him_figdata_20261117_extra/himout/93490/revision_runs/lj13_noHMC'


def jkm(v):
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return float('nan')
    r = np.array([np.delete(v, i).mean() for i in range(n)])
    return float(np.sqrt((n - 1) / n * ((r - r.mean()) ** 2).sum()))
import matplotlib.patches  # noqa: E402,F401
REV = HERE.parent
COL = {'IMH': NS.COL['path-IMH'], 'RT': NS.COL['RT'], 'pRT': NS.COL['pRT'], 'HMC': NS.COL['HMC'], 'HMC + flip': '#999999'}
MK = {'IMH': NS.MRK['path-IMH'], 'RT': NS.MRK['RT'], 'pRT': NS.MRK['pRT'], 'HMC': NS.MRK['HMC'], 'HMC + flip': NS.MRK['HMC']}
OPEN = {'HMC', 'HMC + flip'}
DISP = {'IMH': 'path-IMH', 'RT': 'RT', 'pRT': 'pRT', 'HMC': 'HMC', 'HMC + flip': 'HMC + flip'}
NS.apply()
MKEY = {'path_imh': 'IMH', 'round_trip': 'RT', 'rt_partial': 'pRT'}


def letter(ax, s):
    NS.panel_letter(ax, s)


def w2_1d(a, b, q=2001):
    p = np.linspace(0.0005, 0.9995, q)
    return float(np.sqrt(np.mean((np.quantile(a, p) - np.quantile(b, p)) ** 2)))


def main():
    fig, ax = plt.subplots(2, 3, figsize=(6.45, 4.2))
    # (a, b) DW relaxation of the mode-population error
    for j, sysn in enumerate(('dw4', 'dw8')):
        d = json.load(open(REV / 'dw_noHMC' / f'{sysn}_relax_worst.json'))
        PR.add('ab'[j], str(REV / 'dw_noHMC' / f'{sysn}_relax_worst.json'), '3 runs x 256 chains from proposal_worst (unchanged)')
        for m, lab in MKEY.items():
            mm = next((k for k in d['methods'] if k.split(':')[0] == m), None)
            if mm is None:
                continue
            R = [r['relaxation']['mode_l1_per_transition'] for r in d['methods'][mm]['runs']]
            t = np.array(R[0]['t'])
            y = np.mean([r['mean'] for r in R], 0)
            s = np.sqrt(np.mean([np.square(r['se']) for r in R], 0) / len(R))
            ax[0, j].plot(t, y, color=COL[lab], label=lab)
            ax[0, j].fill_between(t, np.maximum(y - s, 1e-4), y + s, color=COL[lab], alpha=0.2, lw=0)
        ax[0, j].set_xscale('log')
        ax[0, j].set_yscale('log')
        ax[0, j].set_xlabel('MC step')
        ax[0, j].set_ylabel('mode-population L1')
        NS.tag(ax[0, j], sysn.upper().replace('DW', 'MW'), loc='upper right')
        letter(ax[0, j], 'ab'[j])
    # (c) stationary mode-change rate
    xs = {'dw4': 0, 'dw8': 1}
    for sysn, x0 in xs.items():
        rows = []
        for f in sorted((REV / 'dw_noHMC').glob(f'{sysn}_A_*.json')):
          PR.add('c', str(f), 'mode_change_rate of the 3 runs')
          for m, v in json.load(open(f))['methods'].items():
            if not v.get('runs') or 'mode_change_rate' not in v['runs'][0]:
                continue
            lab = 'IMH' if m == 'path_imh' else 'RT' if m == 'round_trip' else 'pRT' if m.startswith('rt_partial') else None
            if lab:
                rows.append((lab, [r['mode_change_rate'] for r in v['runs']]))
        rows.sort(key=lambda r: ['IMH', 'RT', 'pRT'].index(r[0]))
        for fl, lab in ((0, 'HMC'), (1, 'HMC + flip')):
            h = json.load(open(REV / 'dw' / f'{sysn}_hmc_flip{fl}.json'))
            PR.add('c', str(REV / 'dw' / f'{sysn}_hmc_flip{fl}.json'), 'HMC reference, 3 runs')
            rows.append((lab, [float(r['mode_change_rate']) for r in h['runs']]))
        for i, (lab, v) in enumerate(rows):
            dx = (i - 2) * 0.12
            ax[0, 2].errorbar(x0 + dx, np.mean(v), yerr=jkm(v), fmt=MK[lab], ms=4, color=COL[lab],
                              mfc='white' if lab in OPEN else COL[lab], label=DISP[lab] if sysn == 'dw4' else None)
    ax[0, 2].set_xticks([0, 1], ['MW4', 'MW8'])
    ax[0, 2].set_xlim(-0.5, 1.5)
    ax[0, 2].set_yscale('log')
    ax[0, 2].set_ylabel('mode changes / step')
    letter(ax[0, 2], 'c')
    # (d, e) LJ13 relaxation
    pt = np.load(HERE / 'data' / 'lj13_pt_ref_energies.npy').astype(float)
    PR.add('d, e', str(HERE / 'data' / 'lj13_pt_ref_energies.npy'), 'PT reference energies (204800)')
    rng = np.random.default_rng(0)
    for lab, sub in (('IMH', 'relax_imh_res/imh'), ('RT', 'relax_rt_res/rt'), ('pRT', 'relax_rtp8_res/rt_partial8')):
        fs = sorted(glob.glob(str(REV / 'lj13' / f'{sub}_run*_traces.npz')))
        for f in fs:
            PR.add('d, e', f, 'relaxation traces, 3 runs x 256 chains')
        if not fs:                                   # pRT: summary curve only (no traces) -> panel (d) only
            PR.add('d', str(REV / 'lj13_noHMC_93492_20261116' / 'lj13' / 'relax_rtp8' / 'res.json'), 'pRT relaxation: mean/se only (no traces) -> (d) only')
            R = [r['relaxation']['energy'] for r in json.load(open(REV / 'lj13_noHMC_93492_20261116' / 'lj13' / 'relax_rtp8' / 'res.json'))['methods']['rt_partial:8']['runs']]
            t = np.array(R[0]['t']); mu = np.mean([r['mean'] for r in R], 0); se = np.sqrt(np.mean([np.square(r['se']) for r in R], 0) / len(R))
            ax[1, 0].plot(t, mu, color=COL[lab], label=lab); ax[1, 0].fill_between(t, mu - se, mu + se, color=COL[lab], alpha=0.2, lw=0)
            continue
        E = np.concatenate([np.load(f)['energy'] for f in fs], 1)
        t = np.arange(1, E.shape[0] + 1)
        mu = E.mean(1)
        se = E.std(1) / np.sqrt(E.shape[1])
        ax[1, 0].plot(t, mu, color=COL[lab], label=lab)
        ax[1, 0].fill_between(t, mu - se, mu + se, color=COL[lab], alpha=0.2, lw=0)
        ts = np.unique(np.geomspace(1, E.shape[0], 40).astype(int))
        ax[1, 1].plot(ts, [w2_1d(E[k - 1], pt) for k in ts], color=COL[lab], label=lab)
        nens = E.shape[1]
    fl = [w2_1d(rng.choice(pt, nens, replace=False), pt) for _ in range(200)]
    ax[1, 1].axhspan(np.percentile(fl, 5), np.percentile(fl, 95), color=NS.BAND, lw=0, label='reference floor')
    ax[1, 0].axhline(pt.mean(), color=NS.REF_COLOR, ls=NS.REF_LS, lw=0.8, label='PT reference')
    for a in (ax[1, 0], ax[1, 1]):
        a.set_xscale('log')
        a.set_xlabel('MC step')
    ax[1, 1].set_yscale('log')
    ax[1, 1].legend(handles=[plt.Line2D([], [], color=COL['IMH'], label='path-IMH'), plt.Line2D([], [], color=COL['RT'], label='RT'),
                             matplotlib.patches.Patch(color=NS.BAND, label='PT subsample, 5–95%')],      # u4: own band entry
                    loc='upper right', handlelength=1.2, borderaxespad=0.1)
    ax[1, 1].set_ylim(top=ax[1, 1].get_ylim()[1] * 12)                                  # u4: room for the legend
    ax[1, 0].set_ylabel(r'$\langle E\rangle$')
    ax[1, 1].set_ylabel(r'$W_2(E)$')
    NS.tag(ax[1, 0], 'LJ13', loc='upper right')
    letter(ax[1, 0], 'd')
    letter(ax[1, 1], 'e')
    # (f) stationary energy W2, chains from forward endpoints, one point per seed
    UNI = REV / 'unify_20261118'
    SUMJ, FLJ = UNI / 'item4_lj13_summary.json', UNI / 'item4_lj13_floor.json'
    sm = json.load(open(SUMJ))
    flj = json.load(open(FLJ))
    PR.add('f', str(SUMJ), 'u5: per_seed[method][r0,r1,r2].energy_w2_common (common reference subsamples)', 'NEW in u5')
    PR.add('f', str(FLJ), 'u5: val2000_vs_common_ref_mean3, band mean +- 1 sd', 'NEW in u5')
    FV = {}
    for i, (lab, key) in enumerate((('IMH', 'path-IMH'), ('RT', 'RT'), ('pRT', 'pRT'))):
        ps = sm['per_seed'][key]
        v = [ps[r]['energy_w2_common'] for r in ('r0', 'r1', 'r2')]
        FV[lab] = dict(energy_w2=v, dist_w2=[ps[r]['dist_w2_common'] for r in ('r0', 'r1', 'r2')],
                       eq_w2=[ps[r]['eq_w2_common'] for r in ('r0', 'r1', 'r2')])
        ax[1, 2].plot(i + np.linspace(-0.08, 0.08, len(v)) if len(v) > 1 else [i], v, MK[lab], ms=4, color=COL[lab])
    flo = flj['val2000_vs_common_ref_mean3']
    assert flo == sm['floor']['val2000_vs_common_ref_mean3']
    ax[1, 2].axhspan(flo['mean'] - flo['sd'], flo['mean'] + flo['sd'], color=NS.BAND, lw=0)
    ax[1, 2].set_xticks(range(3), ['path-IMH', 'RT', 'pRT'])
    ax[1, 2].set_xlim(-0.5, 2.5)
    ax[1, 2].set_ylabel(r'stationary $W_2(E)$')                                         # u4
    ax[1, 2].legend(handles=[matplotlib.patches.Patch(color=NS.BAND, label='val vs test, ±1 sd')],  # u4: own band entry
                    loc='upper right', handlelength=1.2, borderaxespad=0.1)
    ax[1, 2].set_ylim(top=ax[1, 2].get_ylim()[1] + 0.06)                                 # u4: room for the entry
    letter(ax[1, 2], 'f')
    h = [plt.Line2D([], [], color=COL[k], marker=MK[k], mfc='white' if k in OPEN else COL[k],
                    ls='' if k.startswith('HMC') else '-', ms=4, label=DISP[k]) for k in COL]
    h += [plt.Line2D([], [], color=NS.REF_COLOR, ls=NS.REF_LS, label='PT reference')]   # u4: band entries moved to (e), (f)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.0, w_pad=1.2)
    NS.fig_legend(fig, h, ncol=6)
    NS.save(fig, U.pdf('fig_dw_lj13'))
    out = {}
    for lab, d in FV.items():
        out[lab] = {q: dict(values=d[q], mean=float(np.mean(d[q])), jk=jkm(d[q]), min=float(min(d[q])), max=float(max(d[q])))
                    for q in ('energy_w2', 'dist_w2', 'eq_w2')}
    rt = FV['RT']['energy_w2'] + FV['pRT']['energy_w2']
    rd = FV['RT']['dist_w2'] + FV['pRT']['dist_w2']
    out['RT_and_pRT_energy_w2_range'] = [min(rt), max(rt)]
    out['RT_and_pRT_dist_w2_range'] = [min(rd), max(rd)]
    out['floor_f'] = flo
    out['gap_common'] = sm['gap_common']
    PR.note('u5: (f) common-reference energy W2 and common-reference floor band; (a)-(e) unchanged')
    json.dump(out, open(U.OUT + '/derived_lj13_u5.json', 'x'), indent=1)
    PR.dump(extra=out)
    print(json.dumps({k: v for k, v in out.items() if k.startswith('RT_and')}), {l: (round(out[l]['energy_w2']['mean'], 4), round(out[l]['energy_w2']['jk'], 4)) for l in FV}, flo)

if __name__ == '__main__':
    main()
