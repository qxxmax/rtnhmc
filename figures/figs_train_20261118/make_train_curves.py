"""NHMC phi4 training curves (sketch, 2026-11-18): Var(log w) vs epoch from eval_log.csv.
(a) L=8, T scan; (b) T=16, L scan; (c) epochs to reach 1.1 x final Var vs T and vs V, power-law fits.
Reads csv only; writes train_curves.pdf + train_curves_figdata.json in this folder."""
import os
import glob, hashlib, json, os, re, sys
import numpy as np
import matplotlib.pyplot as plt

REV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'
OUT = os.path.join(REV, 'figs_train_20261118')
sys.path.insert(0, os.path.join(REV, 'figs_20261117u'))
import nhmc_style as NS
NS.apply()

HIM = f'{REV}/him_figdata_20261117/himout/*/revision_runs/kscan_cpu'
IQ = f'{REV}/iqcb_figdata_20261117/phi4_buffer'
EPOCH_FULL = 20000
TCOL = {4: '#9ecae1', 8: '#6baed6', 16: '#4292c6', 32: '#1764ab', 64: '#08306b'}  # same Blues ramp as LCOL


def read(f):
    a = np.genfromtxt(f, delimiter=',', names=True)
    return np.atleast_1d(a['epoch']).astype(int), np.atleast_1d(a['var_w']).astype(float)


def pick(files):
    """identical copies -> one; different copies -> the longest; tie -> prefer iqcb (seed-0 location)."""
    cand = []
    for f in files:
        b = open(f, 'rb').read()
        ep, v = read(f)
        cand.append(dict(f=f, md5=hashlib.md5(b).hexdigest(), ep=ep, v=v, iq='iqcb_figdata' in f))
    cand.sort(key=lambda c: (len(c['ep']), c['iq']), reverse=True)
    best = cand[0]
    others = [c for c in cand[1:] if c['md5'] != best['md5']]
    conflict = []
    for c in others:  # shorter copies should be prefixes of the chosen one
        n = len(c['ep'])
        if not (np.array_equal(c['ep'], best['ep'][:n]) and np.allclose(c['v'], best['v'][:n])):
            conflict.append(c['f'])
    return best, conflict


runs = {}   # (L, T, seed) -> dict
notes = []


def add(key, files, src):
    if not files:
        return
    best, conflict = pick(files)
    runs[key] = dict(L=key[0], T=key[1], seed=key[2], src=src, file=best['f'], epoch=best['ep'].tolist(),
                     var_w=best['v'].tolist(), last_epoch=int(best['ep'][-1]), n_copies=len(files),
                     conflicting_copies=conflict)
    if conflict:
        notes.append(f'{key}: copies differ (not prefixes): chosen {best["f"]}, other {conflict}')


# T scan (kappa=0.27): kscan dirs on HIM + seed-0 dirs on iqcb
names = set(os.path.basename(os.path.dirname(f)) for f in glob.glob(f'{HIM}/L*_k27_K*_s*/eval_log.csv'))
names |= set(os.path.basename(os.path.dirname(f)) for f in glob.glob(f'{IQ}/L*_k27_K*_s*/eval_log.csv'))
for n in names:
    m = re.fullmatch(r'L(\d+)_k27_K(\d+)_s(\d+)', n)
    files = glob.glob(f'{HIM}/{n}/eval_log.csv') + glob.glob(f'{IQ}/{n}/eval_log.csv')
    add((int(m[1]), int(m[2]), int(m[3])), files, 'kscan')

# native T=16 models (kappa=0.27, dir names with or without the dot)
for f in glob.glob(f'{IQ}/L*_k0.27_buf_s*/eval_log.csv') + glob.glob(f'{IQ}/L*_k027_buf_s*/eval_log.csv'):
    m = re.fullmatch(r'L(\d+)_k0\.?27_buf_s(\d+)', os.path.basename(os.path.dirname(f)))
    cfg = json.load(open(os.path.join(os.path.dirname(f), 'config.json')))
    assert cfg['num_steps'] == 16 and abs(cfg['kappa_min'] - 0.27) < 1e-9 and cfg['n_leapfrog'] == 5
    key = (int(m[1]), 16, int(m[2]))
    assert key not in runs
    add(key, [f], 'native')
# kappa=0.275 native models: only used as a marked stand-in where 0.27 is missing (L=4 s0 n/a, L=12)
alt = {}
for f in glob.glob(f'{IQ}/L*_k0275_buf_s0/eval_log.csv'):
    L = int(re.match(r'L(\d+)', os.path.basename(os.path.dirname(f)))[1])
    ep, v = read(f)
    alt[L] = dict(L=L, T=16, kappa=0.275, file=f, epoch=ep.tolist(), var_w=v.tolist())


def group(L, T):
    return [runs[k] for k in sorted(runs) if k[0] == L and k[1] == T]


def stack(rs):
    ep = sorted(set(e for r in rs for e in r['epoch']))
    mean, lo, hi, n = [], [], [], []
    for e in ep:
        vals = [r['var_w'][r['epoch'].index(e)] for r in rs if e in r['epoch']]
        mean.append(np.mean(vals)); lo.append(min(vals)); hi.append(max(vals)); n.append(len(vals))
    return np.array(ep), np.array(mean), np.array(lo), np.array(hi), np.array(n)


def t_reach(ep, v, fac=1.1):
    """first epoch with Var <= fac * final Var (final = value at the last eval), log-linear interpolation
    between eval points; censored (= first eval epoch, flag) if already below at the first eval."""
    ep, v = np.asarray(ep, float), np.asarray(v, float)
    thr = fac * v[-1]
    i = int(np.argmax(v <= thr))
    if i == 0:
        return float(ep[0]), True
    x0, x1, y0, y1 = ep[i - 1], ep[i], np.log(v[i - 1]), np.log(v[i])
    return float(x0 + (np.log(thr) - y0) / (y1 - y0) * (x1 - x0)), False


def plfit(x, y, s=None):
    """log-log weighted LS: log y = a + p log x; returns p, dp (scaled by sqrt(chi2/dof) if > 1), a."""
    lx, ly = np.log(x), np.log(y)
    w = np.ones_like(lx) if s is None else 1 / np.asarray(s) ** 2
    A = np.vstack([np.ones_like(lx), lx]).T
    C = np.linalg.inv(A.T @ (A * w[:, None]))
    beta = C @ A.T @ (w * ly)
    r = ly - A @ beta
    dof = max(len(lx) - 2, 1)
    chi2 = float(np.sum(w * r ** 2))
    scale = max(1.0, chi2 / dof) if s is not None else chi2 / dof
    return float(beta[1]), float(np.sqrt(C[1, 1] * scale)), float(beta[0]), chi2, dof


GRID_SIG = 0.15   # floor on sigma(log t) : 2000-epoch eval spacing, interpolation resolution


def reach_stats(rs):
    """per-seed reach epochs (complete runs only); geometric mean and sigma(log) for the fit."""
    full = [r for r in rs if r['last_epoch'] >= EPOCH_FULL]
    out = []
    for r in full:
        t, cens = t_reach(r['epoch'], r['var_w'])
        out.append(dict(seed=r['seed'], t=t, censored=cens, final=r['var_w'][-1]))
    if not out:
        return None
    lt = np.log([o['t'] for o in out])
    sd = float(np.std(lt, ddof=1) / np.sqrt(len(lt))) if len(lt) > 1 else 0.0
    return dict(per_seed=out, t_geo=float(np.exp(lt.mean())), t_min=float(np.exp(lt.min())),
                t_max=float(np.exp(lt.max())), sig_log=float(np.hypot(sd, GRID_SIG)), n=len(out),
                any_censored=any(o['censored'] for o in out),
                final_mean=float(np.mean([o['final'] for o in out])))


# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(NS.FULL_W, 2 * NS.ROW_H + 0.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.62, wspace=0.95)
axa = fig.add_subplot(gs[0, 0:2]); axb = fig.add_subplot(gs[0, 2:4])
axc1 = fig.add_subplot(gs[1, 0:2]); axc2 = fig.add_subplot(gs[1, 2:4])
figdata = dict(description=__doc__, quantity='var_w = Var(log w) from eval_log.csv', kappa=0.27,
               reach_def='first epoch with var_w <= 1.1 * var_w(last eval), log-linear interpolation; '
                         'only runs complete to 20000 epochs; geometric mean over seeds',
               panel_a={}, panel_b={}, panel_c={}, runs={f'L{k[0]}_T{k[1]}_s{k[2]}': runs[k] for k in sorted(runs)},
               kappa0275_standins=alt, notes=notes)

# (a) L = 8 T scan
incomplete = []
for T in [4, 8, 16, 32, 64]:
    rs = group(8, T)
    ep, m, lo, hi, n = stack(rs)
    c = TCOL[T]
    if len(rs) > 1:
        axa.fill_between(ep, lo, hi, color=c, alpha=0.22, lw=0)
    axa.plot(ep, m, color=c, marker='o', ms=2.2, label=f'$T={T}$' + (r'$^\dagger$' if T == 64 else ''))
    for r in rs:
        if r['last_epoch'] < EPOCH_FULL:
            incomplete.append((T, r['seed'], r['last_epoch']))
    figdata['panel_a'][f'T{T}'] = dict(epoch=ep.tolist(), mean=m.tolist(), min=lo.tolist(), max=hi.tolist(),
                                       n_seeds=n.tolist(), seeds=[r['seed'] for r in rs],
                                       last_epoch={r['seed']: r['last_epoch'] for r in rs})
axa.set_yscale('log'); axa.set_xlabel('epoch'); axa.set_ylabel(r'Var$(\log w)$')
axa.set_xlim(0, 21000); axa.set_xticks([0, 5000, 10000, 15000, 20000]); axa.set_xticklabels(['0', '5k', '10k', '15k', '20k'])
NS.tag(axa, r'$L=8,\ \kappa=0.27$', loc='lower left')
axa.legend(loc='upper right', ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
ylo, yhi = axa.get_ylim(); axa.set_ylim(ylo, yhi * 2.2)

# (b) T = 16 L scan
for L in [4, 6, 8, 10, 12, 16]:
    rs = group(L, 16)
    c = NS.LCOL[L]
    if rs:
        ep, m, lo, hi, n = stack(rs)
        if len(rs) > 1:
            axb.fill_between(ep, lo, hi, color=c, alpha=0.22, lw=0)
        axb.plot(ep, m, color=c, marker='o', ms=2.2, label=f'$L={L}$')
        figdata['panel_b'][f'L{L}'] = dict(epoch=ep.tolist(), mean=m.tolist(), min=lo.tolist(), max=hi.tolist(),
                                           n_seeds=n.tolist(), seeds=[r['seed'] for r in rs], kappa=0.27)
    elif L in alt:
        a = alt[L]
        axb.plot(a['epoch'], a['var_w'], color=c, ls='--', marker='o', ms=2.2, mfc='white',
                 label=f'$L={L}^*$')
        figdata['panel_b'][f'L{L}'] = dict(epoch=a['epoch'], mean=a['var_w'], kappa=0.275, seeds=[0],
                                           note='no kappa=0.27 T=16 run; kappa=0.275 native s0 shown dashed')
axb.set_yscale('log'); axb.set_xlabel('epoch'); axb.set_ylabel(r'Var$(\log w)$')
axb.set_xlim(0, 21000); axb.set_xticks([0, 5000, 10000, 15000, 20000]); axb.set_xticklabels(['0', '5k', '10k', '15k', '20k'])
NS.tag(axb, r'$T=16,\ \kappa=0.27$', loc='lower left')
axb.legend(loc='upper right', ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
ylo, yhi = axb.get_ylim(); axb.set_ylim(ylo, yhi * 2.2)

# (c1) reach epoch vs T, per L
fitT = {}
for L in [4, 6, 8, 16]:
    pts = []
    for T in [4, 8, 16, 32, 64]:
        s = reach_stats(group(L, T))
        if s:
            pts.append((T, s))
    if not pts:
        continue
    Ts = np.array([p[0] for p in pts], float); tg = np.array([p[1]['t_geo'] for p in pts])
    sg = np.array([p[1]['sig_log'] for p in pts])
    c = NS.LCOL[L]
    yerr = np.array([tg - [p[1]['t_min'] for p in pts], [p[1]['t_max'] for p in pts] - tg])
    axc1.errorbar(Ts, tg, yerr=yerr, color=c, marker='o', ls='', ms=3, elinewidth=0.7)
    for T, p in pts:
        if p['any_censored']:
            axc1.plot(T, p['t_geo'], marker='v', color=c, ms=5, mfc='none', ls='')
    entry = dict(T=Ts.tolist(), t_geo=tg.tolist(), sig_log=sg.tolist(),
                 per_T={int(T): p for T, p in pts})
    if len(Ts) >= 3:
        p_, dp, a_, chi2, dof = plfit(Ts, tg, sg)
        xx = np.geomspace(Ts.min() * 0.85, Ts.max() * 1.15, 50)
        axc1.plot(xx, np.exp(a_) * xx ** p_, color=c, ls=NS.FIT_LS, lw=0.8)
        entry.update(exponent=p_, exponent_err=dp, chi2=chi2, dof=dof)
        fitT[L] = (p_, dp)
    figdata['panel_c'][f'vsT_L{L}'] = entry
axc1.set_xscale('log'); axc1.set_yscale('log')
axc1.set_xticks([4, 8, 16, 32, 64]); axc1.set_xticklabels(['4', '8', '16', '32', '64']); axc1.minorticks_off()
axc1.set_xlabel('$T$'); axc1.set_ylabel(r'epochs to $1.1\,$Var$_{\rm final}$')
hs = [plt.Line2D([], [], color=NS.LCOL[L], marker='o', ms=3, ls=NS.FIT_LS, lw=0.8,
                 label=f'$L={L}$: ${fitT[L][0]:+.2f}\\pm{fitT[L][1]:.2f}$') for L in fitT]
axc1.legend(handles=hs, loc='lower right', fontsize=6.3, handlelength=1.6, title='slope', title_fontsize=6.3)

# (c2) reach epoch vs V at T = 16
pts = []
for L in [4, 6, 8, 10, 12, 16]:
    s = reach_stats(group(L, 16))
    if s:
        pts.append((L, s, 0.27))
    elif L in alt:
        t, cens = t_reach(alt[L]['epoch'], alt[L]['var_w'])
        pts.append((L, dict(t_geo=t, t_min=t, t_max=t, sig_log=GRID_SIG, n=1, any_censored=cens,
                            per_seed=[dict(seed=0, t=t, censored=cens, final=alt[L]['var_w'][-1])]), 0.275))
for L, s, k in pts:
    c = NS.LCOL[L]
    V = L * L
    mfc = c if k == 0.27 else 'white'
    axc2.errorbar([V], [s['t_geo']], yerr=[[s['t_geo'] - s['t_min']], [s['t_max'] - s['t_geo']]], color=c,
                  marker='o', mfc=mfc, ls='', ms=3.5, elinewidth=0.7)
    if s['any_censored']:
        axc2.plot(V, s['t_geo'], marker='v', color=c, ms=5, mfc='none', ls='')
main = [(L, s) for L, s, k in pts if k == 0.27]
Vs = np.array([L * L for L, s in main], float); tg = np.array([s['t_geo'] for L, s in main])
sg = np.array([s['sig_log'] for L, s in main])
pV, dpV, aV, chi2V, dofV = plfit(Vs, tg, sg)
allp = [(L, s) for L, s, k in pts]
pV2, dpV2, _, _, _ = plfit(np.array([L * L for L, s in allp], float), np.array([s['t_geo'] for L, s in allp]),
                           np.array([s['sig_log'] for L, s in allp]))
xx = np.geomspace(Vs.min() * 0.8, 256 * 1.25, 50)
axc2.plot(xx, np.exp(aV) * xx ** pV, color=NS.FIT_COLOR, ls=NS.FIT_LS, lw=0.8)
axc2.set_xscale('log'); axc2.set_yscale('log')
axc2.set_xticks([16, 36, 64, 100, 144, 256]); axc2.set_xticklabels(['16', '36', '64', '100', '144', '256'])
axc2.minorticks_off()
axc2.set_xlabel('$V=L^2$'); axc2.set_ylabel(r'epochs to $1.1\,$Var$_{\rm final}$')
NS.tag(axc2, f'$T=16$\nslope ${pV:+.2f}\\pm{dpV:.2f}$', loc='upper left')
figdata['panel_c']['vsV_T16'] = dict(points=[dict(L=L, V=L * L, kappa=k, **s) for L, s, k in pts],
                                     exponent=pV, exponent_err=dpV, chi2=chi2V, dof=dofV,
                                     fit_points='kappa=0.27 only',
                                     exponent_incl_kappa0275=pV2, exponent_incl_kappa0275_err=dpV2)
for ax in (axc1, axc2):
    ax.set_ylim(500, 30000)
NS.panel_letter(axa, 'a'); NS.panel_letter(axb, 'b'); NS.panel_letter(axc1, 'c')

NS.save(fig, os.path.join(OUT, 'train_curves.pdf'))
figdata['caption'] = ('Training curves of NHMC on 2d phi4, kappa=0.27, Var(log w) from eval_log.csv every 2000 epochs. '
    '(a) L=8, T=4..64; line = seed mean, band = min-max over seeds 0-2. dagger: T=64 seed 0 complete (20000 epochs), '
    'seeds 1/2 still training, shown up to epoch 16000 / 14000 (mean over 18000-20000 = seed 0 only). '
    '(b) T=16 native models, L=4 (seeds 1,2), 6, 8 (seeds 0-2), 10 (seed 0), 16 (seeds 1,2); L=12*: no kappa=0.27 run, '
    'kappa=0.275 seed 0 shown dashed/open. (c) epochs to reach 1.1 x Var at epoch 20000 (log-linear interpolation; '
    'triangles: already reached at the first eval, epoch 2000 is an upper bound), geometric mean over complete seeds, '
    'bars = seed range; dashed: power-law fits (left: vs T per L; right: vs V at T=16, fit on kappa=0.27 points only).')
figdata['incomplete_runs'] = [dict(L=8, T=T, seed=s, last_epoch=e) for T, s, e in incomplete]
json.dump(figdata, open(os.path.join(OUT, 'train_curves_figdata.json'), 'w'), indent=1, default=float)
print('incomplete L8:', incomplete)
print('notes:', notes)
for k, v in figdata['panel_c'].items():
    if 'exponent' in v:
        print(k, round(v['exponent'], 3), round(v['exponent_err'], 3), 'chi2/dof', round(v['chi2'], 2), v['dof'])
print('vsV incl 0.275:', pV2, dpV2)
for L, s, k in pts:
    print('V', L, k, round(s['t_geo']), [(o['seed'], round(o['t']), o['censored'], round(o['final'], 3)) for o in s['per_seed']])
for L in [4, 6, 8, 16]:
    e = figdata['panel_c'].get(f'vsT_L{L}')
    if e:
        for T, p in e['per_T'].items():
            print('T', L, T, round(p['t_geo']), [(o['seed'], round(o['t']), o['censored'], round(o['final'], 3)) for o in p['per_seed']])
