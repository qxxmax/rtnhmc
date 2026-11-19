"""2026-11-17 u4 of figs_20261117u2/fig_scaling_u2.py: fig_scaling_var, fig_scaling_tau, fig_tau_fit (fig_scaling_accept is
made by fig_scaling_accept_u4.py).  Data, selection, fits and error bars unchanged; drawing and symbols only:
  - error bars of the per-site fit slopes (scaling_var d, tau_fit b) are read from figs_20261117u2/derived_scaling_u2.json
    (stat / pts of the u2 jackknife, the very numbers u2 plotted) instead of re-running the jackknife here;
    the derived-number dump of u2 is not repeated (no new numbers in u4).
  - scaling_var: (a) y label sigma_alpha^2/V (inferred from the acceptance through erfc^-1), (d) y label bar sigma_alpha^2
    (slope of the erfc fit); (c) legend entry for single-seed NHMC points (open blue).
  - scaling_tau: (b) pilots drawn with kappa-coloured edge, open, grey centre dot; pilots at the same V shifted by
    x 1.12^(i - 1.5), i = index of kappa in (0.20, 0.24, 0.27, 0.30) (gid 'u4xmul:<f>', undone in compare_u4.py);
    zero-accept arrows kappa-coloured; fit curves continued as thin dotted lines from the largest fitted V to V = 9216
    (gid 'u4deco', new artists); (c) transferred points (L = 24, 32) drawn smaller (ms 2.2) with a legend entry; x label
    1/bar alpha - 1/2.
  - tau_fit: (a) fit curves dashed only over the measured range (V <= 256), continued as thin dotted lines (same grid,
    gid 'u4join'); kappa = 0.26 drawn open with a dash-dot line (0.26 and 0.27 have nearly the same colour);
    (b) y label bar sigma_alpha^2 [1e-3].
u2 docstring: 2026-11-17 u2 of figs_20261117u/fig_scaling_20261117u.py: fig_scaling_accept, fig_scaling_var, fig_scaling_tau, fig_tau_fit.
Style from figs_20261117u/nhmc_style.py (imported, unchanged).  Data changes against u (all logged in prov/*.json and
PROVENANCE.md):
  1. native cells: E2_* first, then buf_* (u read E2_* only; adds L12 and L6 at kappa=0.27/0.30 -> 74 RT points).
  2. path-IMH: E2 _imhG, E2 _imh, buf _imhG, buf _imh in this order (u: E2 _imhG only; adds L4 and L6).
  3. pilot_bigL chains (n = 1000 per chain): open grey-edged markers, excluded from every fit.
  4. zero acceptance: 95% upper limit 3/n_post (n_post = post-burn transitions summed over chains), downward arrow.
  5. criteria_met = False: tau drawn as lower bound (upward arrow), excluded from tau fits.
  6. error bars: tau = delete-one-chain jackknife of the chain mean; acceptance = block jackknife (traces) / delete-one-chain;
     <30 accepts: Poisson.  Var(log w): standard error over training seeds.
  7. Var(log w) at kappa=0.27 (scaling_var c): seed rule of u2_common.kscan_points (seed 0 iqcb phi4_buffer GPU pipeline,
     seeds 1, 2 HIM kscan_cpu); fixes the iqcb regex of u (iqcb K-scan points were never read).
  8. HMC / Wolff reference tau read from figs/data/wh_tau_grid.json (u: hard-coded 23.3 / 40.0 / 65.4, same values).
Derived numbers with jackknife errors -> derived_scaling_u2.json."""
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
P = {f: U.Prov(f) for f in ('fig_scaling_var', 'fig_scaling_tau', 'fig_tau_fit')}


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


def val(r):
    if r is None:
        return None
    return OVR.get(r['path'], (r['acc'], r['tau']))


def compute(skip_tr=False, hmc=None, wolff=None):
    hmc = hmc or {L: v[0] for L, v in HMC.items()}
    wolff = wolff or {L: v[0] for L, v in WOLFF.items()}
    Q = {}
    dev = []
    for (L, k), cv in sorted(CV.items()):
        for m, s in (('rt', 0), ('rtp12', 12)):
            r = val(nat(L, k, m))
            if r and s in cv:
                p = erfc(np.sqrt(cv[s]) / 2)
                Q[f'accdev_L{L}_k{k}_{m}'] = r[0] - p
                dev.append(abs(r[0] - p))
    Q['accept_native_maxdev'] = max(dev)
    Q['accept_native_n'] = len(dev)
    for k in ('24', '27', '30'):
        cv = CV[(16, k)]
        for m, s in (('rt', 0), ('rtp12', 12)):
            for L in (24, 32, 48, 64, 96):
                r = tra(L, k, m)
                if r and not r['pilot']:
                    Q[f'tr_ratio_L{L}_k{k}_{m}'] = val(r)[0] / erfc(np.sqrt(cv[s] * L * L / 256) / 2)
    FIT = {}
    for m in ('rt', 'rtp12'):
        FIT[m] = {}
        for k in KAP:
            pts = [(L * L, val(nat(L, k, m))[0]) for L in LS if nat(L, k, m) and 0 < val(nat(L, k, m))[0] < 1]
            if len(pts) < 3:
                continue
            V = np.array([p[0] for p in pts], float)
            a = np.array([p[1] for p in pts])
            (c, s0), *_ = np.linalg.lstsq(np.c_[V, np.ones_like(V)], (2 * erfcinv(a)) ** 2, rcond=None)
            res = np.abs(erfc(np.sqrt(np.maximum(c * V + s0, 0)) / 2) - a)
            FIT[m][k] = (float(c), float(s0), float(res.max()), len(V), res)
            Q[f'sig2_{m}_k{k}'] = float(c)
            Q[f's0_{m}_k{k}'] = float(s0)
    Q['maxres_rt'] = max(v[2] for v in FIT['rt'].values())
    Q['median_of_kappa_maxres_rt'] = float(np.median([v[2] for v in FIT['rt'].values()]))
    Q['median_pointwise_res_rt'] = float(np.median(np.concatenate([v[4] for v in FIT['rt'].values()])))
    pk = max(FIT['rt'].items(), key=lambda t: t[1][0])
    Q['sig2_peak'] = pk[1][0]
    Q['sig2_peak_kappa'] = kv(pk[0])
    Q['s0_broken_mean_k275_30'] = float(np.mean([FIT['rt'][k][1] for k in ('275', '28', '29', '30')]))
    Q['efold_volume_k27'] = 4 / FIT['rt']['27'][0]
    Q['prt_over_rt_sig2_k27'] = FIT['rtp12']['27'][0] / FIT['rt']['27'][0]
    ratio = [FIT['rtp12'][k][0] / FIT['rt'][k][0] for k in KAP if k in FIT['rtp12'] and k in FIT['rt']]
    Q['prt_over_rt_sig2_min'] = float(min(ratio))
    Q['prt_over_rt_sig2_max'] = float(max(ratio))
    pts = [(L, k, *val(nat(L, k, 'rt'))) for L in LS for k in KAP if nat(L, k, 'rt') and nat(L, k, 'rt')['conv']]
    x = np.array([1 / p[2] - 0.5 for p in pts])
    y = np.array([p[3] for p in pts])
    C = float(np.exp(np.median(np.log(y / x))))
    Q['C'] = C
    Q['C_rms'] = float(np.sqrt(np.mean(np.log(y / (C * x)) ** 2)))
    Q['C_npts'] = len(pts)
    Q['C_lstsq_origin'] = float((x @ y) / (x @ x))
    for L in LS:
        for k in ('20', '24', '27', '30'):
            r = nat(L, k, 'rtp12')
            if r:
                a, t = val(r)
                Q[f'C_prt_L{L}_k{k}'] = t / (1 / a - 0.5)
    for L in (24, 32):
        for k in KAP:
            r = tra(L, k, 'rt')
            if r:
                a, t = val(r)
                Q[f'trtau_ratio_L{L}_k{k}'] = t / (C * (1 / a - 0.5))
                Q[f'trtau_pred_L{L}_k{k}'] = C * (1 / a - 0.5)
                Q[f'trtau_meas_L{L}_k{k}'] = t
    for m in ('rt', 'rtp12', 'imh'):
        t = [(L, val(nat(L, '27', m))[1]) for L in (8, 10, 12, 16) if nat(L, '27', m)]
        Q[f'exp_{m}_k27_L8-16fit'] = float(np.polyfit(np.log([a for a, _ in t]), np.log([b for _, b in t]), 1)[0])
        t2 = [(L, val(nat(L, '27', m))[1]) for L in (8, 16)]
        Q[f'exp_{m}_k27_L8-16twopoint'] = float(np.log(t2[1][1] / t2[0][1]) / np.log(2))
        t3 = [(L, val(nat(L, '27', m))[1]) for L in (8, 10, 16) if nat(L, '27', m)]
        Q[f'exp_{m}_k27_L8_10_16'] = float(np.polyfit(np.log([a for a, _ in t3]), np.log([b for _, b in t3]), 1)[0])
    Q['exp_hmc_k27'] = float(np.polyfit(np.log(list(hmc)), np.log(list(hmc.values())), 1)[0])
    Q['exp_wolff_k27'] = float(np.polyfit(np.log(list(wolff)), np.log(list(wolff.values())), 1)[0])
    Q['tau_rt_L16_k27'] = val(nat(16, '27', 'rt'))[1]
    Q['hmc_over_rt_L16_k27'] = hmc[16] / Q['tau_rt_L16_k27']
    c27, s27 = FIT['rt']['27'][:2]
    trt = lambda L: C * (1 / erfc(np.sqrt(c27 * L * L + s27) / 2) - 0.5)
    b, a = np.polyfit(np.log(list(hmc)), np.log(list(hmc.values())), 1)
    for tag, th in (('fit', lambda L: np.exp(a) * L ** b), ('anchor15', lambda L: hmc[16] * (L / 16) ** 1.5)):
        try:
            Q[f'crossover_L_{tag}'] = brentq(lambda L: np.log(trt(L)) - np.log(th(L)), 16, 200)
        except ValueError:
            Q[f'crossover_L_{tag}'] = float('nan')
    Q['tau_pred_L24_k26'] = C * (1 / erfc(np.sqrt(FIT['rt']['26'][0] * 576 + FIT['rt']['26'][1]) / 2) - 0.5)
    Q['tau_pred_L24_k27'] = trt(24)
    if not skip_tr:
        for k in ('20', '24', '27', '30'):
            for m in ('rt', 'rtp12'):
                p_ = [(L * L, val(tra(L, k, m))[0]) for L in (16, 24, 32, 48, 64, 96)
                      if tra(L, k, m) and not tra(L, k, m)['pilot'] and 0 < val(tra(L, k, m))[0] < 1]
                if len(p_) < 3:
                    continue
                V = np.array([p[0] for p in p_], float)
                A = np.array([p[1] for p in p_])
                g, g0 = least_squares(lambda q: np.log(erfc(np.sqrt(np.maximum(q[0] * V + q[1], 1e-12)) / 2)) - np.log(A),
                                      [0.01, 0.0], bounds=([1e-5, -0.5], [1.0, 1.0])).x
                pr = erfc(np.sqrt(np.maximum(g * V + g0, 0)) / 2)
                Q[f'trfit_g_k{k}_{m}'] = float(g)
                Q[f'trfit_g0_k{k}_{m}'] = float(g0)
                Q[f'trfit_n_k{k}_{m}'] = len(V)
                Q[f'trfit_Lmax_k{k}_{m}'] = int(np.sqrt(V.max()))
                Q[f'trfit_maxres_k{k}_{m}'] = float(np.max(np.abs(pr - A)))
                Q[f'trfit_maxrel_k{k}_{m}'] = float(np.max(np.abs(pr / A - 1)))
                Q[f'trfit_atbound_k{k}_{m}'] = bool(g0 <= -0.5 + 1e-6 or g0 >= 1 - 1e-6)
    return Q, FIT


Q0, FIT = compute()
C = Q0['C']


# ------------------------------------------------------------------ jackknife errors
def stat_errors():
    used = set()
    for L in LS:
        for k in KAP:
            for m in ('rt', 'rtp12', 'imh'):
                r = nat(L, k, m)
                if r:
                    used.add(r['path'])
    for L in (24, 32, 48, 64, 96):
        for k in KAP:
            for m in ('rt', 'rtp12'):
                r = tra(L, k, m)
                if r and not r['pilot']:
                    used.add(r['path'])
    var = defaultdict(float)
    for p in sorted(used):
        r = _C[p]
        info = U.cellinfo(p)
        ac = np.array(info['acc_chains'])
        tc = np.array(info['tau_absM_chains'])
        n = len(ac)
        reps = []
        skip = not ('T16_' in p or '_L16_' in p)
        for j in range(n):
            OVR[p] = (float(np.delete(ac, j).mean()), r['tau'] + float(np.delete(tc, j).mean() - tc.mean()))
            q, _ = compute(skip_tr=skip)
            reps.append(q)
        OVR.clear()
        for k, v in Q0.items():
            if isinstance(v, float):
                x = np.array([q.get(k, Q0[k]) if not skip or not k.startswith('trfit') else Q0[k] for q in reps], float)
                if np.all(np.isfinite(x)):
                    var[k] += (n - 1) / n * ((x - x.mean()) ** 2).sum()
    for alg, D in (('hmc', HMC), ('wolff', WOLFF)):
        for L in (8, 12, 16):
            t, s = D[L]
            hp = {LL: D[LL][0] for LL in D}
            hm = dict(hp)
            hp[L], hm[L] = t + s, t - s
            kw = dict(hmc=hp) if alg == 'hmc' else dict(wolff=hp)
            kw2 = dict(hmc=hm) if alg == 'hmc' else dict(wolff=hm)
            qp, _ = compute(skip_tr=True, **kw)
            qm, _ = compute(skip_tr=True, **kw2)
            for k in var.keys() | {kk for kk, vv in Q0.items() if isinstance(vv, float)}:
                if k in qp and isinstance(qp[k], float) and np.isfinite(qp[k]) and np.isfinite(qm[k]):
                    var[k] += ((qp[k] - qm[k]) / 2) ** 2
    return {k: float(np.sqrt(v)) for k, v in var.items()}


def jk(reps):
    t = np.asarray(reps, float)
    t = t[np.isfinite(t)]
    n = len(t)
    return float(np.sqrt((n - 1) / n * ((t - t.mean()) ** 2).sum())) if n > 1 else float('nan')


def point_errors():
    E = {}
    pts = [(L, k) for L in LS for k in KAP if nat(L, k, 'rt') and nat(L, k, 'rt')['conv']]
    reps = []
    for i in range(len(pts)):
        keep = [p for j, p in enumerate(pts) if j != i]
        x = np.array([1 / nat(L, k, 'rt')['acc'] - 0.5 for L, k in keep])
        y = np.array([nat(L, k, 'rt')['tau'] for L, k in keep])
        reps.append(np.exp(np.median(np.log(y / x))))
    E['C'] = jk(reps)
    c27s = None
    for m in ('rt', 'rtp12'):
        for k in KAP:
            p_ = [(L * L, nat(L, k, m)['acc']) for L in LS if nat(L, k, m) and 0 < nat(L, k, m)['acc'] < 1]
            if len(p_) < 4:
                continue
            rc, rs = [], []
            for i in range(len(p_)):
                kp = [p for j, p in enumerate(p_) if j != i]
                V = np.array([p[0] for p in kp], float)
                a = np.array([p[1] for p in kp])
                (c, s0), *_ = np.linalg.lstsq(np.c_[V, np.ones_like(V)], (2 * erfcinv(a)) ** 2, rcond=None)
                rc.append(c)
                rs.append(s0)
            E[f'sig2_{m}_k{k}'] = jk(rc)
            E[f's0_{m}_k{k}'] = jk(rs)
            if m == 'rt' and k == '27':
                E['efold_volume_k27'] = jk([4 / c for c in rc])
                c27s = list(zip(rc, rs))
            if m == 'rt' and k == '26':
                E['tau_pred_L24_k26'] = jk([Q0['C'] * (1 / erfc(np.sqrt(c * 576 + s) / 2) - 0.5) for c, s in zip(rc, rs)])
    E['tau_pred_L24_k27'] = jk([Q0['C'] * (1 / erfc(np.sqrt(c * 576 + s) / 2) - 0.5) for c, s in c27s])
    pk = [k for k in KAP if f'sig2_rt_k{k}' in E]
    E['sig2_peak'] = E.get(f'sig2_rt_k{max(pk, key=lambda k: Q0[f"sig2_rt_k{k}"])}')
    for m in ('rt', 'rtp12', 'imh'):
        t = [(L, nat(L, '27', m)['tau']) for L in (8, 10, 12, 16) if nat(L, '27', m)]
        E[f'exp_{m}_k27_L8-16fit'] = jk([np.polyfit(np.log([a for a, _ in tt]), np.log([b for _, b in tt]), 1)[0]
                                         for tt in [t[:i] + t[i + 1:] for i in range(len(t))]])
    for alg, D in (('hmc', HMC), ('wolff', WOLFF)):
        t = [(L, D[L][0]) for L in (8, 12, 16)]
        E[f'exp_{alg}_k27'] = jk([np.polyfit(np.log([a for a, _ in tt]), np.log([b for _, b in tt]), 1)[0]
                                  for tt in [t[:i] + t[i + 1:] for i in range(3)]])
    hmc = {L: HMC[L][0] for L in HMC}
    b, a = np.polyfit(np.log(list(hmc)), np.log(list(hmc.values())), 1)

    def cross(C_, c27, s27, a, b):
        trt = lambda L: C_ * (1 / erfc(np.sqrt(c27 * L * L + s27) / 2) - 0.5)
        try:
            return brentq(lambda L: np.log(trt(L)) - (a + b * np.log(L)), 16, 200)
        except ValueError:
            return np.nan
    v1 = jk([cross(Q0['C'], c, s, a, b) for c, s in c27s])
    v2 = jk([cross(C_, Q0['sig2_rt_k27'], Q0['s0_rt_k27'], a, b) for C_ in reps])
    HL = list(hmc.items())
    v3 = jk([cross(Q0['C'], Q0['sig2_rt_k27'], Q0['s0_rt_k27'], *np.polyfit(np.log([l for l, _ in hh]), np.log([t for _, t in hh]), 1)[::-1])
             for hh in [HL[:i] + HL[i + 1:] for i in range(3)]])
    E['crossover_L_fit'] = float(np.sqrt(v1 ** 2 + v2 ** 2 + v3 ** 2))
    for k in ('20', '24', '27', '30'):
        for m in ('rt', 'rtp12'):
            p_ = [(L * L, tra(L, k, m)['acc']) for L in (16, 24, 32, 48, 64, 96)
                  if tra(L, k, m) and not tra(L, k, m)['pilot'] and 0 < tra(L, k, m)['acc'] < 1]
            if len(p_) < 4:
                continue
            rg = []
            for i in range(len(p_)):
                kp = [p for j, p in enumerate(p_) if j != i]
                V = np.array([p[0] for p in kp], float)
                A = np.array([p[1] for p in kp])
                rg.append(least_squares(lambda q: np.log(erfc(np.sqrt(np.maximum(q[0] * V + q[1], 1e-12)) / 2)) - np.log(A),
                                        [0.01, 0.0], bounds=([1e-5, -0.5], [1.0, 1.0])).x)
            rg = np.array(rg)
            E[f'trfit_g_k{k}_{m}'] = jk(rg[:, 0])
            E[f'trfit_g0_k{k}_{m}'] = jk(rg[:, 1])
    return E


_DER = json.load(open(U.U2 + '/derived_scaling_u2.json'))                     # u4: cached u2 jackknife (same numbers)
ST = {k: v['stat'] for k, v in _DER.items() if k.startswith('sig2_') and isinstance(v, dict) and v.get('stat') is not None}
PE = {k: v['pts'] for k, v in _DER.items() if k.startswith('sig2_') and isinstance(v, dict) and v.get('pts') is not None}
for _k in ST:
    assert abs(_DER[_k]['value'] - Q0[_k]) < 1e-12, _k                         # same fit as u2


def num(key):
    v = Q0[key]
    s, p = ST.get(key), PE.get(key)
    tot = float(np.sqrt((s or 0) ** 2 + (p or 0) ** 2)) if (s is not None or p is not None) else None
    return dict(value=v, stat=s, pts=p, total=tot)


def errbar(ax, x, y, yerr=None, xerr=None, **kw):
    kw.setdefault('elinewidth', 0.6)
    kw.setdefault('capsize', 0)
    return ax.errorbar(x, y, yerr=yerr, xerr=xerr, **kw)


def arrow_down(ax, x, y, color, ms=6):
    return ax.plot([x], [y], marker=r'$\downarrow$', ms=ms, color=color, ls='')[0]


PILOT_DOT = '0.45'                                       # u4: grey centre dot marks 1000-step pilots
KI = {'20': 0, '24': 1, '27': 2, '30': 3}


def set_gid(cont, gid):
    """gid on the data line and the error-bar LineCollections of an errorbar container (compare_u4 undoes offsets)."""
    cont.lines[0].set_gid(gid)
    for c in cont.lines[2]:
        c.set_gid(gid)


def arrow_up(ax, x, y, color, ms=6):
    ax.plot([x], [y * 1.5], marker=r'$\uparrow$', ms=ms, color=color, ls='')


# ================================================================== fig_scaling_var
pr = P['fig_scaling_var']
fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.6))
a = ax[0, 0]
for k in KAP:
    Ls, s2, e2 = [], [], []
    for L in LS:
        r = nat(L, k, 'rt')
        if r and 0 < r['acc'] < 1:
            x = erfcinv(r['acc'])
            sig = 2 * x
            Ls.append(L)
            s2.append(sig ** 2 / L ** 2)
            e2.append(abs(2 * sig * 2 * np.sqrt(np.pi) / 2 * np.exp(x * x)) * (r['acc_err'] or 0) / L ** 2)
            pr.add('a', r['path'], U_RULE_NATIVE, 'new in u2 (buf_*)' if u_native_file(L, k, 'rt') is None else '')
    if len(Ls) >= 3:
        a.plot(Ls, s2, '-', color=kcol(k), lw=0.9)
        errbar(a, Ls, s2, yerr=e2, fmt=RTM, color=kcol(k), ms=3)
a.set_xlabel('$L$')
a.set_ylabel(r'$\sigma_\alpha^2/V$ (RT)')                                   # u4: inferred from acceptance
NS.kappa_colorbar(fig, a, pad=0.02, fraction=0.05)
b = ax[0, 1]
LSTY = {16: ('-', 'o'), 12: ('-.', '^'), 8: ('--', 's')}
for (L, k), cv in sorted(CV.items()):
    d = sorted(cv)
    ls_, mk_ = LSTY[L]
    b.plot([16 - s for s in d], [cv[s] / L ** 2 for s in d], ls=ls_, marker=mk_, color=kcol(k), ms=3, lw=0.9)
    pr.add('b', CVF[(L, k)], 'wtail2_20261114 (HIM 93289), all stages', 'no per-group values stored -> no error bars')
b.set_xlabel('return depth $d$')
b.set_ylabel(r'$\sigma_s^2/V$')
hb = [plt.Line2D([], [], color='0.3', ls=LSTY[L][0], marker=LSTY[L][1], ms=3, lw=0.9, label=f'$L={L}$') for L in (8, 12, 16)]
hb += [NS.khandle(kv(k)) for k in sorted({k for _, k in CV}, key=kv)]
b.legend(handles=hb, ncol=2, loc='upper left')
c = ax[1, 0]
Lmk = {4: 'v', 6: '<', 8: 'o', 10: '>', 12: 's', 14: 'p', 16: 'D'}
CC = {'NHMC': NS.COL['NHMC'], 'protocol': '#E69F00', 'NE-HMC': NS.COL['HMC']}
KS = U.kscan_points()
for (L, T), v in sorted(KS['cells'].items()):
    vals = [x['var'] for x in v]
    m_ = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
    errbar(c, T / L ** 2, m_, yerr=se, fmt=Lmk[L], color=CC['NHMC'], ms=4, mfc=CC['NHMC'] if len(vals) > 1 else 'white')
    for x in v:
        pr.add('c', x['path'], KS['rule'], f'L={L} T={T} seed={x["seed"]} ({x["src"]}) Var={x["var"]:.4f}' + (' NEW in u2' if x['new'] else ''))
for x in KS['excluded']:
    pr.add('c (excluded)', x['path'], KS['rule'], x['why'])
for f in sorted(glob.glob(U.H17 + '/*/revision_runs/proto_20261114/*/final_eval.json')):
    m = re.search(r'L(\d+)_k027_K(\d+)_s0', f)
    if m:
        d = [x for x in json.load(open(f)) if x['n'] == 262144]
        if d:
            c.plot(int(m[2]) / int(m[1]) ** 2, d[0]['var_W'], Lmk[int(m[1])], color=CC['protocol'], ms=4, mfc='white')
            pr.add('c protocol', f, 'proto_20261114 final_eval n=262144 (unchanged)')
best, bestf = {}, {}
for f in sorted(glob.glob(U.H17 + '/*/revision_runs/nehmc_20261113/L*_k027_K*.json')):
    m = re.search(r'L(\d+)_k027_K(\d+)', f)
    for d in json.load(open(f)) or []:
        v = d.get('var_W')
        if v is not None and np.isfinite(v):
            key = (int(m[1]), int(m[2]))
            if v < best.get(key, 1e9):
                best[key], bestf[key] = v, f
for (L, K), v in best.items():
    c.plot(K / L ** 2, v, Lmk[L], color=CC['NE-HMC'], ms=4, mfc='none')
    pr.add('c NE-HMC', bestf[(L, K)], 'nehmc_20261113: minimum Var over the step sizes in the file and over jobs 93128/93131 (unchanged)')
c.set_xscale('log')
c.set_yscale('log')
c.set_xlabel('$T/V$')
c.set_ylabel(r'Var$(\log w)$')
NS.tag(c, r'$\kappa=0.27$', loc='upper right')
c.set_ylim(0.015, 120)                                                          # u4: headroom for the 2-column legend
h1 = [plt.Line2D([], [], color=CC[k], ls='', marker='o', ms=4, mfc=f, label=l) for k, f, l in
      (('NHMC', CC['NHMC'], 'NHMC'), ('NHMC', 'white', 'NHMC, 1 seed'), ('protocol', 'white', 'protocol only'),
       ('NE-HMC', 'none', 'NE-HMC'))]                                                  # u4: single-seed entry
h2 = [plt.Line2D([], [], color='0.4', ls='', marker=Lmk[L], ms=4, label=f'$L={L}$') for L in sorted({L for L, _ in KS['cells']} | {L for L, _ in best})]
leg1 = c.legend(handles=h1, loc='upper right', bbox_to_anchor=(1.0, 0.93), ncol=2, columnspacing=0.8, handletextpad=0.2)
c.add_artist(leg1)
c.legend(handles=h2, loc='lower left', ncol=4, columnspacing=0.8, handletextpad=0.1, borderaxespad=0.2)
dd = ax[1, 1]
for m, ls_, mk, fc, lab in (('rt', '-', RTM, None, 'RT'), ('rtp12', '--', PRTM, 'white', 'pRT ($s=12$)')):
    ks = [k for k in KAP if k in FIT[m]]
    ys = [FIT[m][k][0] for k in ks]
    es = [np.hypot(ST.get(f'sig2_{m}_k{k}', 0) or 0, PE.get(f'sig2_{m}_k{k}', 0) or 0) for k in ks]
    dd.plot([kv(k) for k in ks], ys, ls_, color=NS.COL['RT' if m == 'rt' else 'pRT'], lw=1.0)
    errbar(dd, [kv(k) for k in ks], ys, yerr=es, fmt=mk, color=NS.COL['RT' if m == 'rt' else 'pRT'], ms=3,
           mfc=fc or NS.COL['RT'], label=lab)
dd.set_xlabel(r'$\kappa$')
dd.set_ylabel(r'$\bar\sigma_\alpha^2$')                                          # u4: erfc-fit slope
dd.legend(loc='upper left')
fig.tight_layout(h_pad=1.2, w_pad=1.5)
for a_, l in zip(ax.flat, 'abcd'):
    NS.panel_letter(a_, l)
NS.save(fig, U.pdf('fig_scaling_var'))
pr.note(KS['rule'])
pr.note('u4: labels (a) sigma_alpha^2/V, (d) bar sigma_alpha^2; legend entry NHMC 1 seed; error bars of (d) from derived_scaling_u2.json')
pr.dump(extra=dict(kscan_cells={f'{L}_{T}': [(x['seed'], x['src'], round(x['var'], 4)) for x in v] for (L, T), v in sorted(KS['cells'].items())}))

# ================================================================== fig_scaling_tau
pr = P['fig_scaling_tau']
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.45))
a = ax[0]
for m, mk, ls_ in (('rt', RTM, '-'), ('rtp12', PRTM, '--'), ('imh', IMHM, ':')):
    for k in ('24', '27', '30'):
        rs = [(L, nat(L, k, m)) for L in LS if nat(L, k, m)]
        if not rs:
            continue
        V = [L * L for L, _ in rs]
        T = [r['tau'] for _, r in rs]
        a.plot(V, T, ls=ls_, color=kcol(k), lw=0.8)
        errbar(a, V, T, yerr=[r['tau_err'] or 0 for _, r in rs], fmt=mk, color=kcol(k), ms=3,
               mfc='white' if m == 'rtp12' else kcol(k))
        for L, r in rs:
            pr.add('a', r['path'], U_RULE_NATIVE, 'new in u2' if u_native_file(L, k, m) is None else '')
hv = [(L * L, HMC[L][0], HMC[L][1]) for L in sorted(HMC)]
a.plot([x[0] for x in hv], [x[1] for x in hv], ls=NS.REF_LS, color=NS.COL['HMC'], lw=0.8)
errbar(a, [x[0] for x in hv], [x[1] for x in hv], yerr=[x[2] for x in hv], fmt=HMCM, color=NS.COL['HMC'], mfc='white', ms=3.5)
pr.add('a', REV + '/figs/data/wh_tau_grid.json', 'HMC kappa=0.27, L=8,12,16, tau_absm/2; errors from jackknife_20261117/fits_jk_buf.json ref_tau')
a.set_yscale('log')
a.set_xlabel('$V$')
a.set_ylabel(r'$\tau_{\rm int}(|m|)$')
b = ax[1]
for k in ('20', '24', '27', '30'):
    for m, ls, mk in (('rt', '-', RTM), ('rtp12', '--', PRTM)):
        rs = [(L, tra(L, k, m)) for L in (16, 24, 32, 48, 64, 96) if tra(L, k, m)]
        for L, r in rs:
            pr.add('b', r['path'], 'transfer T16 (L=16 native); pilots excluded from fit',
                   ('pilot n=1000, n_acc=%s' % r['nacc']) if r['pilot'] else ('n_acc=%s' % r['nacc']))
            if r['pilot']:                                     # u4: kappa edge, open, grey centre dot, x offset by kappa
                f_ = 1.12 ** (KI[k] - 1.5)
                gid = f'u4xmul:{f_!r}'
                if r['acc'] <= 0:
                    arrow_down(b, L * L * f_, r['ul95'], kcol(k)).set_gid(gid)
                else:
                    set_gid(errbar(b, L * L * f_, r['acc'], yerr=r['acc_err'], fmt=mk, color=kcol(k), mfc='white',
                                   mec=kcol(k), ms=3.6, mew=1.0), gid)
                    b.plot([L * L * f_], [r['acc']], 'o', ms=1.7, color=PILOT_DOT, mew=0, gid='u4deco', zorder=4)
                continue
            ec = kcol(k)
            fc = 'white' if m == 'rtp12' else kcol(k)
            if r['acc'] <= 0:
                arrow_down(b, L * L, r['ul95'], ec)
            else:
                errbar(b, L * L, r['acc'], yerr=r['acc_err'], fmt=mk, color=ec, mfc=fc, mec=ec, ms=3)
        if f'trfit_g_k{k}_{m}' in Q0:
            g, g0 = Q0[f'trfit_g_k{k}_{m}'], Q0[f'trfit_g0_k{k}_{m}']
            Vf = [L * L for L, r in rs if not r['pilot'] and 0 < r['acc'] < 1]
            vv = np.geomspace(min(Vf), max(Vf), 200)
            b.plot(vv, erfc(np.sqrt(np.maximum(g * vv + g0, 1e-12)) / 2), ls, color=kcol(k), lw=0.8)
            vx = np.geomspace(max(Vf), 96 * 96, 60)                         # u4: extrapolation to L = 96, thin dotted
            b.plot(vx, erfc(np.sqrt(np.maximum(g * vx + g0, 1e-12)) / 2), ':', color=kcol(k), lw=0.7, alpha=0.6, gid='u4deco')
b.set_xscale('log')
b.set_yscale('log')
b.set_ylim(1e-4, 1)
b.set_xlabel('$V$')
b.set_ylabel('acceptance (transfer)')
c = ax[2]
for m, mk in (('rt', RTM), ('rtp12', PRTM)):
    for k in ('20', '24', '27', '30'):
        rs = [(nat(L, k, m), 3) for L in LS] + [(tra(L, k, m), 2.2) for L in (24, 32)]   # u4: transferred smaller
        for r, ms_ in rs:
            if not r:
                continue
            x = 1 / r['acc'] - 0.5
            xe = (r['acc_err'] or 0) / r['acc'] ** 2
            fc = kcol(k) if m == 'rt' else 'white'
            if r['conv']:
                errbar(c, x, r['tau'], yerr=r['tau_err'] or 0, xerr=xe, fmt=mk, color=kcol(k), mfc=fc, ms=ms_)
            else:
                c.plot(x, r['tau'], mk, color=kcol(k), mfc=fc, ms=ms_)
                arrow_up(c, x, r['tau'], kcol(k))
            pr.add('c', r['path'], 'native L4-16 + T16 L24/32; criteria_met=False -> lower bound (arrow), not fitted',
                   '' if r['conv'] else 'NOT converged: lower bound')
xx = np.logspace(-0.5, 3, 50)
c.plot(xx, C * xx, color=NS.FIT_COLOR, ls=NS.FIT_LS, lw=0.8)
c.set_xscale('log')
c.set_yscale('log')
c.set_xlabel(r'$1/\bar\alpha-1/2$')
c.set_ylabel(r'$\tau_{\rm int}(|m|)$')
h = [plt.Line2D([], [], color='0.2', ls='-', marker=RTM, ms=3.5, label='RT'),
     plt.Line2D([], [], color='0.2', ls='--', marker=PRTM, mfc='white', ms=3.5, label='pRT'),
     plt.Line2D([], [], color='0.2', ls=':', marker=IMHM, ms=3.5, label='path-IMH'),
     plt.Line2D([], [], color=NS.COL['HMC'], ls=NS.REF_LS, marker=HMCM, mfc='white', ms=3.5, label='HMC'),
     plt.Line2D([], [], color=NS.FIT_COLOR, ls=NS.FIT_LS, label='fit'),
     (plt.Line2D([], [], color='0.3', ls='', marker='o', mfc='white', ms=3.5),
      plt.Line2D([], [], color=PILOT_DOT, ls='', marker='o', ms=1.7, mew=0)),
     plt.Line2D([], [], color='0.3', ls='', marker=r'$\uparrow\downarrow$', ms=7, label='bound'),
     plt.Line2D([], [], color='0.2', ls='', marker=RTM, ms=2.2, label='transfer (c)')]
h += [NS.khandle(kv(k)) for k in ('20', '24', '27', '30')]
LABS = [x.get_label() if not isinstance(x, tuple) else '1000-step pilot' for x in h]
fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=1.5)
NS.fig_legend(fig, h, LABS, ncol=6, columnspacing=1.0, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})
for a_, l in zip(ax, 'abc'):
    NS.panel_letter(a_, l)
NS.save(fig, U.pdf('fig_scaling_tau'))
pr.note('u4: (b) pilots kappa-coloured, grey centre dot, x offset 1.12^(i-1.5); dotted fit extrapolation to V=9216; (c) transfers ms 2.2; x label 1/bar alpha-1/2')
pr.dump()

# ================================================================== fig_tau_fit (appendix)
pr = P['fig_tau_fit']
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.35))
sel = ['20', '24', '26', '27', '28', '30']
VV = np.linspace(10, 32 * 32, 300)
for k in sel:
    col = kcol(k)
    c_, s0_ = FIT['rt'][k][:2]
    rs = sorted(((L * L, nat(L, k, 'rt')) for L in LS if nat(L, k, 'rt')), key=lambda t: t[0])
    op = k == '26'                                       # u4: 0.26 open + dash-dot (colour close to 0.27)
    errbar(axs[0], [q[0] for q in rs], [q[1]['tau'] for q in rs], yerr=[q[1]['tau_err'] or 0 for q in rs], fmt=RTM, ms=3,
           color=col, mfc='white' if op else col, label=f'$\\kappa=0.{k}$')
    for _, r in rs:
        pr.add('a', r['path'], U_RULE_NATIVE)
    al = erfc(np.sqrt(np.maximum(c_ * VV + s0_, 0)) / 2)
    vmax = max(q[0] for q in rs)                         # u4: dashed over the data, thin dotted extrapolation (same grid)
    inr = VV <= vmax
    axs[0].plot(VV[inr], C * (1 / al[inr] - 0.5), ls='-.' if op else NS.FIT_LS, color=col, lw=0.8)
    j0 = np.flatnonzero(inr)[-1]
    axs[0].plot(VV[j0:], C * (1 / al[j0:] - 0.5), ls=':', color=col, lw=0.7, gid='u4join')
axs[0].plot([x[0] for x in hv], [x[1] for x in hv], ls=NS.REF_LS, color=NS.COL['HMC'], lw=0.8)
errbar(axs[0], [x[0] for x in hv], [x[1] for x in hv], yerr=[x[2] for x in hv], fmt=HMCM, color=NS.COL['HMC'], mfc='white', ms=3.5,
       label=r'HMC, $\kappa=0.27$')
pr.add('a', REV + '/figs/data/wh_tau_grid.json', 'HMC kappa=0.27 tau_absm/2')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylim(0.8, 400)
axs[0].set_xlabel('$V$')
axs[0].set_ylabel(r'$\tau_{\rm int}(|m|)$')
axs[0].legend(ncol=2, loc='upper left', columnspacing=0.8)
ks = [k for k in KAP if k in FIT['rt']]
ys = [FIT['rt'][k][0] * 1e3 for k in ks]
es = [np.hypot(ST.get(f'sig2_rt_k{k}', 0) or 0, PE.get(f'sig2_rt_k{k}', 0) or 0) * 1e3 for k in ks]
axs[1].plot([kv(k) for k in ks], ys, '-', color=NS.COL['RT'], lw=0.9)
errbar(axs[1], [kv(k) for k in ks], ys, yerr=es, fmt=RTM, ms=3, color=NS.COL['RT'])
axs[1].set_xlabel(r'$\kappa$')
axs[1].set_ylabel(r'$\bar\sigma_\alpha^2\ [10^{-3}]$')                       # u4: erfc-fit slope
fig.tight_layout(w_pad=2.0)
for a_, l in zip(axs, 'ab'):
    NS.panel_letter(a_, l)
NS.save(fig, U.pdf('fig_tau_fit'))
pr.note('u4: (a) fit dashed over data range, dotted beyond; kappa=0.26 open, dash-dot; (b) label bar sigma_alpha^2; errors from derived_scaling_u2.json')
pr.note(f'C = {C:.4f} from {Q0["C_npts"]} native RT points (u: 70)')
pr.dump()

