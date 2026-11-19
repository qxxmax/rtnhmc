"""Jackknife errors of the scaling-law numbers of Sec. V / App. D (2026-11-17). Reads cells_*.json (jk_cells.py) and the
wtail2 / Wolff-HMC files; reproduces figs_20261116/fig_scaling_20261116.py first (variant 'paper'), then the variant with
buf_* fallback for native cells (variant 'buf', REVIEW_code_20261117 B1). Writes fits_jk.json.

Two error components per quantity, quoted separately and combined in quadrature:
  stat  per-cell statistical error propagated by jackknife: for every input cell c and every deleted chain j the quantity
        is recomputed with only cell c replaced by its delete-chain-j value; var = sum_c (n_c-1)/n_c sum_j (Q_cj - Q_c.)^2
        (cells are independent).  acc replica = mean of the other chains; tau replica = paper median shifted by the
        delete-one change of the chain mean ('smooth', primary) -- the delete-one median of 4 chains is erratic; the raw
        median variant is stored as *_stat_med.
  pts   delete-one-data-point jackknife of the fit (points = cells for C, L values for per-kappa fits and exponents)."""
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import erfc, erfcinv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jk_lib import OUT, KAP, LS, HIM, REV, kv, jk_err, sokal_g, bm_g, dump  # noqa: E402

CELLS = {}
for g in ('E2', 'buf', 'T16'):
    CELLS.update(json.load(open(f'{OUT}/cells_{g}.json')))


import functools


@functools.lru_cache(maxsize=None)
def entry(name):
    r = CELLS.get(name)
    if not r or 'acc' not in r or 'tau_absM' not in r:
        return None
    ac = np.array(r['acc_chains'])
    tc = np.array(r['tau_absM_chains'])
    n = len(ac)
    acc_rep = np.array([np.delete(ac, j).mean() for j in range(n)])
    tm = np.array([np.delete(tc, j).mean() for j in range(n)])
    tau_rep_s = r['tau_absM'] + (tm - tc.mean())
    tau_rep_m = np.array([np.median(np.delete(tc, j)) for j in range(n)])
    return dict(acc=r['acc'], tau=r['tau_absM'], acc_rep=acc_rep, tau_rep=tau_rep_s, tau_rep_med=tau_rep_m,
                conv=r['converged'], acc_blk=r.get('acc_jk_block'), n_acc=r.get('n_acc'))


class Data:
    """(L, k, m) -> (acc, tau); variant 'paper' = fig_scaling names, 'buf' = + buf_* fallback; override one cell."""

    def __init__(self, variant):
        self.v, self.over, self.used = variant, {}, set()

    def name(self, L, k, m, kind):
        if kind == 'transfer' and L != 16:
            return f'T16_L{L}_k0{k}_{m}'
        n = f'E2_L{L}_k0{k}_{m}'
        if self.v == 'buf' and n not in CELLS:
            b = f'buf_L{L}_k0{k}_{m}'
            if b in CELLS:
                return b
        return n

    def get(self, L, k, m, kind='native'):
        n = self.name(L, k, m, kind)
        e = entry(n)
        if e is None:
            return None
        self.used.add(n)
        return self.over.get(n, (e['acc'], e['tau']))


native = lambda D, L, k, m: D.get(L, k, m, 'native')
transfer = lambda D, L, k, m: D.get(L, k, m, 'transfer')

CV = {}
import glob, re  # noqa: E401,E402
for f in glob.glob(HIM + '/*/revision_runs/wtail2_20261114/*.json'):
    L, k = re.match(r'L(\d+)_k0(\d+)', os.path.basename(f)).groups()
    CV[(int(L), k)] = {int(s): v['cond_var'] for s, v in json.load(open(f))['stages'].items()}

# HMC / Wolff tau(|m|) at kappa=0.27 (g/2) and their errors: segment jackknife on stored 1e6 traces where present,
# else Madras-Sokal with window ~ 5 g.
G = json.load(open(REV + '/figs/data/wh_tau_grid.json'))
WHT = np.load(REV + '/figs/data/wh_absm_traces.npz')


def ref_tau(alg, L, kmil=270):
    v = [x for x in G.values() if x['alg'] == alg and x['L'] == L and round(x['kappa'] * 1000) == kmil][0]
    tau = v['tau_absm'] / 2
    key = f'L{L}_k0{kmil}_{alg}'
    if key in WHT.files:
        x = WHT[key].astype(float)
        S = 16
        ls = len(x) // S
        seg = np.array([max(sokal_g(x[i * ls:(i + 1) * ls])[0], bm_g(x[i * ls:(i + 1) * ls]) or 0) / 2 for i in range(S)])
        reps = np.array([np.delete(seg, i).mean() for i in range(S)])
        return tau, jk_err(reps) * np.sqrt(1.0), 'segment jackknife (16 x 62500)'
    g = 2 * tau
    return tau, tau * np.sqrt(2 * (2 * 5 * g + 1) / v['n']), 'Madras-Sokal (W = 5g)'


REF = {(a, L): ref_tau(a, L) for a in ('hmc', 'wolff') for L in (4, 8, 12, 16)}


def compute(D, hmc=None, skip_tr=False, wolff=None):
    """all quoted numbers as a flat dict (mirrors fig_scaling_20261116.py)."""
    hmc = hmc or {L: REF[('hmc', L)][0] for L in (8, 12, 16)}
    wolff = wolff or {L: REF[('wolff', L)][0] for L in (8, 12, 16)}
    Q = {}
    dev = []
    for (L, k), cv in sorted(CV.items()):
        for m, s in (('rt', 0), ('rtp12', 12)):
            r = native(D, L, k, m)
            if r and s in cv:
                p = erfc(np.sqrt(cv[s]) / 2)
                Q[f'accdev_L{L}_k{k}_{m}'] = r[0] - p
                dev.append(abs(p - r[0]))
    Q['accept_native_maxdev'] = max(dev)
    for k in ('24', '27', '30'):
        cv = CV.get((16, k))
        for m, s in (('rt', 0), ('rtp12', 12)):
            for L in (24, 32, 48, 64, 96):
                r = transfer(D, L, k, m)
                if r:
                    Q[f'tr_ratio_L{L}_k{k}_{m}'] = r[0] / erfc(np.sqrt(cv[s] * L * L / 256) / 2)
    FIT = {}
    for m in ('rt', 'rtp12'):
        FIT[m] = {}
        for k in KAP:
            pts = [(L * L, native(D, L, k, m)[0]) for L in LS if native(D, L, k, m) and 0 < native(D, L, k, m)[0] < 1]
            if len(pts) < 3:
                continue
            V = np.array([p[0] for p in pts], float); a = np.array([p[1] for p in pts])
            (c, s0), *_ = np.linalg.lstsq(np.c_[V, np.ones_like(V)], (2 * erfcinv(a)) ** 2, rcond=None)
            FIT[m][k] = (c, s0, float(np.max(np.abs(erfc(np.sqrt(np.maximum(c * V + s0, 0)) / 2) - a))))
            Q[f'sig2_{m}_k{k}'] = c
            Q[f's0_{m}_k{k}'] = s0
    Q['maxres_rt'] = max(v[2] for v in FIT['rt'].values())
    Q['medres_rt_of_kmax'] = float(np.median([v[2] for v in FIT['rt'].values()]))
    pk = max(FIT['rt'].items(), key=lambda t: t[1][0])
    Q['sig2_peak'] = pk[1][0]
    Q['sig2_peak_kappa'] = kv(pk[0])
    Q['s0_broken_mean_k275_30'] = float(np.mean([FIT['rt'][k][1] for k in ('275', '28', '29', '30')]))
    Q['efold_volume_k27'] = 4 / FIT['rt']['27'][0]
    Q['prt_over_rt_sig2_k27'] = FIT['rtp12']['27'][0] / FIT['rt']['27'][0]
    pts = [(L, k, *native(D, L, k, 'rt')) for L in LS for k in KAP if native(D, L, k, 'rt')]
    x = np.array([1 / p[2] - 0.5 for p in pts]); y = np.array([p[3] for p in pts])
    C = float(np.exp(np.median(np.log(y / x))))
    Q['C'] = C
    Q['C_rms'] = float(np.sqrt(np.mean(np.log(y / (C * x)) ** 2)))
    Q['C_npts'] = len(pts)
    for L in LS:
        for k in ('20', '24', '27', '30'):
            r = native(D, L, k, 'rtp12')
            if r:
                Q[f'C_prt_L{L}_k{k}'] = r[1] / (1 / r[0] - 0.5)
    for L in (24, 32):
        for k in KAP:
            r = transfer(D, L, k, 'rt')
            if r:
                Q[f'trtau_meas_L{L}_k{k}'] = r[1]
                Q[f'trtau_pred_L{L}_k{k}'] = C * (1 / r[0] - 0.5)
                Q[f'trtau_ratio_L{L}_k{k}'] = r[1] / (C * (1 / r[0] - 0.5))
    for m in ('rt', 'rtp12', 'imhG'):
        t = [(L, native(D, L, '27', m)[1]) for L in (8, 10, 12, 16) if native(D, L, '27', m)]
        Q[f'exp_{m}_k27'] = float(np.polyfit(np.log([a for a, _ in t]), np.log([b for _, b in t]), 1)[0])
        Q[f'exp_{m}_k27_Ls'] = [a for a, _ in t]
    Q['exp_hmc_k27'] = float(np.polyfit(np.log(list(hmc)), np.log(list(hmc.values())), 1)[0])
    Q['exp_wolff_k27'] = float(np.polyfit(np.log(list(wolff)), np.log(list(wolff.values())), 1)[0])
    Q['tau_rt_L16_k27'] = native(D, 16, '27', 'rt')[1]
    Q['hmc_over_rt_L16_k27'] = hmc[16] / Q['tau_rt_L16_k27']
    # crossover: RT extrapolated with the kappa=0.27 fit and C; HMC = A L^b through L = 8, 12, 16 ('fit') or 65.4 (L/16)^1.5
    c27, s27 = FIT['rt']['27'][:2]
    trt = lambda L: C * (1 / erfc(np.sqrt(c27 * L * L + s27) / 2) - 0.5)
    b, a = np.polyfit(np.log(list(hmc)), np.log(list(hmc.values())), 1)
    for tag, th in (('fit', lambda L: np.exp(a) * L ** b), ('anchor15', lambda L: hmc[16] * (L / 16) ** 1.5)):
        try:
            Q[f'crossover_L_{tag}'] = brentq(lambda L: np.log(trt(L)) - np.log(th(L)), 16, 200)
        except ValueError:
            Q[f'crossover_L_{tag}'] = float('nan')
    # transfer acceptance fits (fig_scaling_tau (b))
    for k in ([] if skip_tr else ('20', '24', '27', '30')):
        for m in ('rt', 'rtp12'):
            p_ = [(L * L, transfer(D, L, k, m)[0]) for L in (16, 24, 32, 48, 64, 96) if transfer(D, L, k, m)]
            if len(p_) < 3:
                continue
            p_ = [p for p in p_ if 0 < p[1] < 1]
            V = np.array([p[0] for p in p_], float); A = np.array([p[1] for p in p_])
            g, g0 = least_squares(lambda q: np.log(erfc(np.sqrt(np.maximum(q[0] * V + q[1], 1e-12)) / 2)) - np.log(A),
                                  [0.01, 0.0], bounds=([1e-5, -0.5], [1.0, 1.0])).x
            Q[f'trfit_g_k{k}_{m}'] = g
            Q[f'trfit_g0_k{k}_{m}'] = g0
    return Q


def stat_errors(variant, rep='tau_rep'):
    D = Data(variant)
    Q0 = compute(D)
    used = sorted(D.used)
    var = {k: 0.0 for k, v in Q0.items() if isinstance(v, float)}
    for n in used:
        e = entry(n)
        reps = []
        for j in range(len(e['acc_rep'])):
            D.over = {n: (e['acc_rep'][j], e[rep][j])}
            skip = not (n.startswith('T16_') or '_L16_' in n)
            q = compute(D, skip_tr=skip)
            if skip:
                q.update({kk: vv for kk, vv in Q0.items() if kk.startswith('trfit_')})
            reps.append(q)
        D.over = {}
        nn = len(reps)
        for k in var:
            v = np.array([r.get(k, np.nan) for r in reps], float)
            if np.all(np.isfinite(v)):
                var[k] += (nn - 1) / nn * ((v - v.mean()) ** 2).sum()
    # HMC / Wolff reference errors (independent single chains): linear propagation by +-1 sigma
    for alg in ('hmc', 'wolff'):
      for L in (8, 12, 16):
        t, s, _ = REF[(alg, L)]
        hp = {LL: REF[(alg, LL)][0] for LL in (8, 12, 16)}; hm = dict(hp)
        hp[L] = t + s; hm[L] = t - s
        if alg == 'hmc':
            Qp, Qm = compute(D, hp, skip_tr=True), compute(D, hm, skip_tr=True)
        else:
            Qp, Qm = compute(D, None, True, hp), compute(D, None, True, hm)
        for k in var:
            if k in Qp and np.isfinite(Qp[k]) and np.isfinite(Qm[k]):
                var[k] += ((Qp[k] - Qm[k]) / 2) ** 2
    return Q0, {k: float(np.sqrt(v)) for k, v in var.items()}, used


def point_errors(variant):
    """delete-one-point jackknife of the fits."""
    D = Data(variant)
    Q0 = compute(D)
    E = {}
    pts = [(L, k) for L in LS for k in KAP if native(D, L, k, 'rt')]
    reps = []
    for i in range(len(pts)):
        keep = [p for j, p in enumerate(pts) if j != i]
        x = np.array([1 / native(D, L, k, 'rt')[0] - 0.5 for L, k in keep]); y = np.array([native(D, L, k, 'rt')[1] for L, k in keep])
        reps.append(np.exp(np.median(np.log(y / x))))
    E['C'] = jk_err(reps)
    for m in ('rt', 'rtp12'):
        for k in KAP:
            pts_ = [(L * L, native(D, L, k, m)[0]) for L in LS if native(D, L, k, m) and 0 < native(D, L, k, m)[0] < 1]
            if len(pts_) < 4:
                continue
            rc, rs = [], []
            for i in range(len(pts_)):
                kp = [p for j, p in enumerate(pts_) if j != i]
                V = np.array([p[0] for p in kp], float); a = np.array([p[1] for p in kp])
                (c, s0), *_ = np.linalg.lstsq(np.c_[V, np.ones_like(V)], (2 * erfcinv(a)) ** 2, rcond=None)
                rc.append(c); rs.append(s0)
            E[f'sig2_{m}_k{k}'] = jk_err(rc); E[f's0_{m}_k{k}'] = jk_err(rs)
            if m == 'rt' and k == '27':
                E['efold_volume_k27'] = jk_err([4 / c for c in rc])
                c27s = list(zip(rc, rs))
    for m in ('rt', 'rtp12', 'imhG'):
        t = [(L, native(D, L, '27', m)[1]) for L in (8, 10, 12, 16) if native(D, L, '27', m)]
        if len(t) >= 3:
            E[f'exp_{m}_k27'] = jk_err([np.polyfit(np.log([a for a, _ in tt]), np.log([b for _, b in tt]), 1)[0]
                                        for tt in [t[:i] + t[i + 1:] for i in range(len(t))]])
    for alg in ('hmc', 'wolff'):
        t = [(L, REF[(alg, L)][0]) for L in (8, 12, 16)]
        E[f'exp_{alg}_k27'] = jk_err([np.polyfit(np.log([a for a, _ in tt]), np.log([b for _, b in tt]), 1)[0]
                                      for tt in [t[:i] + t[i + 1:] for i in range(len(t))]])
    # crossover: delete-one point in the kappa=0.27 fit, in C, and in the HMC power law (each separately, then quadrature)
    hmc = {L: REF[('hmc', L)][0] for L in (8, 12, 16)}
    b, a = np.polyfit(np.log(list(hmc)), np.log(list(hmc.values())), 1)

    def cross(C, c27, s27, a, b):
        trt = lambda L: C * (1 / erfc(np.sqrt(c27 * L * L + s27) / 2) - 0.5)
        try:
            return brentq(lambda L: np.log(trt(L)) - (a + b * np.log(L)), 16, 200)
        except ValueError:
            return np.nan
    c27, s27 = Q0['sig2_rt_k27'], Q0['s0_rt_k27']
    v1 = jk_err([cross(Q0['C'], c, s, a, b) for c, s in c27s])
    v2 = jk_err([cross(C_, c27, s27, a, b) for C_ in reps])
    HL = list(hmc.items())
    v3 = jk_err([cross(Q0['C'], c27, s27, *np.polyfit(np.log([l for l, _ in hh]), np.log([t for _, t in hh]), 1)[::-1])
                 for hh in [HL[:i] + HL[i + 1:] for i in range(3)]])
    E['crossover_L_fit'] = float(np.sqrt(v1 ** 2 + v2 ** 2 + v3 ** 2))
    E['crossover_L_fit_parts'] = dict(sig2_fit=v1, C=v2, hmc_powerlaw=v3)
    for k in ('20', '24', '27', '30'):
        for m in ('rt', 'rtp12'):
            p_ = [(L * L, transfer(D, L, k, m)[0]) for L in (16, 24, 32, 48, 64, 96) if transfer(D, L, k, m)]
            p_ = [p for p in p_ if 0 < p[1] < 1]
            if len(p_) < 4:
                continue
            rg = []
            for i in range(len(p_)):
                kp = [p for j, p in enumerate(p_) if j != i]
                V = np.array([p[0] for p in kp], float); A = np.array([p[1] for p in kp])
                rg.append(least_squares(lambda q: np.log(erfc(np.sqrt(np.maximum(q[0] * V + q[1], 1e-12)) / 2)) - np.log(A),
                                        [0.01, 0.0], bounds=([1e-5, -0.5], [1.0, 1.0])).x)
            rg = np.array(rg)
            E[f'trfit_g_k{k}_{m}'] = jk_err(rg[:, 0]); E[f'trfit_g0_k{k}_{m}'] = jk_err(rg[:, 1])
    return E


if __name__ == '__main__':
    out = dict(ref_tau={f'{a}_L{L}': dict(tau=v[0], err=v[1], method=v[2]) for (a, L), v in REF.items()})
    variants = sys.argv[1:] or ['paper', 'buf']
    for variant in variants:
        Q0, st, used = stat_errors(variant)
        _, st_med, _ = stat_errors(variant, 'tau_rep_med')
        pe = point_errors(variant)
        res = {}
        for k, v in Q0.items():
            if isinstance(v, float):
                s, p = st.get(k, np.nan), pe.get(k)
                res[k] = dict(value=v, stat=s, stat_med=st_med.get(k), pts=p,
                              total=float(np.sqrt(s ** 2 + (p or 0) ** 2)))
            else:
                res[k] = dict(value=v)
        res['_crossover_parts'] = pe.get('crossover_L_fit_parts')
        res['_n_cells_used'] = len(used)
        out[variant] = res
    dump(out, f'fits_jk_{"_".join(variants)}.json')
    for variant in variants:
        r = out[variant]
        print(variant, {k: (r[k]['value'] if not isinstance(r[k]['value'], float) else round(r[k]['value'], 4), round(r[k].get('stat', 0), 4), r[k].get('pts') and round(r[k]['pts'], 4))
                        for k in ('C', 'C_rms', 'C_npts', 'sig2_peak', 'sig2_rt_k20', 'sig2_rt_k27', 's0_rt_k27', 'sig2_rt_k30',
                                  's0_broken_mean_k275_30', 'efold_volume_k27', 'exp_rt_k27', 'exp_rtp12_k27', 'exp_imhG_k27',
                                  'exp_hmc_k27', 'exp_wolff_k27', 'crossover_L_fit', 'crossover_L_anchor15',
                                  'accept_native_maxdev', 'maxres_rt', 'trtau_ratio_L24_k27') if k in r})
