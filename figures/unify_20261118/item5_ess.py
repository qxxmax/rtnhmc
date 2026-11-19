"""Item 5: App G tab:udb_compare ESS column at fixed N = 16384: 4 disjoint batches of the 65536 stored forward works per
cell (CMCD/DBS: logz_udb_20261116/{cmcd,dbs}_L8_k0XX_rt.npz WF; NHMC: raw_snis/E2_L8_k0XX.npz W), log w = -W.
Per batch: Kish ESS/N, PSIS Pareto k-hat and PSIS-ESS/N (= 1/sum(w~^2)/N with the PSIS-smoothed normalised weights).
PSIS = arviz _psislw / _gpdfit (Vehtari et al. 2024; r_eff = 1, tail M = ceil(min(0.2 N, 3 sqrt N)), weak prior on k),
re-implemented here because the local arviz import is broken. Old table values: figs/table_udb_compare.json."""
import os
import json, numpy as np
from scipy.special import logsumexp
REV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/'
def gpdfit(ary):
    prior_bs, prior_k = 3, 10; n = len(ary); m = 30 + int(n ** 0.5)
    b = 1 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5)); b /= prior_bs * ary[int(n / 4 + 0.5) - 1]; b += 1 / ary[-1]
    k = np.log1p(-b[:, None] * ary).mean(1); L = n * (np.log(-(b / k)) - k - 1)
    w = 1 / np.exp(L - L[:, None]).sum(1); ok = w >= 10 * np.finfo(float).eps; w, b = w[ok], b[ok]; w /= w.sum()
    bp = (b * w).sum(); kp = np.log1p(-bp * ary).mean(); sig = -kp / bp
    return (n * kp + prior_k * 0.5) / (n + prior_k), sig
def gpinv(p, k, s):
    return s * (-np.log1p(-p) if abs(k) < np.finfo(float).eps else np.expm1(-k * np.log1p(-p)) / k)
def psislw(lw):
    x = lw - lw.max(); n = x.size; ci = -int(np.ceil(min(0.2 * n, 3 * np.sqrt(n)))) - 1
    si = np.argsort(x); xc = max(x[si[ci]], np.log(np.finfo(float).tiny)); ec = np.exp(xc)
    ti = np.where(x > xc)[0]; xt = x[ti]; o = np.argsort(xt); xt = np.exp(xt) - ec
    k, s = gpdfit(xt[o])
    if np.isfinite(k):
        st = np.arange(0.5, len(ti)) / len(ti); x[ti[o]] = np.log(gpinv(st, k, s) + ec); x[x > 0] = 0
    return x - logsumexp(x), float(k)
def kish(lw): lw = lw - lw.max(); w = np.exp(lw); return float(w.sum() ** 2 / (w ** 2).sum() / w.size)
tab = {(r['sampler'], r['kappa']): r for r in json.load(open(REV + 'figs/table_udb_compare.json'))}
rows = []
for k in ['020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030']:
    kap = '0.' + k[1:]
    for alg in ('NHMC', 'CMCD', 'DBS'):
        f = REV + (f'raw_snis/E2_L8_k{k}.npz' if alg == 'NHMC' else f'logz_udb_20261116/{alg.lower()}_L8_k{k}_rt.npz')
        z = np.load(f); lw = -(z['W'] if alg == 'NHMC' else z['WF']).astype(np.float64)
        nfin = int(np.isfinite(lw).sum()); assert lw.size == 65536 and nfin == 65536, (f, lw.size, nfin)
        K, P, KH = [], [], []
        for bt in lw.reshape(4, 16384):
            K.append(kish(bt)); l2, kh = psislw(bt.copy()); P.append(float(1 / np.exp(2 * l2).sum() / bt.size)); KH.append(kh)
        la, kha = psislw(lw.copy())
        t = tab[(alg, kap)]
        r = dict(kappa=kap, alg=alg, file=f, old_table_ess=t['ess'], old_N=262144 if alg == 'NHMC' else 102400,
                 kish_b4=K, kish_med=float(np.median(K)), kish_min=min(K), kish_max=max(K),
                 psis_ess_b4=P, psis_med=float(np.median(P)), psis_min=min(P), psis_max=max(P),
                 khat_b4=KH, khat_med=float(np.median(KH)), khat_max=max(KH), khat_all65536=kha, kish_all65536=kish(lw), var_logw=float(lw.var()))
        rows.append(r)
        print(f"{kap} {alg:4s} old {r['old_table_ess']:.3f} | Kish16384 {r['kish_med']:.3f} [{r['kish_min']:.3f},{r['kish_max']:.3f}] | PSIS-ESS {r['psis_med']:.3f} [{r['psis_min']:.3f},{r['psis_max']:.3f}] | khat {r['khat_med']:.2f} (max {r['khat_max']:.2f}, all {kha:.2f}) | Kish65536 {r['kish_all65536']:.3f}")
json.dump(rows, open(REV + 'unify_20261118/item5_ess.json', 'w'), indent=1)
