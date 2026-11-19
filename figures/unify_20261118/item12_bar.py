"""Item 1 (tab:bar_transfer tau, unified) + item 2 (fig_bar_prefix with 20 blocks). Read-only."""
import os
import sys, json, glob, math, numpy as np
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118'); import ucommon as U
REV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'
sys.path.insert(0, REV + '/ti_wolff_hmc'); import ti_logz as T
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/revision/longchain'); import logz_estimators as E
old = json.load(open(REV + '/figs_20261116/bar_table_20261116.json'))
def load_ti(n):
    s = {}
    for f in glob.glob(f'{REV}/ti_rect2_20261116/*R{n}x8_k*_wolff2.json'):
        d = json.load(open(f)); s[round(d['params']['kappa'] * 1000)] = (d['observables']['hop']['mean'], d['observables']['hop']['err'])
    return s
def ti(s, kmax):
    def integ(stride):
        tot, var = 0, 0
        for seg in (list(range(0, 251, 5)), list(range(250, kmax + 1))):
            seg = [i for i in seg if i <= kmax][::stride]
            if len(seg) < 2: continue
            w = T.weights(len(seg) - 1, (seg[1] - seg[0]) * 1e-3)
            tot += float(w @ np.array([s[i][0] for i in seg])); var += float(((w * np.array([s[i][1] for i in seg])) ** 2).sum())
        return tot, math.sqrt(var)
    a, e = integ(1); b, _ = integ(2)
    return T.LOGZ1 + 2 * a, math.hypot(2 * e, 2 * abs(a - b) / 15)
def files(n, k):
    return [f'{REV}/logz_chains/R_64x8_k0{k}_rtp12.npz'] if n == 64 else sorted(glob.glob(f'{REV}/logz_rect2_20261115/R_{n}x8_k0{k}_rtp12_s*.npz'))
out = dict(item1={}, item2={}, item1_bar_sensitivity={})
# tau estimator variant for the BAR weight: max(Sokal, BM) per chain (sensitivity only)
_orig_tau = E.tau_of
def tau_of_max(x):
    s = _orig_tau(x); bm = U.batch_means_tau(x); return max(s, bm or 0.0, 1.0)
for k in ('20', '30'):
    for n in (64, 128, 256, 512):
        key = f'{n}x8_k0{k}'; zs = [np.load(f) for f in files(n, k)]
        per_seed = []; allc = []
        for z in zs:
            M = z['absM']; N = M.shape[0]; b = N // 10; post = M[b:b + ((N - b) // 20) * 20]
            cs = [U.chain_tau(post[:, c]) for c in range(M.shape[1])]; allc += cs
            med, err = U.jk_median([c['tau'] for c in cs])
            per_seed.append(dict(median=med, jk=err, chains=[c['tau'] for c in cs], chains_sokal=[c['tau_sokal'] for c in cs]))
        t_all = [c['tau'] for c in allc]; t_s = [c['tau_sokal'] for c in allc]
        m16, e16 = U.jk_median(t_all)
        seedmeds = [p['median'] for p in per_seed]
        out['item1'][key] = dict(old_tau_sokal16_half=old[key]['tau_M'], sokal16_median_recomputed=float(np.median(t_s)),
            unified16_median=m16, unified16_jk_chain=e16, seed_medians=seedmeds,
            median_of_seed_medians=float(np.median(seedmeds)), seed_range=[min(seedmeds), max(seedmeds)],
            seed_jk=U.jk_median(seedmeds)[1] if len(seedmeds) > 1 else None,
            bm_over_sokal_median=float(np.median([c['g_bm'] / c['g_sokal'] for c in allc])), per_seed=per_seed)
        o = out['item1'][key]
        print(f"{key}: old {o['old_tau_sokal16_half']:.2f}  sokal16 {o['sokal16_median_recomputed']:.2f}  unified16 {m16:.2f}({e16:.2f})  seed medians {np.round(seedmeds,2)}  BM/S {o['bm_over_sokal_median']:.2f}", flush=True)
        # BAR sensitivity: same table estimator but tau from max(Sokal,BM)
        WF = np.load(f'{REV}/raw_snis/R_{n}x8_k0{k}.npz')['W']
        WR = np.concatenate([z['WR'] for z in zs], 1); M = np.concatenate([z['absM'] for z in zs], 1); A = np.concatenate([z['acc'] for z in zs], 1)
        lz, Fe = ti(load_ti(n), int(k) * 10); V = n * 8; F = -lz * V; Fe *= V
        orig = E.tau_of
        r0 = E.estimates(dict(WR=WR, WF=WF, absM=M, acc=A))
        E.tau_of = tau_of_max; r1 = E.estimates(dict(WR=WR, WF=WF, absM=M, acc=A)); E.tau_of = orig
        f = lambda r: [(r['bar'][0] + F) / V * 1e4, math.hypot(r['bar'][1], Fe) / V * 1e4]
        out['item1_bar_sensitivity'][key] = dict(bar_sokal=f(r0), bar_maxSBM=f(r1), tauM_g_sokal=r0['tau_M'], tauM_g_max=r1['tau_M'])
        print('   BAR sokal %+.3f(%.3f)  max(S,BM) %+.3f(%.3f)' % (*f(r0), *f(r1)), flush=True)
# item 2
for k in ('20', '30'):
    for n in (128, 256, 512):
        V = n * 8; lz, _ = ti(load_ti(n), int(k) * 10); F = -lz * V
        zs = [np.load(f) for f in files(n, k)]
        WF = np.load(f'{REV}/raw_snis/R_{n}x8_k0{k}.npz')['W']
        WR = np.concatenate([z['WR'] for z in zs], 1); M = np.concatenate([z['absM'] for z in zs], 1); A = np.concatenate([z['acc'] for z in zs], 1)
        rows = []
        for m in (2500, 5000, 10000, 20000):
            d = dict(WR=WR[:m], WF=WF, absM=M[:m], acc=A[:m])
            r10 = E.estimates(d, B=10); r20 = E.estimates(d, B=20)
            rows.append(dict(prefix=m, bar10=(r10['bar'][0] + F) / V * 1e4, err10=r10['bar'][1] / V * 1e4,
                             bar20=(r20['bar'][0] + F) / V * 1e4, err20=r20['bar'][1] / V * 1e4, n_fwd10=r10['n_fwd'], n_fwd20=r20['n_fwd']))
            q = rows[-1]; print(f"prefix {n}x8 k0.{k} m={m}: B10 {q['bar10']:+.3f}({q['err10']:.3f})  B20 {q['bar20']:+.3f}({q['err20']:.3f})  ratio {q['err20']/q['err10']:.2f}", flush=True)
        out['item2'][f'{n}x8_k0{k}'] = rows
json.dump(out, open(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/item12_bar.json', 'w'), indent=1)
