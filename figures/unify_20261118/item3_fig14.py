"""Item 3: fig14 (L=8 ACF) chain-length unification + long-chain-table convention tau for all 6 curves. Read-only."""
import os
import sys, json, numpy as np
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118'); import ucommon as U
R = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/longchain_min'
TEXT = {'k027_imhG': 3.1, 'k030_imhG': 3.0, 'k027_rt': 1.9, 'k030_rt': 1.6, 'k027_rtp12': 4.3, 'k030_rtp12': 1.2}
def acf_pool(x, lag):  # figure estimator (pooled after per-chain centering)
    x = x - x.mean(0); v = (x * x).mean()
    return np.array([1.0] + [float((x[:-j] * x[j:]).mean() / v) for j in range(1, lag + 1)])
def tau_g2(rho):
    for m in range(1, len(rho)):
        t = 0.5 + rho[1:m + 1].sum()
        if m >= 5 * 2 * t: return t
    return 0.5 + rho[1:].sum()
out = {}
for k in ('027', '030'):
    for m in ('imhG', 'rt', 'rtp12'):
        z = np.load(f'{R}/E2_L8_k{k}_{m}_traces.npz'); a = z['absM']; acc = z['acc']; N = a.shape[0]
        js = json.load(open(f'{R}/E2_L8_k{k}_{m}.json'))
        json_tau = [max(c['absM']['tau_sokal'], c['absM']['tau_batch_means'] or 0) / 2 for c in js['chains']]
        res = dict(n_transitions=N, text=TEXT[f'k{k}_{m}'], json_unified_median=float(np.median(json_tau)))
        for lab, n in (('full', N), ('trunc1e4', 10000)):
            x = a[:n]; b = n // 10; post = x[b:]
            rho = acf_pool(post, 400)
            cs = [U.chain_tau(post[:, c]) for c in range(4)]
            med, jk = U.jk_median([c['tau'] for c in cs]); meds, jks = U.jk_median([c['tau_sokal'] for c in cs])
            res[lab] = dict(n=n, fig_pooled_tau=tau_g2(rho), unified_median=med, unified_jk=jk, sokal_median=meds,
                            chains=[c['tau'] for c in cs], acc=float(acc[b:n].mean()),
                            rho_1_5_10_20_30=[float(rho[j]) for j in (1, 5, 10, 20, 30)], rho_0_40=rho[:41].tolist())
        if N > 10000:
            res['acf_maxabs_diff_full_vs_trunc_j<=40'] = float(np.abs(np.array(res['full']['rho_0_40']) - np.array(res['trunc1e4']['rho_0_40'])).max())
        out[f'k{k}_{m}'] = res
        f, t = res['full'], res['trunc1e4']
        print(f"k{k} {m:6s} N={N}: text {res['text']}  json {res['json_unified_median']:.2f} | full: fig {f['fig_pooled_tau']:.2f} unif {f['unified_median']:.2f}({f['unified_jk']:.2f}) acc {f['acc']:.3f} | 1e4: fig {t['fig_pooled_tau']:.2f} unif {t['unified_median']:.2f}({t['unified_jk']:.2f}) acc {t['acc']:.3f}" + (f"  dACF {res['acf_maxabs_diff_full_vs_trunc_j<=40']:.3f}" if N > 10000 else ''))
json.dump(out, open(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/item3_fig14.json', 'w'), indent=1)
