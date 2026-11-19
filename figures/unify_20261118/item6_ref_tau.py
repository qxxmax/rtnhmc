"""Item 6: HMC / Wolff reference tau errors: Madras-Sokal (as in the figures: sigma = tau sqrt(2(2W+1)/n), W = 5 g) vs
delete-one-segment jackknife (16 contiguous segments; tau recomputed on the remaining 15 segments, ACF per segment
averaged so that segment junctions do not enter). Traces available locally: figs/data/wh_absm_traces.npz (|m|, 1e6)
and figs/data/wh_m_signed.npz (signed m, Wolff, 2e5 -> |m| and sign m). Read-only."""
import os
import sys, json, math, numpy as np
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118'); import ucommon as U
DD = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs/data'
G = json.load(open(f'{DD}/wh_tau_grid.json')); S = json.load(open(f'{DD}/wh_sign_summary.json'))
NSEG = 16
def seg_acf(segs):
    """mean over segments of the per-segment autocovariance (global mean), normalised"""
    mu = np.mean(np.concatenate(segs)); acc = None
    for s in segs:
        x = s - mu; n = x.size; f = np.fft.rfft(x, 2 * n); a = np.fft.irfft(f * np.conj(f))[:n]
        acc = a if acc is None else acc + a[:acc.size] if a.size >= acc.size else acc[:a.size] + a
    return acc / acc[0]
def sokal_from_rho(rho, c=5.0):
    cs = np.cumsum(rho[1:])
    for m in range(1, rho.size):
        g = 1 + 2 * cs[m - 1]
        if m >= c * g: return float(g), m
    return float(g), rho.size - 1
def bm_g(segs, nb=50):
    x = np.concatenate(segs); return U.batch_means_tau(x, nb)
def analyse(x):
    n = x.size; segs = np.array_split(x, NSEG)
    gS, W = U.sokal_single(x); gB = U.batch_means_tau(x)
    tS = gS / 2; tU = max(gS, gB) / 2
    ms = tS * math.sqrt(2 * (2 * 5 * gS + 1) / n)
    jS, jU = [], []
    for i in range(NSEG):
        rest = [s for j, s in enumerate(segs) if j != i]
        g, _ = sokal_from_rho(seg_acf(rest)); jS.append(g / 2); jU.append(max(g, bm_g(rest)) / 2)
    jk = lambda v: math.sqrt((NSEG - 1) / NSEG * ((np.array(v) - np.mean(v)) ** 2).sum())
    per_seg = [U.sokal_single(s)[0] / 2 for s in segs]
    return dict(n=n, tau_sokal=tS, tau_unified=tU, g_bm_over_sokal=gB / gS, window=W, ms_err=ms,
                jk16_err_sokal=jk(jS), jk16_err_unified=jk(jU), jk_over_ms=jk(jS) / ms,
                segment_tau_sd_over_sqrt16=float(np.std(per_seg, ddof=1) / 4))
out = dict(with_traces={}, without_traces=[])
A = np.load(f'{DD}/wh_absm_traces.npz')
for key in A.files:
    L, k, alg = key.split('_'); r = analyse(np.asarray(A[key], float)); gk = f'{L}_{alg}_k{k[2:]}'
    r['file_value_g/2'] = G[gk]['tau_absm'] / 2 if gk in G else None; r['obs'] = 'absM'; r['src'] = 'wh_absm_traces.npz'
    out['with_traces'][f'{key}_absM'] = r
    print(f"{key} |m|: tau(S) {r['tau_sokal']:.2f} (file {r['file_value_g/2']}) unif {r['tau_unified']:.2f} | MS {r['ms_err']:.2f} JK16 {r['jk16_err_sokal']:.2f} (ratio {r['jk_over_ms']:.2f}) JKunif {r['jk16_err_unified']:.2f}", flush=True)
Sg = np.load(f'{DD}/wh_m_signed.npz')
for key in Sg.files:
    m = np.asarray(Sg[key], float); L, k, alg = key.split('_'); kap = int(k[1:]) / 1000
    sk = [s for s, v in S.items() if v['L'] == int(L[1:]) and v['alg'] == alg and abs(v['kappa'] - kap) < 1e-9]
    for obs, x in (('absM', np.abs(m)), ('sign', np.sign(m))):
        if obs == 'sign' and np.all(np.sign(m) == np.sign(m[0])): continue
        r = analyse(x); r['obs'] = obs; r['src'] = 'wh_m_signed.npz'
        r['file_value_g/2'] = (S[sk[0]]['tau_absm' if obs == 'absM' else 'tau_sign'] or float('nan')) / 2 if sk else None
        out['with_traces'][f'{key}_{obs}'] = r
        print(f"{key} {obs}: tau(S) {r['tau_sokal']:.2f} (file {r['file_value_g/2']}) unif {r['tau_unified']:.2f} | MS {r['ms_err']:.3f} JK16 {r['jk16_err_sokal']:.3f} (ratio {r['jk_over_ms']:.2f})", flush=True)
have_abs = {f"{k.split('_')[0]}_{k.split('_')[2]}_k{k.split('_')[1][2:]}" for k in A.files}
have_sig = {(int(k.split('_')[0][1:]), int(k.split('_')[1][1:]) / 1000) for k in Sg.files}
for gk, v in G.items():
    if gk not in have_abs: out['without_traces'].append(dict(file='wh_tau_grid.json', key=gk, L=v['L'], alg=v['alg'], kappa=v['kappa'], n=v['n'], obs='absM'))
for sk, v in S.items():
    if not (v['alg'] == 'wolff' and (v['L'], v['kappa']) in have_sig):
        out['without_traces'].append(dict(file='wh_sign_summary.json', key=sk, L=v['L'], alg=v['alg'], kappa=v['kappa'], n=v['n'], obs='absM+sign'))
r = [v['jk_over_ms'] for v in out['with_traces'].values()]
out['jk_over_ms_summary'] = dict(median=float(np.median(r)), min=float(min(r)), max=float(max(r)), n=len(r))
print('JK/MS ratio', out['jk_over_ms_summary'], 'without traces:', len(out['without_traces']))
json.dump(out, open(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/item6_ref_tau.json', 'w'), indent=1)
