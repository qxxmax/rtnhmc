#!/usr/bin/env python3
"""log Z(kappa) by thermodynamic integration of the Wolff / HMC hop density (2026-11-07).

    log Z(kappa) = V log z1 + int_0^kappa 2 V <h>_k dk,   h = (1/V) sum_{x,mu} phi_x phi_{x+mu},
z1 = single-site integral (phi4_logz_ref.site_quadrature, same convention as the AIS log Z: no + lambda constant).
Inputs: the 61-point kappa grid (0 ... 0.30, step 0.005) per L and algorithm in this folder (1e6 samples each).
  * per point <h> = inverse-variance mean of Wolff and HMC (both also integrated alone);
  * composite Simpson at h = 0.005 (h = 0.001 on [0.25, 0.29] where that refinement exists); final 3/8 panel if odd;
  * statistical se = sqrt(sum (w_i 2V se_i)^2) (independent runs);
  * discretisation: D = I(h) - I(2h) (every other node per segment); quadrature error estimate |D| / 15 (Richardson);
  * Wolff vs HMC consistency per point: z_i = (h_W - h_H) / sqrt(se_W^2 + se_H^2).
Compared with the AIS log Z (logz_grid/, results_sync/logz_ref_*.json) and the NHMC estimates (buffer-model SNIS log Z at
N = 262144, models/phi4_buffer/*/eval_N262144.json; paper_L8 SNIS log_mean_w, phi4_E1_lmw/, mean +- sd/sqrt(3) of 3 reps).
Outputs TI_logZ.md, ti_logz.csv, ti_hop_consistency.csv."""
import csv
import glob
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RS = HERE.parents[1]
sys.path.insert(0, str(RS.parent))
from phi4_logz_ref import site_quadrature  # noqa: E402

LAM = 0.022
LOGZ1 = site_quadrature(LAM)[0]
FINE_LO, FINE_HI = 250, 290                         # 1e-3 grid where the hop density is steep (L12 / L16)


def weights(n, h):
    """Composite Newton-Cotes weights for n intervals (Simpson; a final 3/8 panel if n is odd)."""
    w = np.zeros(n + 1)
    m = n if n % 2 == 0 else n - 3
    if m < 0:
        raise ValueError(n)
    for j in range(0, m, 2):
        w[j:j + 3] += h / 3 * np.array([1.0, 4.0, 1.0])
    if m != n:
        w[m:m + 4] += 3 * h / 8 * np.array([1.0, 3.0, 3.0, 1.0])
    return w


def load():
    D = {}
    for f in glob.glob(str(HERE / '*.json')):
        d = json.load(open(f))
        if 'observables' not in d:
            continue
        L = d['shape'][0]
        k = round(d['params']['kappa'] * 1000)            # kappa in units of 1e-3
        h = d['observables']['hop']
        D[(L, d['algorithm'], k)] = (h['mean'], h['err'], h['tau_int'])
    return D


def series(D, L, alg):
    ks = sorted(k for (l, a, k) in D if l == L and a == alg)
    return {k: D[(L, alg, k)][:2] for k in ks}


def combine(a, b):
    out = {}
    for k in set(a) & set(b):
        (m1, s1), (m2, s2) = a[k], b[k]
        w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
        out[k] = ((w1 * m1 + w2 * m2) / (w1 + w2), 1 / math.sqrt(w1 + w2))
    return out


def nodes(s, kmax):
    """Integration segments up to kmax (units 1e-3): step 5 everywhere, step 1 inside [FINE_LO, FINE_HI] if complete."""
    fine = all(i in s for i in range(FINE_LO, FINE_HI + 1))
    segs = []
    if fine and kmax > FINE_LO:
        segs.append(list(range(0, FINE_LO + 1, 5)))
        segs.append(list(range(FINE_LO, min(kmax, FINE_HI) + 1)))
        if kmax > FINE_HI:
            segs.append(list(range(FINE_HI, kmax + 1, 5)))
    else:
        segs.append(list(range(0, kmax + 1, 5)))
    return segs


def ti(s, L, kmax, stride):
    """stride 1: nodes as given; stride 2: every other node of each segment (quadrature-error estimate)."""
    V = L * L
    tot, var = 0.0, 0.0
    for seg in nodes(s, kmax):
        seg = seg[::stride]
        if len(seg) == 1:
            continue
        if len(seg) < 3 or any(i not in s for i in seg):
            return None
        hs = (seg[1] - seg[0]) * 1e-3
        w = weights(len(seg) - 1, hs)
        m = np.array([s[i][0] for i in seg])
        e = np.array([s[i][1] for i in seg])
        tot += float(w @ m)
        var += float(((w * e) ** 2).sum())
    return V * LOGZ1 + 2 * V * tot, 2 * V * math.sqrt(var)


def ais_refs():
    R = {}
    for f in sorted(glob.glob(str(RS / 'revision' / 'logz_grid' / 'logz_L*.json'))) + \
            sorted(glob.glob(str(RS / 'logz_ref_L*.json'))):
        d = json.load(open(f))
        L = d['config']['L']
        for k, v in d['results'].items():
            R.setdefault((L, round(float(k), 3)), (v['logZ_mean'], v['logZ_sem'], pathlib.Path(f).name))
    return R


def nhmc_refs():
    R = {}
    for f in glob.glob(str(RS / 'models' / 'phi4_buffer' / '*' / 'eval_N262144.json')) + \
            glob.glob(str(RS / 'revision' / 'phi4_buffer_evals' / '*eval_N262144.json')):
        d = json.load(open(f))
        e = d[-1] if isinstance(d, list) else d
        R[('buf', int(e['L']), round(float(e['kappa']), 3))] = (e['logZ'], e['logZ_se'], e.get('ess_frac'))
    grp = {}
    for f in glob.glob(str(RS / 'revision' / 'phi4_E1_lmw' / '*.json')):
        d = json.load(open(f))
        a = d['args']
        r = d['methods']['snis']['runs'][0]
        grp.setdefault((int(a['L']), round(float(a['kappa']), 3)), []).append((r['log_mean_w'], r['kish_ess_frac']))
    for (L, k), v in grp.items():
        x = np.array([t[0] for t in v])
        R[('paperL8', L, k)] = (float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float('nan'),
                                float(np.mean([t[1] for t in v])))
    return R


def main():
    D = load()
    Ls = sorted({l for (l, a, k) in D})
    rows, cons = [], []
    for L in Ls:
        W, H = series(D, L, 'wolff'), series(D, L, 'hmc')
        C = combine(W, H)
        for k in sorted(set(W) & set(H)):
            (m1, s1), (m2, s2) = W[k], H[k]
            cons.append(dict(L=L, kappa=k * 1e-3, h_wolff=m1, se_wolff=s1, h_hmc=m2, se_hmc=s2,
                             z=(m1 - m2) / math.hypot(s1, s2), tau_wolff=D[(L, 'wolff', k)][2], tau_hmc=D[(L, 'hmc', k)][2]))
        for ki in range(10, max(C) + 1, 10):                # kappa = 0.01 ... 0.30
            r = dict(L=L, kappa=round(ki * 1e-3, 3), fine=all(i in C for i in range(FINE_LO, FINE_HI + 1)))
            for lab, s in (('comb', C), ('wolff', W), ('hmc', H)):
                a = ti(s, L, ki, 1)
                if a is None:
                    continue
                r[f'logZ_{lab}'], r[f'se_{lab}'] = a
            b = ti(C, L, ki, 2)
            if b is not None and 'logZ_comb' in r:
                r['logZ_comb_h2'] = b[0]
                r['quad_err'] = abs(r['logZ_comb'] - b[0]) / 15
            rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('L', 'kappa'), k))
    with open(HERE / 'ti_logz.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(HERE / 'ti_hop_consistency.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cons[0]))
        w.writeheader()
        w.writerows(cons)

    A, N = ais_refs(), nhmc_refs()
    T = {(r['L'], r['kappa']): r for r in rows}
    out = ['# log Z by thermodynamic integration (Wolff / HMC hop density) vs AIS vs NHMC (`ti_logz.py`)', '',
           f'log z1 = {LOGZ1:.10f} (lambda = {LAM}); grid kappa = 0 ... 0.30, step 0.005, 1e6 samples per point and algorithm.',
           'TI = Simpson on Wolff/HMC inverse-variance mean; se_stat from the per-point errors; quad = |I(0.005) - I(0.01)| / 15.',
           '', '## Wolff vs HMC hop density, per point', '']
    for L in Ls:
        z = np.array([c['z'] for c in cons if c['L'] == L])
        out.append(f'- L = {L}: {len(z)} points, chi2/dof {np.mean(z ** 2):.2f}, max abs z {np.abs(z).max():.2f}')
    out += ['', '## log Z comparison', '',
            '| L | kappa | TI | se_stat | quad | TI Wolff - HMC | AIS | TI - AIS (z) | NHMC buffer SNIS (ESS) | NHMC - TI (z) | paper_L8 SNIS (ESS) | - TI (z) |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|']
    keys = sorted({(L, k) for (L, k) in A} | {(L, k) for (_, L, k) in N})
    for (L, k) in keys:
        r = T.get((L, k))
        if r is None:
            continue
        ti_v, ti_s, q = r['logZ_comb'], r['se_comb'], r.get('quad_err', float('nan'))
        tot = math.hypot(ti_s, q if q == q else 0.0)
        dwh = r.get('logZ_wolff', float('nan')) - r.get('logZ_hmc', float('nan'))
        cells = [str(L), f'{k:.2f}', f'{ti_v:.4f}', f'{ti_s:.4f}', f'{q:.4f}', f'{dwh:+.4f}']
        if (L, k) in A:
            a, s, _ = A[(L, k)]
            cells += [f'{a:.4f} +- {s:.4f}', f'{ti_v - a:+.4f} ({(ti_v - a) / math.hypot(tot, s):+.1f})']
        else:
            cells += ['-', '-']
        for src in ('buf', 'paperL8'):
            if (src, L, k) in N:
                v, s, ess = N[(src, L, k)]
                s2 = s if s == s else 0.0
                cells += [f'{v:.4f} +- {s:.4f} ({100 * ess:.1f}%)', f'{v - ti_v:+.4f} ({(v - ti_v) / math.hypot(tot, s2):+.1f})']
            else:
                cells += ['-', '-']
        out.append('| ' + ' | '.join(cells) + ' |')
    (HERE / 'TI_logZ.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
