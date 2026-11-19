#!/usr/bin/env python3
"""Error / reference audit of the single long chains in longchain_min (2026-11-06).

For every phi4 / CMCD / DBS chain file with an HMC reference:
  * recompute from the traces (post burn-in = json 'burn'), per chain and observable (|M|, M2 = |M|^2, S/V):
    mean, tau = max(Sokal, batch means with 50 batches), se = sqrt(var tau / n); estimators and the convention
    tau = 1 + 2 sum rho (= n / ESS) exactly as in longchain.py, so the |M| / S/V numbers reproduce the chain jsons;
  * pool the 4 chains: mean of means, se_pool = sqrt(sum se_i^2) / 4;
  * between-chain check: chi2 = sum ((m_i - pooled) / se_i)^2 with 3 dof (p-value), and
    R = sd(chain means) / rms(se_i)  (R >> 1: the per-chain se underestimates the error);
  * reference se = max(sqrt(sum se_run^2) / n_runs, sd_runs / sqrt(n_runs)) from the HMC json;
  * z_chain = (pooled - ref) / se_pool,  z_full = (pooled - ref) / sqrt(se_pool^2 + se_ref^2).
Files without traces fall back to the per-chain json stats (|M| and S/V only, flagged 'json').
Outputs audit_chains.csv and AUDIT_chains.md next to this file."""
import csv
import glob
import json
import math
import os
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REV = HERE.parents[1]
LC = REV / 'longchain_min'
REF_DIRS = [REV / 'phi4_E1', REV / 'phi4' / 'refL', REV / 'phi4']
OBS = ('absM', 'M2', 'action')


def chi2_sf(x, k):
    # survival function of chi2 with k = 3 dof (closed form)
    if k != 3:
        raise ValueError
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


def tau_sokal(x, c=5.0):
    # same estimator and convention as longchain.py: tau = 1 + 2 sum rho (= n / ESS), window m >= c tau
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 1.0
    rho = ac / ac[0]
    cs = np.cumsum(rho[1:])
    for m in range(1, n):
        tau = 1.0 + 2.0 * cs[m - 1]
        if m >= c * tau:
            return float(tau)
    return float(tau)


def tau_bm(x, nb=50):
    x = np.asarray(x, float)
    b = len(x) // nb
    if b < 2 or np.var(x) <= 0:
        return 0.0
    m = x[:nb * b].reshape(nb, b).mean(1)
    return float(b * np.var(m, ddof=1) / np.var(x))


def stats(x):
    t = max(tau_sokal(x), tau_bm(x))
    return float(np.mean(x)), float(math.sqrt(np.var(x) * t / len(x))), float(t)


def find_ref(path):
    if not path:
        return None
    b = os.path.basename(path)
    sub = os.path.basename(os.path.dirname(path))
    order = sorted(REF_DIRS, key=lambda d: d.name != sub)
    for d in order:
        if (d / b).exists():
            return d / b
    return None


def ref_vals(p):
    d = json.load(open(p))
    out = {}
    for o in OBS:
        mean = d['reference'][o]['mean']
        ses = [r[o]['se'] for r in d['runs']]
        n = len(ses)
        se1 = math.sqrt(sum(s * s for s in ses)) / n
        se2 = d['between_runs'][o]['sd_runs'] / math.sqrt(n)
        out[o] = (mean, max(se1, se2))
    return out


def main():
    files = sorted(glob.glob(str(LC / '*.json')) + glob.glob(str(LC / 'local_buf_*' / '*.json')))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = d.get('args', {})
        if a.get('kind') not in ('phi', 'cmcd', None) or 'chains' not in d:
            continue
        rp = find_ref(a.get('ref', ''))
        if rp is None:
            continue
        ref = ref_vals(rp)
        name = os.path.basename(f)[:-5]
        if '/local_buf_' in f:
            name = 'local_' + name
        tr = f[:-5] + '_traces.npz'
        src = 'trace' if os.path.exists(tr) else 'json'
        per = {o: [] for o in OBS}
        if src == 'trace':
            z = np.load(tr)
            burn = int(d.get('burn', 0))
            am = z['absM'][burn:].astype(float)
            ac = z['action'][burn:].astype(float)
            for k in range(am.shape[1]):
                per['absM'].append(stats(am[:, k]))
                per['M2'].append(stats(am[:, k] ** 2))
                per['action'].append(stats(ac[:, k]))
        else:
            for c in d['chains']:
                for o in ('absM', 'action'):
                    t = max(c[o]['tau_sokal'], c[o].get('tau_batch_means') or 0.0)
                    per[o].append((c[o]['mean'], c[o]['se'], t))
        for o in OBS:
            if not per[o]:
                continue
            m = np.array([p[0] for p in per[o]])
            s = np.array([p[1] for p in per[o]])
            t = np.array([p[2] for p in per[o]])
            k = len(m)
            pooled = float(m.mean())
            se_pool = float(math.sqrt((s ** 2).sum()) / k)
            chi2 = float((((m - pooled) / s) ** 2).sum())
            r_ratio = float(np.std(m, ddof=1) / math.sqrt((s ** 2).mean()))
            rm, rs = ref[o]
            rows.append(dict(cell=name, obs=o, src=src, n=d['n_transitions'], ok=d.get('criteria_met'),
                             tau_med=float(np.median(t)), tau_max=float(t.max()), mean=pooled, se=se_pool,
                             ref=rm, ref_se=rs, ref_frac=rs / se_pool, z_chain=(pooled - rm) / se_pool,
                             z_full=(pooled - rm) / math.hypot(se_pool, rs),
                             zi_max=float(np.max(np.abs((m - rm) / np.hypot(s, rs)))),
                             chi2=chi2, p_between=chi2_sf(chi2, 3) if k == 4 else float('nan'), R=r_ratio,
                             ref_file=rp.relative_to(REV).as_posix()))
    keys = list(rows[0])
    with open(HERE / 'audit_chains.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), 'rows,', len({r['cell'] for r in rows}), 'cells')


if __name__ == '__main__':
    main()
