"""Shared helpers for the 2026-11-17 jackknife pass (read-only on all existing data; numpy only, light).

File lookup mirrors figs_20261116/fig_scaling_20261116.py files()/native()/transfer() exactly (variant 'paper');
variant 'buf' additionally falls back to buf_L{L}_k0{k}_{m}.json for native cells (REVIEW_code_20261117 B1).
tau_int = g/2 = max(Sokal c=5, batch means nb=50)/2 per chain, median over chains (paper statistic).
Delete-one jackknife: err^2 = (n-1)/n sum (theta_i - mean theta_i)^2."""
import glob
import json
import os

import numpy as np

REV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'
OUT = REV + '/jackknife_20261117'
HIM = REV + '/him_figdata_20261116/himout'
IQ = REV + '/iqcb_figdata_20261116'
EXTRA = REV + '/figs/data/traces_extra'
KAP = ['20', '21', '22', '23', '24', '25', '26', '265', '27', '275', '28', '29', '30']
kv = lambda k: int(k) / (100 if len(k) == 2 else 1000)
LS = (4, 6, 8, 10, 12, 16)


def files(pattern='*.json'):
    out = {}
    for d in [REV + '/longchain_min'] + sorted(glob.glob(HIM + '/*/revision_runs/longchain_min')) + \
             sorted(glob.glob(HIM + '/*/revision_runs/pilot_bigL')) + [IQ + '/longchain_min']:
        for f in glob.glob(f'{d}/{pattern}'):
            out.setdefault(os.path.basename(f), f)
    return out


def jk_err(theta):
    t = np.asarray(theta, float)
    n = len(t)
    if n < 2:
        return float('nan')
    return float(np.sqrt((n - 1) / n * ((t - t.mean()) ** 2).sum()))


def jk_delete_one(x, stat):
    """delete-one jackknife of stat over the first axis of x; returns (full, err, replicas)."""
    x = np.asarray(x)
    reps = np.array([stat(np.delete(x, i, 0)) for i in range(len(x))])
    return stat(x), jk_err(reps), reps


def sokal_g(x, c=5.0):
    """identical to longchain.sokal_single (returns g = 1 + 2 sum rho and window)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 1.0, 0
    cs = np.cumsum(ac[1:] / ac[0])
    m = np.arange(1, n)
    g = 1.0 + 2.0 * cs
    ok = np.nonzero(m >= c * g)[0]
    if len(ok):
        return float(g[ok[0]]), int(m[ok[0]])
    return float(g[-1]), n - 1


def bm_g(x, nb=50):
    """identical to longchain.batch_means_tau."""
    x = np.asarray(x, dtype=np.float64)
    b = x.size // nb
    if b < 2 or x.var() <= 0:
        return None
    bm = x[:b * nb].reshape(nb, b).mean(1)
    return float(b * bm.var(ddof=1) / x.var())


def traces_for(jf):
    b = jf[:-5] + '_traces.npz'
    for p in (b, os.path.join(EXTRA, os.path.basename(b))):
        if os.path.exists(p):
            return p
    return None


def load_json(p):
    return json.load(open(p))


def dump(obj, name):
    json.dump(obj, open(os.path.join(OUT, name), 'w'), indent=1, default=float)
