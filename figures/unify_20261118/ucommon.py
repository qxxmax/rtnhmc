"""Unified estimators (2026-11-18): per chain g = max(Sokal c=5, 50-batch means), tau_int = g/2,
median over chains, error = leave-one-chain-out jackknife of the median. Copies of longchain.py functions."""
import numpy as np
def sokal_single(x, c=5.0):
    x = np.asarray(x, dtype=np.float64); x = x - x.mean(); n = x.size
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0: return 1.0, 0
    rho = ac / ac[0]; cs = np.cumsum(rho[1:])
    for m in range(1, n):
        tau = 1.0 + 2.0 * cs[m - 1]
        if m >= c * tau: return float(tau), m
    return float(tau), n - 1
def batch_means_tau(x, nb=50):
    x = np.asarray(x, dtype=np.float64); b = x.size // nb
    if b < 2 or x.var() <= 0: return None
    bm = x[:b * nb].reshape(nb, b).mean(1)
    return float(b * bm.var(ddof=1) / x.var())
def chain_tau(x):
    s, w = sokal_single(x); bm = batch_means_tau(x)
    return dict(g_sokal=s, g_bm=bm, tau=max(s, bm or 0.0) / 2, tau_sokal=s / 2, window=w)
def jk_median(v):
    v = np.asarray(v, float); n = len(v)
    if n < 2: return float(np.median(v)), float('nan')
    th = np.array([np.median(np.delete(v, i)) for i in range(n)])
    return float(np.median(v)), float(np.sqrt((n - 1) / n * ((th - th.mean()) ** 2).sum()))
def acf(x, nmax):
    x = np.asarray(x, float) - np.mean(x); n = x.size
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac[:nmax + 1] / ac[0]
