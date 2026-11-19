r"""Integrated autocorrelation time with Wolff automatic windowing.

Reference: U. Wolff, "Monte Carlo errors with less errors",
Comput. Phys. Commun. 156 (2004) 143, arXiv:hep-lat/0306017.

Physics / statistics background
-------------------------------
For a Markov-chain observable series a_1, ..., a_N the mean estimator has a
variance that is inflated relative to i.i.d. sampling by twice the integrated
autocorrelation time:

    var(mean) = (2 * tau_int / N) * var(a) ,

so the honest statistical error is

    err(mean) = sqrt(2 * tau_int / N) * std(a) .

tau_int is defined from the normalised autocorrelation function
rho(t) = Gamma(t) / Gamma(0), with the (biased) autocovariance
Gamma(t) = (1/N) sum_i (a_i - abar)(a_{i+t} - abar):

    tau_int(W) = 0.5 + sum_{t=1..W} rho(t) .

The tail of the sum is dominated by noise, so Wolff truncates at an automatic
window W* = smallest W such that W >= c * tau_int(W) (we use c = 6, the value
Wolff recommends for exponential-like autocorrelations).  The relative error on
tau_int itself is the Madras-Sokal estimate

    dtau / tau = sqrt( 2 * (2 W + 1) / N ) .

Public API
----------
    tau_int(series)   -> (tau, dtau, W)
    stat_error(series)-> (mean, err)

Self-test (python _wolff_tau.py): AR(1) series with analytically known
tau_int = (1 + phi) / (2 (1 - phi)); asserts recovery within 15 %.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["tau_int", "stat_error"]


def _autocov_fft(x):
    """Biased autocovariance Gamma(t) = (1/N) sum_i dx_i dx_{i+t}, t=0..N-1.

    Computed with zero-padded FFT so it is O(N log N) and free of the circular
    wrap-around that a same-length FFT would introduce."""
    n = x.size
    dx = x - x.mean()
    size = 1
    while size < 2 * n:
        size <<= 1
    f = np.fft.rfft(dx, n=size)
    acov = np.fft.irfft(f * np.conj(f), n=size)[:n]
    return acov / n


def tau_int(series, c=6.0):
    """Integrated autocorrelation time with Wolff automatic windowing.

    Parameters
    ----------
    series : 1-D array-like of length N.
    c      : window parameter (default 6.0); window W* is the smallest W with
             W >= c * tau_int(W).

    Returns
    -------
    (tau, dtau, W) : tau_int, its Madras-Sokal error, and the chosen window.
    A pure-noise / constant series returns tau = 0.5 (the i.i.d. value).
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 0.5, 0.0, 0

    acov = _autocov_fft(x)
    if acov[0] <= 0.0:                      # constant series -> no fluctuation
        return 0.5, 0.0, 0
    rho = acov / acov[0]

    tau = 0.5
    W = 0
    for t in range(1, n):
        tau += rho[t]
        W = t
        # Once the running estimate is non-positive the window condition
        # W >= c*tau is trivially met; this is the i.i.d. / anti-correlated
        # regime and we stop immediately.
        if t >= c * tau:
            break

    tau = max(tau, 0.5)                     # floor at the i.i.d. value
    dtau = tau * math.sqrt(2.0 * (2 * W + 1) / n)
    return float(tau), float(dtau), int(W)


def stat_error(series):
    """Autocorrelation-corrected (mean, error) of a Markov-chain series.

    err = sqrt(2 * tau_int / N) * std(series)  (std with ddof=1).
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = x.size
    mean = float(x.mean())
    if n < 2:
        return mean, 0.0
    tau, _, _ = tau_int(x)
    std = float(x.std(ddof=1))
    err = math.sqrt(2.0 * tau / n) * std
    return mean, float(err)


def _selftest():
    """AR(1): a_t = phi a_{t-1} + eta_t has rho(t) = phi^|t|, hence
    tau_int = 0.5 + sum_{t>=1} phi^t = (1 + phi) / (2 (1 - phi))."""
    rng = np.random.default_rng(20260801)
    ok = True
    for phi in (0.8, 0.9, 0.95):
        tau_true = (1.0 + phi) / (2.0 * (1.0 - phi))
        n = 400_000
        eta = rng.standard_normal(n + 2000)
        a = np.empty_like(eta)
        a[0] = eta[0]
        for i in range(1, a.size):          # simple, dependency-free AR(1)
            a[i] = phi * a[i - 1] + eta[i]
        a = a[2000:]                         # drop transient
        tau, dtau, W = tau_int(a)
        mean, err = stat_error(a)
        rel = abs(tau - tau_true) / tau_true
        good = rel < 0.15
        ok &= good
        print(f"AR(1) phi={phi:.2f}: tau_int={tau:8.3f} +- {dtau:6.3f} "
              f"(W={W:4d})  exact={tau_true:8.3f}  rel.dev={rel*100:5.2f}% "
              f"{'OK' if good else '<-- FAIL'}")
        print(f"           mean={mean:+.4f}  stat_err={err:.4f}")
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
