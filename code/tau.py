"""The integrated autocorrelation time rule of the paper and jackknife helpers.

Per chain (after burn-in, 10 % of the chain): g = max(Sokal self-consistent window with c = 5, batch means with
50 batches), where g = 1 + 2 sum_t rho(t); tau_int = g / 2. For a chain set the reported value is the median of
tau_int over the chains (four chains unless stated otherwise); its error is the delete-one-chain jackknife.
A chain set is doubled in length until every chain passes chain_criteria (n_post >= 1000 g, n_post >= 20 x the
longest holding run, Sokal window < n_post / 50, sign not frozen); otherwise tau_int is a lower bound.
"""
import numpy as np


def sokal_single(x, c=5.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 1.0, 0
    rho = ac / ac[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * rho[1:m + 1].sum()
        if m >= c * tau:
            return float(tau), m
    return float(tau), n - 1


def batch_means_tau(x, nb=50):
    x = np.asarray(x, dtype=np.float64)
    b = x.size // nb
    if b < 2 or x.var() <= 0:
        return None
    bm = x[:b * nb].reshape(nb, b).mean(1)
    return float(b * bm.var(ddof=1) / x.var())


def holding_runs(acc):
    runs, r = [], 0
    for v in acc:
        if v:
            if r:
                runs.append(r)
            r = 0
        else:
            r += 1
    if r:
        runs.append(r)
    return np.asarray(runs if runs else [0])


def chain_criteria(trM, trS, acc, n, burn_fraction, tau_mult, hold_mult, win_ratio, use_sign=True):
    """per chain: Sokal tau / window (|M| and sign, single-chain), max holding on the post-burn part of the first n
    transitions; returns (all_ok, per-chain list of dicts with the failing criteria)."""
    burn = int(burn_fraction * n)
    out, ok_all = [], True
    for c in range(trM.shape[1]):
        post_n = n - burn
        taus, wins = [], []
        for tr in ((trM[burn:n, c], trS[burn:n, c].astype(np.float64)) if use_sign else (trM[burn:n, c],)):
            t, w = sokal_single(tr)
            taus.append(t)
            wins.append(w)
        mh = int(holding_runs(acc[burn:n, c]).max())
        fail = []
        sg = trS[burn:n, c]
        if use_sign and sg.size and np.all(sg == sg[0]):          # frozen sector: the single-chain sign ACF is degenerate (tau 1)
            fail.append('sign_frozen')
        if post_n < tau_mult * max(taus):
            fail.append(f'n_post<{tau_mult:g}tau')
        if post_n < hold_mult * mh:
            fail.append(f'n_post<{hold_mult:g}maxhold')
        if max(wins) * win_ratio > post_n:
            fail.append(f'window>n_post/{win_ratio:g}')
        ok_all &= not fail
        out.append(dict(n=n, tau_max=max(taus), window_max=max(wins), max_hold=mh, failing=fail))
    return ok_all, out


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
    """same rule as sokal_single, vectorised (returns g = 1 + 2 sum rho and window)."""
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
    """same as batch_means_tau."""
    x = np.asarray(x, dtype=np.float64)
    b = x.size // nb
    if b < 2 or x.var() <= 0:
        return None
    bm = x[:b * nb].reshape(nb, b).mean(1)
    return float(b * bm.var(ddof=1) / x.var())


def tau_seg(x):
    g, _ = sokal_g(x)
    b = bm_g(x)
    return max(g, b or 0.0) / 2



def tau_int_chains(x, burn_fraction=0.1):
    """Paper statistic from a trace of shape (transitions, chains): per chain tau_int = max(Sokal, BM) / 2 on the
    post-burn part (sokal_single, batch_means_tau: the functions that wrote tau_sokal / tau_batch_means into the chain
    files), then the median over chains and its delete-one-chain jackknife error.
    Returns (median, jackknife error, per-chain values)."""
    x = np.asarray(x, dtype=np.float64)
    burn = int(burn_fraction * x.shape[0])
    t = np.array([max(sokal_single(x[burn:, c])[0], batch_means_tau(x[burn:, c]) or 0.0) / 2
                  for c in range(x.shape[1])])
    med, err, _ = jk_delete_one(t, np.median)
    return float(med), err, t
