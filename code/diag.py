"""Multi-chain convergence diagnostics (split rank R-hat, between-chain variance, holding) used by rt_cmcd_phi4.py.

chain_diagnostics(trace, tau, acc=None):
  trace (n_post, C) post-burn values of one observable, tau = integrated autocorrelation time used for the SE
  (per transition; the caller passes max(per-chain, global)), acc (n_post, C) boolean accept trace (optional).
  * n_post, n_post / tau
  * split-R-hat, rank-normalised (Vehtari, Gelman, Simpson, Carpenter, Buerkner 2021, bulk version): every chain is
    split in halves, all draws are replaced by normal scores of their pooled (average) ranks, then the classic
    potential scale reduction sqrt(var_plus / W) on the 2C half-chains.
  * between-chain variance of the chain means vs the tau-predicted one, Var_pi(f) tau / n_post (ratio ~ 1 when the
    chains are long enough and tau is right; >> 1 flags stuck chains or an underestimated tau).
  * max holding / n_post (from acc), flags: max holding > n_post / 10, R-hat > 1.01, constant_trace (any chain
    whose post-burn trace never changes, e.g. a frozen sign sector).
"""
import math

import numpy as np


def _normal_scores(x):
    from scipy.stats import norm, rankdata
    r = rankdata(x.reshape(-1), method='average').reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size + 0.25))


def split_rhat_rank(trace):
    x = np.asarray(trace, dtype=np.float64)
    n = x.shape[0] // 2
    if n < 2:
        return None
    halves = np.concatenate([x[:n], x[n:2 * n]], axis=1)           # (n, 2C)
    if np.all(halves == halves.flat[0]):
        return None
    z = _normal_scores(halves)
    m = z.mean(0)
    W = z.var(0, ddof=1).mean()
    B_over_n = m.var(ddof=1)
    if W <= 0:
        return math.inf
    var_plus = (n - 1) / n * W + B_over_n
    return float(math.sqrt(var_plus / W))


def max_holding(acc):
    a = np.asarray(acc, dtype=bool)
    best = 0
    for c in range(a.shape[1]):
        run = longest = 0
        for v in a[:, c]:
            run = 0 if v else run + 1
            longest = max(longest, run)
        best = max(best, longest)
    return int(best)


def chain_diagnostics(trace, tau, acc=None):
    x = np.asarray(trace, dtype=np.float64)
    n, C = x.shape
    cm = x.mean(0)
    var = float(x.var())
    pred = var * tau / n if n > 0 else float('nan')
    out = dict(n_post=int(n), n_chains=int(C), n_post_over_tau=float(n / tau) if tau > 0 else None,
               rhat_rank=split_rhat_rank(x),
               between_chain_var=float(cm.var(ddof=1)) if C > 1 else None,
               tau_pred_between_var=float(pred),
               between_over_pred=float(cm.var(ddof=1) / pred) if C > 1 and pred > 0 else None)
    const = np.all(x == x[:1], axis=0) if n > 0 else np.zeros(C, bool)
    out['n_chains_constant'] = int(const.sum())          # chains whose post-burn trace never changes
    if acc is not None:
        mh = max_holding(acc)
        out['max_holding'] = mh
        out['max_holding_over_npost'] = mh / n
    flags = []
    if out.get('max_holding_over_npost') is not None and out['max_holding_over_npost'] > 0.1:
        flags.append('max_holding>n_post/10')
    if out['rhat_rank'] is not None and out['rhat_rank'] > 1.01:
        flags.append('rhat>1.01')
    if out['n_chains_constant'] > 0:
        flags.append('constant_trace')   # e.g. frozen sign sector: a tau of 1 from a constant trace must not pass
    out['flags'] = flags
    return out
