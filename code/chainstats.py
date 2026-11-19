"""Shared chain statistics for the revision comparisons (DW, phi4, LJ13).

Conventions (identical for every system and method):
  * traces have shape (transitions, chains) per scalar observable, burn-in already removed;
  * tau_int: Sokal self-consistent window (c=5) on the chain-averaged normalised autocovariance;
  * within-run effective count  n_eff = chains * transitions / tau_int;
  * between-run effective count n_eff_between = Var_pi(O) / Var_runs(mean_O), using the exact or reference
    variance Var_pi(O) when it is known, otherwise the pooled sample variance;
  * cost per effective sample = cost per transition * tau_int (paths, force evaluations, or seconds).
A round-trip transition costs two paths (reverse + forward); a path-IMH transition costs one path.
"""
import math

import numpy as np


def acf(trace):
    """Chain-averaged normalised autocovariance of a (T, C) trace."""
    x = np.asarray(trace, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    n = x.shape[0]
    f = np.fft.rfft(x, 2 * n, axis=0)
    ac = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(1)
    if ac[0] <= 0:
        return np.ones(1)
    return ac / ac[0]


def tau_int(trace, c=5.0):
    """Sokal integrated autocorrelation time; returns (tau, window). tau=1 for a constant trace."""
    rho = acf(trace)
    if rho.size == 1:
        return 1.0, 0
    tau = 1.0
    for m in range(1, rho.size):
        tau = 1.0 + 2.0 * rho[1:m + 1].sum()
        if m >= c * tau:
            return float(max(tau, 1e-12)), m
    return float(max(tau, 1e-12)), rho.size - 1


def observable_summary(trace, exact_mean=None, exact_var=None, cost_per_transition=None):
    """Summary of one observable traced over (T, C)."""
    x = np.asarray(trace, dtype=np.float64)
    T, C = x.shape
    tau, window = tau_int(x)
    var = float(x.var()) if exact_var is None else float(exact_var)
    n_eff = T * C / tau
    out = dict(mean=float(x.mean()), var=float(x.var()), tau_int=tau, window=window, n_eff=n_eff,
               se=math.sqrt(var / n_eff) if n_eff > 0 else float('nan'))
    if exact_mean is not None:
        out['exact_mean'] = float(exact_mean)
        out['error'] = out['mean'] - float(exact_mean)
        out['z'] = out['error'] / out['se'] if out['se'] > 0 else float('nan')
    if cost_per_transition:
        for k, v in cost_per_transition.items():
            out[f'{k}_per_eff_sample'] = v * tau
    return out


def between_runs(run_means, exact_var=None, pooled_var=None, cost_per_run=None):
    """N_eff from the spread of independent run means (end-to-end, includes burn-in and bias)."""
    m = np.asarray(run_means, dtype=np.float64)
    if m.size < 2:
        return dict(n_runs=int(m.size))
    v_runs = float(m.var(ddof=1))
    v = exact_var if exact_var is not None else pooled_var
    out = dict(n_runs=int(m.size), mean=float(m.mean()), sd_runs=math.sqrt(v_runs))
    if v is not None and v_runs > 0:
        out['n_eff_between'] = float(v) / v_runs
        if cost_per_run:
            for k, c in cost_per_run.items():
                out[f'{k}_per_eff_sample_between'] = c / out['n_eff_between']
    return out


def kish_ess(log_w):
    lw = np.asarray(log_w, dtype=np.float64)
    lw = lw[np.isfinite(lw)]
    if lw.size == 0:
        return 0.0
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (w ** 2).sum())


def snis_mean(values, log_w):
    lw = np.asarray(log_w, dtype=np.float64)
    ok = np.isfinite(lw)
    w = np.exp(lw[ok] - lw[ok].max())
    return float((w * np.asarray(values, dtype=np.float64)[ok]).sum() / w.sum())


def holding_times(accept):
    """Rejection-run statistics of a (T, C) boolean accept trace: lengths of maximal runs of consecutive
    rejections (the chain holding one state), pooled over chains."""
    a = np.asarray(accept, dtype=bool)
    runs = []
    for c in range(a.shape[1]):
        r = 0
        for v in a[:, c]:
            if v:
                if r:
                    runs.append(r)
                r = 0
            else:
                r += 1
        if r:
            runs.append(r)
    runs = np.asarray(runs if runs else [0])
    T, C = a.shape
    held = lambda L_: float(runs[runs >= L_].sum() / (T * C))
    return dict(mean_run=float(runs.mean()), q50=float(np.quantile(runs, .5)), q90=float(np.quantile(runs, .9)),
                q99=float(np.quantile(runs, .99)), max_run=int(runs.max()),
                frac_time_in_runs_ge10=held(10), frac_time_in_runs_ge50=held(50))


def relaxation_curve(trace, n_points=200):
    """Chain-averaged observable vs transition (log-spaced subsample), for relaxation from a biased start."""
    x = np.asarray(trace, dtype=np.float64)
    T = x.shape[0]
    idx = np.unique(np.round(np.geomspace(1, T, n_points)).astype(int) - 1)
    return dict(t=(idx + 1).tolist(), mean=x[idx].mean(1).tolist(),
                se=(x[idx].std(1) / math.sqrt(x.shape[1])).tolist())


def burnin_check(series, burn, early_keys, early=0.2, n_blocks=20, z_cut=3.0):
    """Burn-in diagnostics of one run. series: name -> (T, C) trace over the full chain (from transition 1).
    halves: first vs second half of the post-burn chain for every series; z = (m1 - m2) / hypot(se1, se2) with
      the tau-based SE of each half (observable_summary).
    early: for early_keys, the chain-averaged value in n_blocks equal blocks of w transitions over the first
      `early` fraction of all transitions; z = (block mean - post-burn mean) / se_block,
      se_block = sd_post / sqrt(C) * sqrt(min(1, tau_post / w)), tau_post from the post-burn chain.
      t_settle = end of the last block with |z| >= z_cut (0 if none; None if the last early block is still off).
    ok: every early key settled, burn >= t_settle, and every |z_half| < z_cut. ok is False for fewer than 4
    post-burn transitions or any non-finite value. With 2 early keys x 20 blocks at |z| >= 3, about 10 % of
    perfectly mixed runs get t_settle > 0 (still ok when <= burn) and about 0.5 % get None."""
    T, C = next(iter(series.values())).shape
    n_post = T - burn
    h = n_post // 2
    out = dict(burn=int(burn), n_post=int(n_post), halves={}, early={})
    nonfinite = [k for k, x in series.items() if not np.all(np.isfinite(np.asarray(x, dtype=np.float64)))]
    if n_post < 4 or nonfinite:
        out.update(t_settle=None, zmax_halves=float('nan'), ok=False, nonfinite=nonfinite)
        return out
    zmax = 0.0
    for k, x in series.items():
        if h < 4:
            continue
        sa, sb = observable_summary(x[burn:burn + h]), observable_summary(x[burn + h:burn + 2 * h])
        se = math.hypot(sa['se'], sb['se'])
        z = (sa['mean'] - sb['mean']) / se if se > 0 else 0.0
        zmax = max(zmax, abs(z))
        out['halves'][k] = dict(mean1=sa['mean'], mean2=sb['mean'], se1=sa['se'], se2=sb['se'], z=z,
                                tau1=sa['tau_int'], tau2=sb['tau_int'])
    n_early = max(int(early * T), 1)
    w = max(n_early // n_blocks, 1)
    settles = []
    for k in early_keys:
        x = np.asarray(series[k], dtype=np.float64)
        mu, sd = float(x[burn:].mean()), float(x[burn:].std())
        tau = tau_int(x[burn:])[0] if n_post > 1 else 1.0
        se = sd / math.sqrt(C) * math.sqrt(min(1.0, tau / w))
        ends, means, zs = [], [], []
        for i in range(0, n_early - w + 1, w):
            m = float(x[i:i + w].mean())
            ends.append(i + w)
            means.append(m)
            zs.append((m - mu) / se if se > 0 else 0.0)
        bad = [j for j, z in enumerate(zs) if abs(z) >= z_cut]
        if not bad:
            t_settle = 0
        elif bad[-1] + 1 < len(ends):
            t_settle = ends[bad[-1]]
        else:
            t_settle = None
        settles.append(t_settle)
        out['early'][k] = dict(t_end=ends, mean=means, z=zs, post_mean=mu, post_sd=sd, tau_post=tau, block=w,
                               se_block=se, t_settle=t_settle)
    settled = all(t is not None for t in settles)
    out['t_settle'] = max(settles) if settled else None
    out['zmax_halves'] = zmax
    out['ok'] = bool(settled and burn >= out['t_settle'] and zmax < z_cut)
    return out


def burnin_line(bc):
    """One-line text summary of burnin_check()."""
    return (f"burn-in: burn={bc['burn']} t_settle={bc['t_settle']} zmax_halves={bc['zmax_halves']:.2f} ok={bc['ok']} "
            + ' '.join(f"{k}:{v['mean1']:.4g}/{v['mean2']:.4g}(z{v['z']:+.1f})" for k, v in bc['halves'].items()))
