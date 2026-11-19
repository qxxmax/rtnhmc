"""Markov-chain statistics on the conventions the paper's tables were built on.

Table 2 (LJ13 chain comparison) and Table 8 (DW round-trip) are the only rows
that are chain quantities rather than single-shot path quantities, and nothing
in our evaluation pipeline produced them: we had no chain runner, no integrated
autocorrelation time, and no force-cost accounting. The estimators below are
transcribed from the script that produced Table 2,
`scripts/lj13_matched_force_chains.py`, so that the numbers are comparable
rather than merely similar. The two places where a reasonable implementation
would differ, and therefore the two worth naming:

  * the autocorrelation is normalized by the number of terms at each lag
    (`np.arange(n, 0, -1)`), not by n, so the tail is unbiased but noisy;
  * the Sokal window closes at the first lag with `lag > c * max(2*tau, 1)`
    with c = 5, and tau is floored at 0.5.

`n_eff` uses the pooled sample count over the mean per-chain tau, and the
standard error divides the pooled ddof=1 standard deviation by sqrt(n_eff).
"""

import json
import math

import numpy as np

SOKAL_C = 5.0
BURN_FRACTION = 0.1


def autocorr_fft(values, max_lag):
    """Normalized autocorrelation of one chain, via FFT."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    x = x - x.mean()
    var = np.dot(x, x) / max(n, 1)
    lag = min(max_lag, max(n - 1, 0))
    if n == 0 or var <= 0.0 or lag <= 0:
        out = np.zeros(lag + 1, dtype=np.float64)
        if out.size:
            out[0] = 1.0
        return out
    size = 1 << (2 * n - 1).bit_length()
    freq = np.fft.rfft(x, size)
    acf = np.fft.irfft(freq * np.conjugate(freq), size)[:n]
    acf = acf / np.arange(n, 0, -1)
    return acf[:lag + 1] / acf[0]


def sokal_tau(values, c=SOKAL_C):
    """Integrated autocorrelation time with a self-consistent window."""
    rho = autocorr_fft(values, min(len(values) // 4, 4096))
    tau = 0.5
    for lag in range(1, len(rho)):
        tau += rho[lag]
        if lag > c * max(2.0 * tau, 1.0):
            break
    return float(max(tau, 0.5))


def observable_stats(values, n_chains, force_evals_used):
    """Chain-ensemble summary of one observable, shaped (steps, chains)."""
    taus = np.asarray([sokal_tau(values[:, c]) for c in range(n_chains)])
    tau_mean = float(taus.mean())
    n_eff = float(values.size / (2.0 * tau_mean))
    return {
        'mean': float(values.mean()),
        'stderr_autocorr': float(
            values.reshape(-1).std(ddof=1) / math.sqrt(max(n_eff, 1.0))
        ),
        'tau_int_mean': tau_mean,
        'tau_int_per_chain': taus.tolist(),
        'tau_int_spread': float(taus.max() - taus.min()),
        'n_eff': n_eff,
        'n_eff_per_1e6_force_evals': float(n_eff * 1.0e6 / force_evals_used),
    }


def cost_normalized_tau(values, n_chains, paths_per_transition,
                        force_evals_per_transition):
    """Table 8's autocorrelation columns for the DW round-trip ensemble.

    Two conventions differ from the LJ13 table and are easy to carry over by
    mistake. The per-chain taus are reduced by their *median*, not their mean,
    and the reported numbers are rescaled into two cost currencies: one
    transition of the round trip spends two paths where path-IMH spends one, so
    comparing raw transition counts would credit the round trip with sampling
    it did not pay for.
    """
    taus = np.asarray([sokal_tau(values[:, c]) for c in range(n_chains)])
    median = float(np.median(taus))
    return {
        'tau_int_median': median,
        'tau_int_per_chain': taus.tolist(),
        'paths_per_transition': int(paths_per_transition),
        'force_evals_per_transition': int(force_evals_per_transition),
        'path_cost_tau_int': median * paths_per_transition,
        'force_cost_tau_int': median * paths_per_transition
        * force_evals_per_transition,
    }


def summarize_trace(name, trace, force_evals_per_step_per_chain,
                    force_budget=0, burn_fraction=BURN_FRACTION):
    """Table 2-style summary of one chain ensemble at a matched force budget.

    `trace` holds `energy`, `min_dist` and `accept`, each shaped
    (steps, chains). Matching the budget is what makes the round-trip row
    comparable: it costs two paths per transition, so it is truncated to half
    as many steps rather than being given twice the compute.
    """
    energy = np.asarray(trace['energy'], dtype=np.float64)
    available_steps, n_chains = energy.shape
    cost = int(force_evals_per_step_per_chain)
    used_steps = available_steps
    if force_budget:
        used_steps = min(available_steps, force_budget // (n_chains * cost))
    burn = int(round(used_steps * burn_fraction))
    if used_steps - burn < 100:
        raise ValueError(f"insufficient post-burn trace for {name}")

    energy = energy[:used_steps][burn:]
    min_dist = np.asarray(trace['min_dist'], dtype=np.float64)[:used_steps][burn:]
    accept = np.asarray(trace['accept'], dtype=bool)[:used_steps][burn:]
    force_evals_used = used_steps * n_chains * cost

    return {
        'method': name,
        'n_chains': n_chains,
        'available_steps_per_chain': int(available_steps),
        'used_steps_per_chain': int(used_steps),
        'burn_in_steps': int(burn),
        'post_burn_samples': int(energy.size),
        'acceptance_rate_post_burn': float(accept.mean()),
        'force_evals_per_step_per_chain': cost,
        'force_evals_used': int(force_evals_used),
        'energy': observable_stats(energy, n_chains, force_evals_used),
        'min_pair_distance': observable_stats(min_dist, n_chains,
                                              force_evals_used),
    }


def force_evals_per_path(num_steps, n_leapfrog, momentum_law_force_calls=0):
    """Counted target-force evaluations for one path.

    Each stage spends n_leapfrog + 1 forces on the integrator, plus whatever
    the momentum law reads to build its features. The force-equivariant law
    reads two per stage, which is why the paper counts 616 and not 504 for its
    56-stage, 8-leapfrog LJ13 configuration.
    """
    return int(num_steps * (n_leapfrog + 1 + momentum_law_force_calls))


def write_summary(path, payload):
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2)
