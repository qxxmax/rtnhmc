r"""CLI benchmark driver for 2D spin/scalar models -> SCHEMA json+npz.

    python run_spin.py --theory ising --L 32 --params T=2.269 --algo wolff \
                       --n-samples 20000 --burn 2000

Theories : ising | xy | phi4 (2D only; 3D is out of scope for this module)
Algos    : metro | wolff | hmc (hmc is phi4 only)
Params   : `--params key=value ...`; T for ising/xy, kappa and lam for phi4.

Output follows `common\SCHEMA.md` v1: one json + one npz per run under
`algo_bench\results\{theory}2d\`, named `L{L}[x{L2}]_{algo}_{paramtag}[_tag]`.
(The module brief called this directory `results\spin\`; SCHEMA.md's
`results\{theory}{dim}d\` layout wins so the cross-module report generator
finds these runs next to the gauge ones.  Override with `--out-dir`.)

Observables (same functional forms for all three theories, so they can be
compared directly):

    abs_m  = <|m|>                              m = per-site magnetisation
    chi    = V * ( <m^2> - <|m|>^2 )            (no beta factor)
    binder = 1 - <m^4> / (3 <m^2>^2)
    energy = <E>/V  (ising/xy) or <S>/V (phi4, see phi4.py for the convention)

Errors
------
`abs_m` and `energy` are plain series means, so they use
`common\wolff_tau.stat_error` (Wolff automatic windowing) directly.  `chi` and
`binder` are nonlinear functions of several means, so they use a blocked
jackknife with block length 4 * max(tau_int) (at least 20 blocks); the
`tau_int` reported for them is that of the underlying m^2 series.

Cost accounting
---------------
`cost.sweeps` counts one *sweep-equivalent* per update iteration:

    metro (all theories) : 1 per lattice sweep (V attempted single-site moves)
    wolff (ising/xy)     : 1 per single-cluster flip, as required by the module
                           brief.  A cluster touches <|C|> sites, not V, so
                           this favours Wolff whenever <|C|> < V; the honest
                           site-normalised count is reported separately as
                           `extra.site_sweep_equiv` = (total flipped sites)/V,
                           together with `extra.mean_cluster_frac` = <|C|>/V.
    wolff (phi4)         : 1 + n_cluster per iteration (one Metropolis sweep
                           plus n_cluster cluster flips; the cluster move alone
                           is not ergodic, see phi4.py).
    hmc                  : 0 sweeps; `cost.grad_evals` = trajectories *
                           (n_leap + 1) force evaluations.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:                                        # SCHEMA: torch.set_num_threads(8)
    import torch
    torch.set_num_threads(8)
except Exception:                           # torch is optional here (numpy code)
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from common.wolff_tau import stat_error, tau_int          # noqa: E402

from ising import Ising2D, exact_energy_torus             # noqa: E402
from phi4 import Phi4_2D                                  # noqa: E402
from xy import XY2D                                       # noqa: E402


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def _blocked_jackknife(cols, func, tau, min_blocks=20):
    """Jackknife over contiguous blocks of length 4*tau for func(list_of_means)."""
    n = cols[0].size
    b = max(1, int(math.ceil(4.0 * max(tau, 0.5))))
    if n // b < min_blocks:
        b = max(1, n // min_blocks)
    nb = n // b
    if nb < 2:
        return float(func([float(c.mean()) for c in cols])), 0.0
    tot = [c[:nb * b].reshape(nb, b).sum(axis=1) for c in cols]
    grand = [float(t.sum()) for t in tot]
    n_used = nb * b
    full = float(func([g / n_used for g in grand]))
    jk = np.array([func([(grand[k] - tot[k][i]) / (n_used - b)
                         for k in range(len(cols))]) for i in range(nb)])
    err = math.sqrt((nb - 1.0) / nb * float(((jk - full) ** 2).sum()))
    return full, err


def _obs(mean, err, tau, exact=None):
    d = {"mean": float(mean), "err": float(err), "tau_int": float(tau),
         "exact": None, "dev_pct": None}
    if exact is not None:
        d["exact"] = float(exact)
        if exact != 0.0:
            d["dev_pct"] = float(100.0 * (mean - exact) / abs(exact))
    return d


def summarize(abs_m, m2, m4, en, V, energy_exact=None):
    """Build the SCHEMA `observables` block from the raw measurement series."""
    am_mean, am_err = stat_error(abs_m)
    am_tau, _, _ = tau_int(abs_m)
    en_mean, en_err = stat_error(en)
    en_tau, _, _ = tau_int(en)
    m2_tau, _, _ = tau_int(m2)

    chi, chi_err = _blocked_jackknife(
        [m2, abs_m], lambda mu: V * (mu[0] - mu[1] ** 2), max(m2_tau, am_tau))
    bin_, bin_err = _blocked_jackknife(
        [m2, m4], lambda mu: 1.0 - mu[1] / (3.0 * mu[0] ** 2), m2_tau)

    return {
        "abs_m": _obs(am_mean, am_err, am_tau),
        "chi": _obs(chi, chi_err, m2_tau),
        "binder": _obs(bin_, bin_err, m2_tau),
        "energy": _obs(en_mean, en_err, en_tau, energy_exact),
    }


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------

def _alloc(n):
    return (np.empty(n), np.empty(n), np.empty(n), np.empty(n))


def _run_local(model, algo, n_samples, burn, n_cluster, tune):
    """Shared burn-in + measurement loop for metro / wolff.

    Returns (series tuple, sweeps, extra dict).
    """
    am, m2, m4, en = _alloc(n_samples)
    is_phi4 = isinstance(model, Phi4_2D)
    # phi4's cluster move only flips signs, so it must be interleaved with a
    # Metropolis sweep to be ergodic (see phi4.py).
    use_metro = (algo == "metro") or (is_phi4 and algo == "wolff")
    use_cluster = (algo == "wolff")
    csize = np.zeros(n_samples) if use_cluster else None
    hop = np.empty(n_samples) if is_phi4 else None
    msg = np.empty(n_samples) if is_phi4 else None     # signed m (sector / tunnelling)
    n_flips = n_cluster if is_phi4 else 1

    def iteration():
        acc = model.metropolis_sweep() if use_metro else 0.0
        tot = 0
        if use_cluster:
            for _ in range(n_flips):
                tot += model.wolff_cluster()
        return acc, tot / n_flips

    for i in range(burn):
        acc, _ = iteration()
        if tune and use_metro and (i + 1) % 50 == 0 and i < 0.8 * burn:
            model.tune_delta(acc)

    acc_sum = 0.0
    for i in range(n_samples):
        acc, mean_c = iteration()
        acc_sum += acc
        if use_cluster:
            csize[i] = mean_c
        am[i], m2[i], m4[i], en[i] = model.measure()
        if hop is not None:
            hop[i] = model.hop_density()
            msg[i] = float(model.phi.mean())

    per_iter = (1 if use_metro else 0) + (n_flips if use_cluster else 0)
    sweeps = (burn + n_samples) * per_iter

    extra = {}
    if use_metro:
        extra["metro_acceptance"] = acc_sum / n_samples
        extra["metro_delta"] = float(getattr(model, "delta", 0.0))
    if csize is not None:
        extra["mean_cluster_size"] = float(csize.mean())
        extra["mean_cluster_frac"] = float(csize.mean()) / model.V
        extra["site_sweep_equiv"] = float(csize.sum() * n_flips) / model.V
    if hop is not None:
        extra["_hop_series"] = hop
        extra["_m_series"] = msg
    return (am, m2, m4, en), sweeps, extra, csize


def run(theory, L, L2, params, algo, n_samples, burn, seed=20260801,
        n_cluster=2, dt=0.1, n_leap=10, delta=1.0, tag=None, out_dir=None,
        quiet=False):
    shape = (L, L2)
    V = L * L2
    t0 = time.time()
    energy_exact = None
    grad_evals = 0
    algo_params = {}

    if theory == "ising":
        T = float(params["T"])
        model = Ising2D(shape, T, seed=seed)
        pdict = {"T": T, "J": 1.0}
        ptag = "T%g" % T
        energy_exact = exact_energy_torus(L, L2, T)
    elif theory == "xy":
        T = float(params["T"])
        model = XY2D(shape, T, seed=seed, delta=delta)
        pdict = {"T": T, "J": 1.0}
        ptag = "T%g" % T
    elif theory == "phi4":
        kappa = float(params["kappa"])
        lam = float(params.get("lam", 0.022))
        model = Phi4_2D(shape, kappa, lam, seed=seed, delta=delta)
        pdict = {"kappa": kappa, "lam": lam}
        ptag = "k%g" % kappa
    else:
        raise ValueError("unknown theory " + theory)

    if algo == "hmc":
        if theory != "phi4":
            raise ValueError("hmc is implemented for phi4 only")
        am, m2, m4, en = _alloc(n_samples)
        hop = np.empty(n_samples)
        msg = np.empty(n_samples)
        for _ in range(burn):
            model.hmc_step(dt, n_leap)
        model.n_acc_hmc = model.n_try_hmc = 0
        for i in range(n_samples):
            model.hmc_step(dt, n_leap)
            am[i], m2[i], m4[i], en[i] = model.measure()
            hop[i] = model.hop_density()
            msg[i] = float(model.phi.mean())
        series = (am, m2, m4, en)
        sweeps = 0
        grad_evals = (burn + n_samples) * (n_leap + 1)
        algo_params = {"dt": dt, "n_leap": n_leap, "traj_len": dt * n_leap}
        extra = {"acceptance": model.acceptance, "_hop_series": hop, "_m_series": msg}
        csize = None
    elif algo in ("metro", "wolff"):
        series, sweeps, extra, csize = _run_local(
            model, algo, n_samples, burn, n_cluster,
            tune=(theory in ("xy", "phi4")))
        if algo == "wolff" and theory == "phi4":
            algo_params["n_cluster"] = n_cluster
        if "metro_delta" in extra and theory in ("xy", "phi4"):
            algo_params["delta"] = float(model.delta)
    else:
        raise ValueError("unknown algo " + algo)

    am, m2, m4, en = series
    hop = extra.pop("_hop_series", None)
    msg = extra.pop("_m_series", None)
    wall = time.time() - t0
    obs = summarize(am, m2, m4, en, V, energy_exact)
    if theory == "phi4":
        extra["u_density_ref_conv"] = obs["energy"]["mean"] - model.lam
        # hopping density h for thermodynamic integration: d log Z / d kappa = 2 V <h>
        h_mean, h_err = stat_error(hop)
        h_tau, _, _ = tau_int(hop)
        obs["hop"] = _obs(h_mean, h_err, h_tau)

    result = {
        "schema": 1,
        "theory": theory,
        "dim": 2,
        "shape": [L, L2],
        "params": pdict,
        "algorithm": algo,
        "algo_params": algo_params,
        "device": "cpu",
        "seed": seed,
        "n_samples": n_samples,
        "burn": burn,
        "cost": {"wall_s": wall, "sweeps": sweeps,
                 "grad_evals": grad_evals, "model_evals": 0},
        "observables": obs,
        "extra": extra,
    }

    if out_dir is None:
        out_dir = ROOT / "results" / (theory + "2d")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "L%d" % L + ("x%d" % L2 if L2 != L else "")
    stem += "_%s_%s" % (algo, ptag)
    if tag:
        stem += "_" + tag
    with open(out_dir / (stem + ".json"), "w") as f:
        json.dump(result, f, indent=2)
    arrays = {"abs_m": am, "m2": m2, "m4": m4, "energy": en}
    if csize is not None:
        arrays["cluster_size"] = csize
    if hop is not None:
        arrays["hop"] = hop
    if msg is not None:
        arrays["m"] = msg
    np.savez(out_dir / (stem + ".npz"), **arrays)

    if not quiet:
        print("[done] %s 2d %dx%d %s algo=%s n=%d wall=%.1fs" %
              (theory, L, L2, ptag, algo, n_samples, wall), flush=True)
        for k in ("abs_m", "chi", "binder", "energy"):
            o = obs[k]
            line = "   %-7s %+.6f +- %.6f  tau=%.2f" % (k, o["mean"], o["err"],
                                                        o["tau_int"])
            if o["exact"] is not None:
                line += "  exact=%+.6f  dev=%+.3f%%" % (o["exact"], o["dev_pct"])
            print(line, flush=True)
        if "acceptance" in extra:
            print("   hmc acceptance %.3f" % extra["acceptance"], flush=True)
        if "mean_cluster_frac" in extra:
            print("   <|C|>/V %.4f  site_sweep_equiv %.0f"
                  % (extra["mean_cluster_frac"], extra["site_sweep_equiv"]),
                  flush=True)
    return result


def _parse_params(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError("--params entries must be key=value, got " + it)
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="2D spin/scalar model benchmark")
    ap.add_argument("--theory", choices=("ising", "xy", "phi4"), required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--L2", type=int, default=0, help="second extent (default L)")
    ap.add_argument("--params", nargs="+", required=True,
                    help="e.g. T=2.269   or   kappa=0.3 lam=0.022")
    ap.add_argument("--algo", choices=("metro", "wolff", "hmc"), required=True)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--n-cluster", type=int, default=2,
                    help="phi4 wolff: cluster flips per Metropolis sweep")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-leap", type=int, default=10)
    ap.add_argument("--delta", type=float, default=1.0,
                    help="initial Metropolis step (auto-tuned in burn-in)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    run(args.theory, args.L, args.L2 or args.L, _parse_params(args.params),
        args.algo, args.n_samples, args.burn, seed=args.seed,
        n_cluster=args.n_cluster, dt=args.dt, n_leap=args.n_leap,
        delta=args.delta, tag=args.tag, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
