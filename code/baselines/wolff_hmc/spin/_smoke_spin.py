r"""Smoke test for the spin module.  ASCII only, CPU only, no GPU, no kills.

    python _smoke_spin.py            # full test (~5 min)
    python _smoke_spin.py --fast     # reduced statistics, structure only

Checks
------
(0) The exact finite-torus Ising energy (Kaufman / Ferdinand-Fisher) against
    brute-force enumeration of the 4x4 lattice.
(a) 2D Ising L=16 at T=2.0 and T=2.5: energy against the EXACT L=16 torus
    value, |m| against the loose finite-size expectation, and Metropolis vs
    Wolff agreement within 4 sigma.  At T_c = 2.269 the headline number is the
    autocorrelation ratio tau_int(|m|)_metro / tau_int(|m|)_wolff.
(b) 2D XY L=16 at T=0.8: Metropolis vs Wolff agreement within 4 sigma.
(c) 2D phi4 16x8, lambda=0.022, kappa in {0.25, 0.3}: HMC vs Metropolis vs
    Brower-Tamayo Wolff agreement within 4 sigma on abs_m and energy, and
    consistency with the reference HMC results of the free-energy project
    (obs_16x8_k0.25.json, obs_16x8_k0.3.json).  Those files use the action
    convention without the constant lambda per site, so the comparison is
    energy_here - lambda vs their u_density.

Runs write into results\_smoke\ so they never collide with benchmark output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_spin                                          # noqa: E402
from ising import TC, _brute_force_energy, exact_energy_torus  # noqa: E402
from ising import onsager_magnetisation                  # noqa: E402

REF_DIR = Path(r"D:\thermocode\free_energy_hmc_phi4_latest")
OUT = HERE.parents[0] / "results" / "_smoke"
LAM = 0.022

_fails = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  " + detail) if detail else ""), flush=True)
    if not ok:
        _fails.append(name)
    return ok


def nsig(a, b):
    """Deviation of two independent estimates (mean, err) in sigma."""
    s = math.hypot(a[1], b[1])
    return abs(a[0] - b[0]) / s if s > 0 else float("inf")


def val(res, key):
    o = res["observables"][key]
    return (o["mean"], o["err"])


def go(theory, L, L2, params, algo, n, burn, seed=20260801, **kw):
    return run_spin.run(theory, L, L2, params, algo, n, burn, seed=seed,
                        out_dir=OUT, tag="smoke", quiet=True, **kw)


# --------------------------------------------------------------------------

def test_exact_formula():
    print("\n(0) exact finite-torus Ising energy vs brute force (4x4)")
    for T in (2.0, TC, 2.5):
        a = exact_energy_torus(4, 4, T)
        b = _brute_force_energy(4, T)
        check("FF vs brute force T=%.4f" % T, abs(a - b) < 1e-6,
              "FF=%+.9f brute=%+.9f" % (a, b))


def test_ising(n, burn):
    print("\n(a) 2D Ising L=16")
    for T in (2.0, 2.5):
        ex = exact_energy_torus(16, 16, T)
        mw = go("ising", 16, 16, {"T": T}, "wolff", n, burn)
        mm = go("ising", 16, 16, {"T": T}, "metro", n, burn)
        print("    T=%.3f  exact energy (16x16 torus) = %+.6f" % (T, ex))
        for tag, r in (("metro", mm), ("wolff", mw)):
            e, am = val(r, "energy"), val(r, "abs_m")
            print("      %-5s energy %+.6f +- %.6f (%.1f sig from exact)"
                  "   abs_m %.6f +- %.6f  tau(|m|)=%.2f"
                  % (tag, e[0], e[1], abs(e[0] - ex) / e[1], am[0], am[1],
                     r["observables"]["abs_m"]["tau_int"]))
            check("ising T=%.1f %s energy vs exact" % (T, tag),
                  abs(e[0] - ex) <= 4.0 * e[1] + 2e-3,
                  "dev=%+.4f%%" % (100 * (e[0] - ex) / abs(ex)))
        # loose |m| expectation: spontaneous magnetisation below T_c, a broad
        # finite-size crossover window above it.
        for tag, r in (("metro", mm), ("wolff", mw)):
            am = val(r, "abs_m")[0]
            if T < TC:
                m0 = onsager_magnetisation(T)
                check("ising T=%.1f %s abs_m vs Onsager m_spont" % (T, tag),
                      abs(am - m0) / m0 < 0.03,
                      "%.4f vs %.4f (%+.2f%%)" % (am, m0, 100 * (am - m0) / m0))
            else:
                check("ising T=%.1f %s abs_m in finite-size window" % (T, tag),
                      0.15 < am < 0.45, "%.4f in (0.15, 0.45)" % am)
        for key in ("abs_m", "energy"):
            s = nsig(val(mm, key), val(mw, key))
            check("ising T=%.1f metro vs wolff %s" % (T, key), s < 4.0,
                  "%.2f sigma" % s)

    print("\n    headline: tau_int(|m|) at T_c = %.6f (L=16)" % TC)
    cw = go("ising", 16, 16, {"T": TC}, "wolff", n, burn)
    cm = go("ising", 16, 16, {"T": TC}, "metro", n, burn)
    tw = cw["observables"]["abs_m"]["tau_int"]
    tm = cm["observables"]["abs_m"]["tau_int"]
    fc = cw["extra"]["mean_cluster_frac"]
    print("      metro tau_int(|m|) = %8.3f  (per lattice sweep)" % tm)
    print("      wolff tau_int(|m|) = %8.3f  (per single-cluster flip)" % tw)
    print("      RATIO metro/wolff  = %8.2f" % (tm / tw))
    print("      <|C|>/V = %.4f -> site-normalised wolff tau = %.3f,"
          " work-matched ratio = %.2f" % (fc, tw * fc, tm / (tw * fc)))
    for key in ("abs_m", "energy"):
        s = nsig(val(cm, key), val(cw, key))
        check("ising T_c metro vs wolff %s" % key, s < 4.0, "%.2f sigma" % s)
    return {"tau_metro": tm, "tau_wolff": tw, "ratio": tm / tw,
            "cluster_frac": fc}


def test_xy(n, burn):
    print("\n(b) 2D XY L=16 at T=0.8")
    w = go("xy", 16, 16, {"T": 0.8}, "wolff", n, burn)
    m = go("xy", 16, 16, {"T": 0.8}, "metro", n, burn)
    for tag, r in (("metro", m), ("wolff", w)):
        e, am = val(r, "energy"), val(r, "abs_m")
        print("      %-5s abs_m %.6f +- %.6f  energy %+.6f +- %.6f"
              "  tau(|m|)=%.2f" % (tag, am[0], am[1], e[0], e[1],
                                   r["observables"]["abs_m"]["tau_int"]))
    print("      tau_int(|m|) ratio metro/wolff = %.2f"
          % (m["observables"]["abs_m"]["tau_int"]
             / w["observables"]["abs_m"]["tau_int"]))
    for key in ("abs_m", "energy"):
        s = nsig(val(m, key), val(w, key))
        check("xy T=0.8 metro vs wolff %s" % key, s < 4.0, "%.2f sigma" % s)


def test_phi4(n, burn):
    print("\n(c) 2D phi4 16x8, lambda=%.3f" % LAM)
    table = {}
    for kappa in (0.25, 0.3):
        p = {"kappa": kappa, "lam": LAM}
        res = {
            "hmc": go("phi4", 16, 8, p, "hmc", n, burn),
            "metro": go("phi4", 16, 8, p, "metro", n, burn),
            "wolff": go("phi4", 16, 8, p, "wolff", n, burn, n_cluster=2),
        }
        ref_path = REF_DIR / ("obs_16x8_k%g.json" % kappa)
        with open(ref_path) as f:
            ref = json.load(f)
        # reference action convention omits the constant lambda per site
        ref_e = (ref["u_density"] + LAM, ref["u_density_err"])
        ref_m = (ref["abs_mag"], ref["abs_mag_err"])

        print("\n    kappa = %g" % kappa)
        print("      %-14s %-24s %-24s" % ("source", "abs_m", "energy = <S>/V"))
        for tag in ("hmc", "metro", "wolff"):
            am, e = val(res[tag], "abs_m"), val(res[tag], "energy")
            print("      %-14s %10.6f +- %-9.6f %+10.6f +- %-9.6f"
                  % (tag, am[0], am[1], e[0], e[1]))
        print("      %-14s %10.6f +- %-9.6f %+10.6f +- %-9.6f"
              % ("reference", ref_m[0], ref_m[1], ref_e[0], ref_e[1]))
        print("      (reference u_density = %+.6f in the convention without"
              " the constant lambda)" % ref["u_density"])

        pairs = (("hmc", "metro"), ("hmc", "wolff"), ("metro", "wolff"))
        for a, b in pairs:
            for key in ("abs_m", "energy"):
                s = nsig(val(res[a], key), val(res[b], key))
                check("phi4 k=%g %s vs %s %s" % (kappa, a, b, key), s < 4.0,
                      "%.2f sigma" % s)
        for tag in ("hmc", "metro", "wolff"):
            for key, rv in (("abs_m", ref_m), ("energy", ref_e)):
                s = nsig(val(res[tag], key), rv)
                check("phi4 k=%g %s vs reference %s" % (kappa, tag, key),
                      s < 4.0, "%.2f sigma" % s)
        table[kappa] = (res, ref_m, ref_e)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        n_is, b_is, n_xy, b_xy, n_p4, b_p4 = 4000, 500, 4000, 500, 4000, 500
    else:
        n_is, b_is, n_xy, b_xy, n_p4, b_p4 = 40000, 4000, 40000, 4000, 30000, 3000

    print("=" * 72)
    print("  spin module smoke test  (CPU, %s statistics)"
          % ("reduced" if args.fast else "full"))
    print("=" * 72)
    test_exact_formula()
    tau = test_ising(n_is, b_is)
    test_xy(n_xy, b_xy)
    test_phi4(n_p4, b_p4)

    print("\n" + "=" * 72)
    print("  Ising L=16 at T_c: tau_int(|m|) metro/wolff = %.2f"
          % tau["ratio"])
    if _fails:
        print("  SMOKE FAILED (%d): %s" % (len(_fails), "; ".join(_fails)))
    else:
        print("  SMOKE PASSED")
    print("=" * 72)
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
