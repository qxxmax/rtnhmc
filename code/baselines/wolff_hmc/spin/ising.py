r"""2D Ising model: checkerboard Metropolis, Wolff single cluster, and the
exact finite-torus energy (Kaufman / Ferdinand-Fisher) used as a reference.

Conventions
-----------
    E = - sum_{<xy>} s_x s_y ,      s_x = +-1,   J = 1,   beta = 1/T
    weight = exp(-beta E)

`energy` is reported per site, e = E / V, so the ferromagnetic ground state has
e = -2 on a 2D torus.  The critical point is T_c = 2 / ln(1 + sqrt(2))
= 2.269185...

Magnetisation observables use the per-site magnetisation m = (1/V) sum_x s_x:
    abs_m  = < |m| >
    chi    = V * ( <m^2> - <|m|>^2 )
    binder = 1 - <m^4> / (3 <m^2>^2)

Algorithms
----------
Metropolis: two-colour (checkerboard) sweep.  Sites of one colour have only
neighbours of the other colour, so within a colour the single-spin Metropolis
acceptances are independent and can be applied simultaneously without
violating detailed balance.  One "sweep" = both colours = V attempted flips.

Wolff single cluster: the embedded-Ising construction of `lattice.grow_cluster`
with unit weights, so the bond probability is the textbook

    p = 1 - exp(-2 beta J) = 1 - exp(-2 beta) .

The whole cluster is flipped unconditionally (rejection free).

Exact finite-volume reference
-----------------------------
`exact_energy_torus(m, n, T)` evaluates the Kaufman closed form for the
partition function of an m x n periodic Ising lattice as written by
Ferdinand and Fisher, Phys. Rev. 185 (1969) 832,

    Z = (1/2) (2 sinh 2K)^{mn/2} [Z_1 + Z_2 + Z_3 + Z_4] ,
    Z_1 = prod_{r=0}^{n-1} 2 cosh(m gamma_{2r+1} / 2) ,
    Z_2 = prod_{r=0}^{n-1} 2 sinh(m gamma_{2r+1} / 2) ,
    Z_3 = prod_{r=0}^{n-1} 2 cosh(m gamma_{2r}   / 2) ,
    Z_4 = prod_{r=0}^{n-1} 2 sinh(m gamma_{2r}   / 2) ,

with cosh(gamma_l) = cosh^2(2K)/sinh(2K) - cos(pi l / n) for l >= 1 and
gamma_0 = 2K + ln(tanh K) (which is negative above T_c, making Z_4 negative).
All products are accumulated in logs with explicit signs, and the energy
<E> = -d ln Z / dK is taken by central difference.  `_selftest` checks the
formula against brute-force enumeration of the 4x4 torus (2^16 states).
"""
from __future__ import annotations

import math
import random

import numpy as np

from lattice import checkerboard, grow_cluster, neighbor_table

__all__ = ["Ising2D", "TC", "exact_energy_torus", "onsager_energy",
           "onsager_magnetisation"]

TC = 2.0 / math.log(1.0 + math.sqrt(2.0))


# --------------------------------------------------------------------------
# exact references
# --------------------------------------------------------------------------

def _log_partition_torus(m, n, K):
    """ln Z of the m x n periodic Ising lattice at coupling K = J/(kT)."""
    c2 = math.cosh(2.0 * K)
    s2 = math.sinh(2.0 * K)
    base = c2 * c2 / s2

    l = np.arange(2 * n, dtype=np.float64)
    arg = base - np.cos(np.pi * l / n)
    gamma = np.arccosh(np.clip(arg, 1.0, None))
    gamma[0] = 2.0 * K + math.log(math.tanh(K))

    x = 0.5 * m * gamma
    ax = np.abs(x)
    log2cosh = ax + np.log1p(np.exp(-2.0 * ax))
    with np.errstate(divide="ignore"):
        log2sinh = ax + np.log1p(-np.exp(-2.0 * ax))
    sgn_sinh = np.where(x < 0.0, -1.0, 1.0)

    odd = (np.arange(2 * n) % 2 == 1)
    even = ~odd
    logs = np.array([log2cosh[odd].sum(), log2sinh[odd].sum(),
                     log2cosh[even].sum(), log2sinh[even].sum()])
    signs = np.array([1.0, float(np.prod(sgn_sinh[odd])),
                      1.0, float(np.prod(sgn_sinh[even]))])

    finite = np.isfinite(logs)
    mx = logs[finite].max()
    tot = float(np.sum(signs[finite] * np.exp(logs[finite] - mx)))
    ln_bracket = mx + math.log(tot)

    return 0.5 * m * n * math.log(2.0 * s2) + ln_bracket - math.log(2.0)


def exact_energy_torus(m, n, T, h=1e-5):
    """Exact energy per site of the m x n periodic Ising lattice at T."""
    K = 1.0 / T
    dlnZ = (_log_partition_torus(m, n, K + h)
            - _log_partition_torus(m, n, K - h)) / (2.0 * h)
    return -dlnZ / (m * n)


def onsager_energy(T):
    """Onsager thermodynamic-limit energy per site (for orientation only)."""
    from scipy.special import ellipk
    K = 1.0 / T
    k = 2.0 * math.sinh(2.0 * K) / math.cosh(2.0 * K) ** 2
    # scipy's ellipk takes the parameter m = k^2
    return -(1.0 / math.tanh(2.0 * K)) * (
        1.0 + (2.0 * math.tanh(2.0 * K) ** 2 - 1.0) * (2.0 / math.pi)
        * ellipk(k * k))


def onsager_magnetisation(T):
    """Onsager spontaneous magnetisation (0 above T_c)."""
    if T >= TC:
        return 0.0
    return (1.0 - math.sinh(2.0 / T) ** -4) ** 0.125


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class Ising2D:
    """2D Ising model on an n0 x n1 torus at temperature T."""

    def __init__(self, shape, T, seed=20260801, start="hot"):
        self.shape = (int(shape[0]), int(shape[1]))
        self.V = self.shape[0] * self.shape[1]
        self.T = float(T)
        self.beta = 1.0 / self.T
        self.rng = np.random.default_rng(seed)
        self._rnd = random.Random(seed ^ 0x5EED).random
        if start == "cold":
            self.s = np.ones(self.shape, dtype=np.int8)
        else:
            self.s = self.rng.integers(0, 2, self.shape).astype(np.int8) * 2 - 1
        self._even = checkerboard(self.shape)
        self._nbr = neighbor_table(self.shape)
        self._inc = bytearray(self.V)
        self._p_add = 1.0 - math.exp(-2.0 * self.beta)

    # -- updates -----------------------------------------------------------

    def metropolis_sweep(self):
        """One checkerboard sweep; returns the acceptance fraction."""
        s = self.s
        n_acc = 0
        for parity in (True, False):
            h = (np.roll(s, 1, 0) + np.roll(s, -1, 0)
                 + np.roll(s, 1, 1) + np.roll(s, -1, 1)).astype(np.float64)
            dE = 2.0 * s * h
            r = self.rng.random(self.shape)
            acc = (dE <= 0.0) | (r < np.exp(-self.beta * np.minimum(dE, 700.0)))
            flip = acc & (self._even if parity else ~self._even)
            s[flip] = -s[flip]
            n_acc += int(flip.sum())
        return n_acc / self.V

    def wolff_cluster(self):
        """Flip one Wolff cluster; returns its size."""
        flat = self.s.ravel()
        sgn = flat.tolist()
        seed = int(self.rng.integers(self.V))
        members = grow_cluster(seed, sgn, None, 0.0, self._nbr, self._rnd,
                               self._inc, p_const=self._p_add)
        flat[np.asarray(members, dtype=np.intp)] = -sgn[seed]
        return len(members)

    # -- measurement -------------------------------------------------------

    def measure(self):
        """(|m|, m^2, m^4, energy per site) of the current configuration."""
        s = self.s
        m = float(s.sum()) / self.V
        bonds = float((s.astype(np.int64)
                       * (np.roll(s, -1, 0).astype(np.int64)
                          + np.roll(s, -1, 1).astype(np.int64))).sum())
        e = -bonds / self.V
        m2 = m * m
        return abs(m), m2, m2 * m2, e


# --------------------------------------------------------------------------
# self-test of the exact formula
# --------------------------------------------------------------------------

def _brute_force_energy(L, T):
    """Energy per site of the L x L torus by exact enumeration (L=4 only)."""
    V = L * L
    bits = np.arange(1 << V, dtype=np.int64)
    spins = ((bits[:, None] >> np.arange(V)[None, :]) & 1).astype(np.int8)
    spins = spins * 2 - 1
    spins = spins.reshape(-1, L, L).astype(np.int64)
    E = -(spins * (np.roll(spins, -1, 1) + np.roll(spins, -1, 2))).sum(axis=(1, 2))
    E = E.astype(np.float64)
    logw = -E / T
    logw -= logw.max()
    w = np.exp(logw)
    return float((E * w).sum() / w.sum()) / V


def _selftest():
    ok = True
    print("Ferdinand-Fisher exact torus energy vs brute force (4x4):")
    for T in (1.5, 2.0, TC, 2.5, 4.0):
        a = exact_energy_torus(4, 4, T)
        b = _brute_force_energy(4, T)
        good = abs(a - b) < 1e-6
        ok &= good
        print("  T=%6.4f  FF=%+.9f  brute=%+.9f  diff=%.2e  %s"
              % (T, a, b, abs(a - b), "OK" if good else "<-- FAIL"))
    print("Large-L limit vs Onsager (L=256):")
    for T in (2.0, 2.5):
        a = exact_energy_torus(256, 256, T)
        b = onsager_energy(T)
        good = abs(a - b) < 2e-3
        ok &= good
        print("  T=%.3f  FF(256)=%+.6f  Onsager=%+.6f  diff=%.2e  %s"
              % (T, a, b, abs(a - b), "OK" if good else "<-- FAIL"))
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
