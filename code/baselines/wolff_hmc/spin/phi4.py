r"""2D single-component phi^4 lattice field theory: HMC, site Metropolis, and
the Brower-Tamayo embedded-cluster (Wolff) algorithm.

Action convention
-----------------
Exactly the convention of the free-energy project
(`free_energy_hmc_phi4_latest`), in the Nicoli et al. (PRL 126, 032001) form:

    S[phi] = sum_x [ -2 kappa sum_{mu=1..d} phi_x phi_{x+mu}
                     + phi_x^2 + lambda (phi_x^2 - 1)^2 ]

with weight exp(-S).  Expanding the quartic term,

    phi^2 + lambda (phi^2 - 1)^2 = (1 - 2 lambda) phi^2 + lambda phi^4 + lambda ,

so this differs from the `free_energy_reweighting.py` / `run_observables.py`
implementation (which drops the constant) by exactly lambda per site.  Hence

    energy_here = <S>/V  =  u_density_reference + lambda .

Both numbers are reported: `energy` uses the convention above, and
`extra.u_density_ref_conv` = energy - lambda matches the reference json files.

The force used by HMC is

    -dS/dphi_x = -[ 2 phi_x + 4 lambda phi_x (phi_x^2 - 1)
                    - 2 kappa sum_{mu} (phi_{x+mu} + phi_{x-mu}) ] ,

identical to the reference implementation (whose constant term drops out).

Observables use the per-site mean field m = (1/V) sum_x phi_x:
    abs_m  = < |m| >
    chi    = V * ( <m^2> - <|m|>^2 )
    binder = 1 - <m^4> / (3 <m^2>^2)
matching `run_observables.py` exactly.

Algorithms
----------
HMC
    Leapfrog with unit-mass Gaussian momenta and a Metropolis accept/reject on
    exp(-dH); the integrator is copied step for step from
    `free_energy_reweighting.py` (half-kick sandwich, n_leap = tau/eps).

Site Metropolis
    Two-colour sweep, proposal phi -> phi + delta * U(-1,1).  delta is
    auto-tuned to ~50 % acceptance during burn-in and then frozen.

Brower-Tamayo embedded cluster
    Derivation of the bond probability.  Write phi_x = eps_x |phi_x| with
    eps_x = sign(phi_x).  The single-site part phi_x^2 + lambda(phi_x^2-1)^2
    depends on |phi_x| only, so it is invariant under eps_x -> -eps_x.  The
    only eps-dependent piece of the weight is the hopping term

        exp( +2 kappa sum_{<xy>} phi_x phi_y )
      = exp( +2 kappa sum_{<xy>} |phi_x| |phi_y| eps_x eps_y ) ,

    where <xy> runs over each nearest-neighbour bond once.  Conditioned on the
    magnitudes {|phi_x|} this is an inhomogeneous ferromagnetic Ising model
    with couplings K_xy = 2 kappa |phi_x| |phi_y| >= 0 (for kappa > 0).  The
    Fortuin-Kasteleyn representation of exp(K_xy eps_x eps_y) activates a bond
    between *aligned* neighbours with probability 1 - exp(-2 K_xy), i.e.

        p_add = 1 - exp( -4 kappa |phi_x| |phi_y| )
              = 1 - exp( -4 kappa phi_x phi_y )   when phi_x phi_y > 0,

    and never between anti-aligned ones.  Flipping the resulting single Wolff
    cluster (phi_x -> -phi_x for x in the cluster) is rejection free.

    The cluster move changes only the signs, never the magnitudes, so it is
    NOT ergodic on its own.  As Brower and Tamayo prescribe, it is interleaved
    with an ordinary Metropolis sweep: one `wolff` iteration here is
    1 Metropolis sweep followed by `n_cluster` cluster flips.  Deep in the
    broken phase the bond probability saturates and the cluster degenerates to
    a global sign flip, which is the correct behaviour (it restores tunnelling
    between the two vacua) but means the magnitude decorrelation is carried
    entirely by the Metropolis sweep.
"""
from __future__ import annotations

import math
import random

import numpy as np

from lattice import checkerboard, grow_cluster, neighbor_table

__all__ = ["Phi4_2D"]


class Phi4_2D:
    """2D phi^4 field on an n0 x n1 torus at (kappa, lambda)."""

    def __init__(self, shape, kappa, lam=0.022, seed=20260801, delta=1.0):
        self.shape = (int(shape[0]), int(shape[1]))
        self.V = self.shape[0] * self.shape[1]
        self.kappa = float(kappa)
        self.lam = float(lam)
        self.rng = np.random.default_rng(seed)
        self._rnd = random.Random(seed ^ 0x5EED).random
        self.phi = self.rng.standard_normal(self.shape)
        self.delta = float(delta)
        self._even = checkerboard(self.shape)
        self._nbr = neighbor_table(self.shape)
        self._inc = bytearray(self.V)
        self.n_acc_hmc = 0
        self.n_try_hmc = 0

    # -- action / force ----------------------------------------------------

    def action(self, phi=None):
        """Total action S[phi] in the convention documented above."""
        p = self.phi if phi is None else phi
        p2 = p * p
        site = p2 + self.lam * (p2 - 1.0) ** 2
        hop = p * (np.roll(p, -1, 0) + np.roll(p, -1, 1))
        return float(site.sum() - 2.0 * self.kappa * hop.sum())

    def hop_density(self):
        """h = (1/V) sum_{x,mu} phi_x phi_{x+mu};  d log Z / d kappa = 2 V <h>."""
        p = self.phi
        return float((p * (np.roll(p, -1, 0) + np.roll(p, -1, 1))).sum()) / self.V

    def force(self, phi):
        """-dS/dphi."""
        nb = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
              + np.roll(phi, 1, 1) + np.roll(phi, -1, 1))
        grad = (2.0 * phi + 4.0 * self.lam * phi * (phi * phi - 1.0)
                - 2.0 * self.kappa * nb)
        return -grad

    def _hamiltonian(self, phi, pi):
        return 0.5 * float((pi * pi).sum()) + self.action(phi)

    # -- updates -----------------------------------------------------------

    def hmc_step(self, dt=0.1, n_leap=10):
        """One HMC trajectory; returns True if accepted."""
        pi = self.rng.standard_normal(self.shape)
        h0 = self._hamiltonian(self.phi, pi)

        phi = self.phi + 0.5 * dt * pi
        for _ in range(n_leap - 1):
            pi = pi + dt * self.force(phi)
            phi = phi + dt * pi
        pi = pi + dt * self.force(phi)
        phi = phi + 0.5 * dt * pi

        dh = self._hamiltonian(phi, pi) - h0
        self.n_try_hmc += 1
        if dh <= 0.0 or self.rng.random() < math.exp(-min(dh, 700.0)):
            self.phi = phi
            self.n_acc_hmc += 1
            return True
        return False

    def metropolis_sweep(self):
        """One checkerboard sweep; returns the acceptance fraction."""
        p = self.phi
        lam, kap = self.lam, self.kappa
        n_acc = 0
        for parity in (True, False):
            nb = (np.roll(p, 1, 0) + np.roll(p, -1, 0)
                  + np.roll(p, 1, 1) + np.roll(p, -1, 1))
            prop = p + self.delta * (2.0 * self.rng.random(self.shape) - 1.0)
            p2, q2 = p * p, prop * prop
            dS = ((q2 + lam * (q2 - 1.0) ** 2) - (p2 + lam * (p2 - 1.0) ** 2)
                  - 2.0 * kap * (prop - p) * nb)
            r = self.rng.random(self.shape)
            acc = (dS <= 0.0) | (r < np.exp(-np.minimum(dS, 700.0)))
            m = acc & (self._even if parity else ~self._even)
            p[m] = prop[m]
            n_acc += int(m.sum())
        return n_acc / self.V

    def tune_delta(self, acc, target=0.5):
        """Nudge the Metropolis step size; burn-in only, then frozen."""
        fac = min(2.0, max(0.5, math.sqrt(max(acc, 1e-3) / target)))
        self.delta = float(np.clip(self.delta * fac, 1e-3, 20.0))

    def wolff_cluster(self):
        """Flip one Brower-Tamayo embedded cluster; returns its size."""
        flat = self.phi.ravel()
        sgn = np.where(flat >= 0.0, 1, -1).tolist()
        w = np.abs(flat).tolist()
        seed = int(self.rng.integers(self.V))
        members = grow_cluster(seed, sgn, w, 4.0 * self.kappa, self._nbr,
                               self._rnd, self._inc)
        idx = np.asarray(members, dtype=np.intp)
        flat[idx] = -flat[idx]
        return len(members)

    # -- measurement -------------------------------------------------------

    def measure(self):
        """(|m|, m^2, m^4, S/V) of the current configuration."""
        m = float(self.phi.mean())
        m2 = m * m
        return abs(m), m2, m2 * m2, self.action() / self.V

    @property
    def acceptance(self):
        return self.n_acc_hmc / self.n_try_hmc if self.n_try_hmc else 0.0
