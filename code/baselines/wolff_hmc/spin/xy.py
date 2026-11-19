r"""2D XY model: checkerboard Metropolis and Wolff single cluster with a
random reflection axis (embedded Ising).

Conventions
-----------
    E = - sum_{<xy>} cos(theta_x - theta_y) ,   J = 1,   beta = 1/T
    weight = exp(-beta E)

`energy` is per site, e = E / V (ground state e = -2).  The Berezinskii-
Kosterlitz-Thouless transition sits at T_KT ~ 0.893.

Magnetisation uses the per-site vector magnetisation
m_vec = (1/V) sum_x (cos theta_x, sin theta_x):
    abs_m  = < |m_vec| >
    chi    = V * ( <|m_vec|^2> - <|m_vec|>^2 )
    binder = 1 - <|m_vec|^4> / (3 <|m_vec|^2>^2)
(the same functional forms as the Ising / phi4 modules, for cross-theory
consistency; note the O(2)-specific Binder normalisation 2 - <m^4>/<m^2>^2 is
NOT used here).

Metropolis
----------
Two-colour sweep with a uniform proposal theta -> theta + delta * U(-1,1).
`delta` is auto-tuned during burn-in towards ~50 % acceptance and then frozen,
so the measurement phase uses a fixed, detailed-balance-respecting kernel.

Wolff with a random reflection axis
-----------------------------------
Draw a unit vector r = (cos alpha, sin alpha) with alpha uniform in [0, pi).
Split each spin into components parallel and perpendicular to r,

    s_x = p_x r + s_x^perp ,        p_x = s_x . r = cos(theta_x - alpha) ,

so that

    s_x . s_y = p_x p_y + s_x^perp . s_y^perp .

The proposed move flips the parallel component only,

    s_x -> s_x - 2 (s_x . r) r  ,   i.e.  theta_x -> 2 alpha + pi - theta_x ,

which sends p_x -> -p_x and leaves s_x^perp untouched.  Conditioned on the
frozen perpendicular components, the signs eps_x = sign(p_x) are therefore
distributed as an inhomogeneous ferromagnetic Ising model with couplings
beta |p_x| |p_y|, and the Fortuin-Kasteleyn bond probability between aligned
neighbours is

    p_add = 1 - exp(-2 beta p_x p_y)        (only when p_x p_y > 0).

Growing one such cluster and flipping it is rejection free.
"""
from __future__ import annotations

import math
import random

import numpy as np

from lattice import checkerboard, grow_cluster, neighbor_table

__all__ = ["XY2D", "T_KT"]

T_KT = 0.893


class XY2D:
    """2D XY model on an n0 x n1 torus at temperature T."""

    def __init__(self, shape, T, seed=20260801, start="hot", delta=1.0):
        self.shape = (int(shape[0]), int(shape[1]))
        self.V = self.shape[0] * self.shape[1]
        self.T = float(T)
        self.beta = 1.0 / self.T
        self.rng = np.random.default_rng(seed)
        self._rnd = random.Random(seed ^ 0x5EED).random
        if start == "cold":
            self.theta = np.zeros(self.shape, dtype=np.float64)
        else:
            self.theta = self.rng.random(self.shape) * (2.0 * math.pi)
        self.delta = float(delta)
        self._even = checkerboard(self.shape)
        self._nbr = neighbor_table(self.shape)
        self._inc = bytearray(self.V)

    # -- updates -----------------------------------------------------------

    def metropolis_sweep(self):
        """One checkerboard sweep; returns the acceptance fraction."""
        th = self.theta
        n_acc = 0
        for parity in (True, False):
            c, s = np.cos(th), np.sin(th)
            hc = (np.roll(c, 1, 0) + np.roll(c, -1, 0)
                  + np.roll(c, 1, 1) + np.roll(c, -1, 1))
            hs = (np.roll(s, 1, 0) + np.roll(s, -1, 0)
                  + np.roll(s, 1, 1) + np.roll(s, -1, 1))
            prop = th + self.delta * (2.0 * self.rng.random(self.shape) - 1.0)
            dE = -((np.cos(prop) - c) * hc + (np.sin(prop) - s) * hs)
            r = self.rng.random(self.shape)
            acc = (dE <= 0.0) | (r < np.exp(-self.beta * np.minimum(dE, 700.0)))
            m = acc & (self._even if parity else ~self._even)
            th[m] = prop[m]
            n_acc += int(m.sum())
        return n_acc / self.V

    def tune_delta(self, acc, target=0.5):
        """Nudge the Metropolis step size towards `target` acceptance.

        Only ever called during burn-in; delta is frozen for measurements.
        """
        fac = min(2.0, max(0.5, math.sqrt(max(acc, 1e-3) / target)))
        self.delta = float(np.clip(self.delta * fac, 1e-3, 2.0 * math.pi))

    def wolff_cluster(self):
        """Flip one embedded-Ising cluster; returns its size."""
        th = self.theta
        alpha = float(self.rng.random()) * math.pi
        proj = np.cos(th - alpha)
        sgn = np.where(proj >= 0.0, 1, -1).ravel().tolist()
        w = np.abs(proj).ravel().tolist()
        seed = int(self.rng.integers(self.V))
        members = grow_cluster(seed, sgn, w, 2.0 * self.beta, self._nbr,
                               self._rnd, self._inc)
        idx = np.asarray(members, dtype=np.intp)
        flat = th.ravel()
        flat[idx] = np.mod(2.0 * alpha + math.pi - flat[idx], 2.0 * math.pi)
        return len(members)

    # -- measurement -------------------------------------------------------

    def measure(self):
        """(|m|, |m|^2, |m|^4, energy per site) of the current configuration."""
        th = self.theta
        c, s = np.cos(th), np.sin(th)
        mx = float(c.mean())
        my = float(s.mean())
        m2 = mx * mx + my * my
        e = -float((np.cos(th - np.roll(th, -1, 0))
                    + np.cos(th - np.roll(th, -1, 1))).sum()) / self.V
        return math.sqrt(m2), m2, m2 * m2, e
