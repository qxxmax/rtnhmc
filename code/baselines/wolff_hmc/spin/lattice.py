r"""Shared 2D-lattice helpers for the spin module: neighbour tables and the
single-cluster (Wolff) growth routine used by Ising, XY and phi4.

Unified embedded-Ising cluster
------------------------------
All three theories flip a *sign-like* degree of freedom eps_x = +-1 with a
Boltzmann weight that, conditioned on the remaining (eps-invariant) degrees of
freedom, is an inhomogeneous ferromagnetic Ising model

    W({eps}) = exp( sum_{<xy>} c * w_x * w_y * eps_x * eps_y ) ,   w_x >= 0 ,

so the Fortuin-Kasteleyn / Wolff bond-activation probability between two
*aligned* neighbours is

    p_xy = 1 - exp(-2 * c * w_x * w_y) .

Per theory (see the individual modules for the derivations):

    Ising : eps_x = s_x,             w_x = 1,        c = beta
    XY    : eps_x = sign(s_x . r),   w_x = |s_x . r|, c = beta
    phi4  : eps_x = sign(phi_x),     w_x = |phi_x|,  c = 2 * kappa

`grow_cluster` therefore takes (sgn, w, two_c) with two_c = 2*c and uses
p = 1 - exp(-two_c * w_x * w_y).

Implementation notes
--------------------
- BFS with `collections.deque`; the queue holds flat site indices.
- `sgn` and `w` are plain Python lists (built once per cluster with
  `ndarray.tolist()`): scalar element access is ~5x faster than on ndarrays,
  which matters because the growth loop is the hot path.
- Bond coin flips come from a caller-supplied `rnd()` returning U(0,1).  We use
  a dedicated `random.Random` stream for these so the numpy stream that drives
  the field updates stays reproducible independently of cluster sizes.
- `inc` is a caller-owned bytearray membership flag; `grow_cluster` leaves it
  clean (it resets only the sites it set), so it can be reused across calls.
"""
from __future__ import annotations

from collections import deque
from math import exp

import numpy as np

__all__ = ["neighbor_table", "grow_cluster", "checkerboard"]


def neighbor_table(shape):
    """Flat nearest-neighbour index table of a periodic 2D lattice.

    Returns a list of 4-tuples: nbr[i] = (i+x, i-x, i+y, i-y).
    """
    n0, n1 = shape
    idx = np.arange(n0 * n1, dtype=np.int64).reshape(n0, n1)
    cols = [np.roll(idx, -1, axis=0), np.roll(idx, 1, axis=0),
            np.roll(idx, -1, axis=1), np.roll(idx, 1, axis=1)]
    tab = np.stack([c.ravel() for c in cols], axis=1)
    return [tuple(int(v) for v in row) for row in tab]


def checkerboard(shape):
    """Boolean mask of the even sublattice ((x+y) % 2 == 0)."""
    n0, n1 = shape
    x = np.arange(n0)[:, None]
    y = np.arange(n1)[None, :]
    return ((x + y) % 2 == 0)


def grow_cluster(seed, sgn, w, two_c, nbr, rnd, inc, p_const=None):
    """Grow one Wolff cluster by BFS from `seed`; return the member index list.

    Parameters
    ----------
    seed    : flat index of the seed site.
    sgn     : list of +-1 embedded-Ising signs (only equal signs may join).
    w       : list of non-negative bond weights, or None for unit weights.
    two_c   : 2*c in p = 1 - exp(-two_c * w_x * w_y).
    nbr     : neighbour table from `neighbor_table`.
    rnd     : callable returning a U(0,1) float.
    inc     : bytearray of length V used as the membership flag; must be all
              zero on entry and is restored to all zero on exit.
    p_const : if not None, use this constant bond probability and ignore `w`
              (the Ising case, where w == 1 everywhere).

    A bond is tested at most once, always from the endpoint that entered the
    cluster first, which is exactly Wolff's single-cluster construction.
    """
    inc[seed] = 1
    members = [seed]
    queue = deque((seed,))
    s0 = sgn[seed]
    if p_const is not None:
        while queue:
            i = queue.popleft()
            for j in nbr[i]:
                if inc[j] or sgn[j] != s0:
                    continue
                if rnd() < p_const:
                    inc[j] = 1
                    members.append(j)
                    queue.append(j)
    else:
        while queue:
            i = queue.popleft()
            wi = two_c * w[i]
            for j in nbr[i]:
                if inc[j] or sgn[j] != s0:
                    continue
                if rnd() < 1.0 - exp(-wi * w[j]):
                    inc[j] = 1
                    members.append(j)
                    queue.append(j)
    for i in members:
        inc[i] = 0
    return members
