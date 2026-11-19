"""The three path updates for a frozen NHMC model, one transition each, for a batch of independent chains.

  path-IMH  propose a fresh forward path; the chain carries the work W of the path that produced its state;
            accept with min(1, exp(W_x - W_y)).
  RT        full round trip: draw a reverse path from the current state x to u = x_0, a forward path from the same u
            to y, accept with min(1, exp(W_x - W_y)); the log q0(u) terms cancel.
  pRT(s)    truncated round trip through the intermediate field x_s (1 <= s <= T - 1): only stages s+1..T are redrawn,
            and the works are the partial works W_s of the two half paths.

These functions are the transition of code/exp_chains.py (PhiStepper.propose and the accept step of main) written out
as small functions; they use the same calls in the same order, so with the same random state they give the same
chains. Non-finite log ratios are rejections (NaN) or acceptances (+inf).
"""
import math

import torch

from kernels import base_draw
from paths import fwd_partial, rev_partial


def _accept(x, y, Wx, Wy):
    la = torch.nan_to_num((Wx - Wy).double(), nan=-math.inf, posinf=math.inf, neginf=-math.inf)
    ac = torch.log(torch.rand_like(la)) < la
    x = torch.where(ac.view(-1, *([1] * (x.dim() - 1))), y, x)
    return x, ac, torch.exp(torch.clamp(la, max=0.0))


@torch.no_grad()
def start_states(k, n, L, L2, T, kappa, lam, device):
    """Chain starts: endpoints of fresh forward paths and their works (the IMH record)."""
    u, _ = base_draw(k, (n, 1, L, L2), device, kappa)
    return fwd_partial(k, u, T, 0, kappa, lam)


@torch.no_grad()
def path_imh_step(k, x, W, T, kappa, lam):
    """One path-IMH transition. Returns (x, W, accepted, acceptance probability)."""
    u, _ = base_draw(k, x.shape, x.device, kappa)
    y, Wy = fwd_partial(k, u, T, 0, kappa, lam)
    x, ac, al = _accept(x, y, W, Wy)
    W = torch.where(ac, Wy, W)
    return x, W, ac, al


@torch.no_grad()
def truncated_round_trip_step(k, x, T, s, kappa, lam):
    """One round trip through stage s (s = 0: full round trip).
    Returns (x, accepted, acceptance probability, W_s of the old record, W_s of the new record)."""
    u, Wx = rev_partial(k, x, T, s, kappa, lam)
    y, Wy = fwd_partial(k, u, T, s, kappa, lam)
    x, ac, al = _accept(x, y, Wx, Wy)
    return x, ac, al, Wx, Wy


@torch.no_grad()
def round_trip_step(k, x, T, kappa, lam):
    """One full round trip (pRT with s = 0)."""
    return truncated_round_trip_step(k, x, T, 0, kappa, lam)
