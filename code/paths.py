"""Forward and reverse NHMC paths and their work.

W(path) = log q0(u) + sum_t [log q_F,t - log q_R,t] + S(x_T) = log Q(path) - log P~(path), the same recorded work for a
forward path u -> x_T and for a reverse record x_T -> u, so E_Q[e^{-W}] = Z. Partial paths (stages s+1..T) give the
work W_s of a truncated round trip through the intermediate field x_s (no log q0 term for s > 0).
"""
import torch

from phi4 import phi4_action
from kernels import base_draw, base_logprob


def fwd_path(kernel, u, T, kappa, lam):
    s = base_logprob(kernel, u)
    x = u
    for t in range(1, T + 1):
        x, lf, lb = kernel.forward_step(x, t, T, kappa=kappa)
        s = s + lf - lb
    return x, s + phi4_action(x.squeeze(1), lam=lam, kappa=kappa)


def rev_path(kernel, xT, T, kappa, lam):
    """Reverse record from x_T (differentiable unless called under no_grad)."""
    s = phi4_action(xT.squeeze(1), lam=lam, kappa=kappa)
    x = xT
    for t in range(T, 0, -1):
        x, lf, lb = kernel.reverse_step(x, t, T, kappa=kappa)
        s = s + lf - lb
    return x, s + base_logprob(kernel, x)


@torch.no_grad()
def rt_step(kernel, x, T, kappa, lam):
    """One round-trip MH update (pi-invariant for any parameters, Theorem 2)."""
    u, Wx = rev_path(kernel, x, T, kappa, lam)
    y, Wy = fwd_path(kernel, u, T, kappa, lam)
    acc = torch.rand(x.size(0), device=x.device) < torch.exp(torch.clamp(Wx - Wy, max=0.0))
    return torch.where(acc.view(-1, 1, 1, 1), y, x), acc


@torch.no_grad()
def draw(kernel, n, L, T, kappa, lam, device):
    """L = int (square L x L) or (L, L2) (rectangular)."""
    u, _ = base_draw(kernel, (n, 1) + (tuple(L) if isinstance(L, (tuple, list)) else (L, L)), device, kappa)
    return fwd_path(kernel, u, T, kappa, lam)


def rev_partial(k, x, T, s_stop, kappa, lam):
    """Reverse record x_T -> u_s through k.reverse_step; W_s = S(x_T) + sum_{t>s}(log qF_t - log qR_t),
    plus log q0(u) if s_stop == 0 (full record work)."""
    acc = phi4_action(x.squeeze(1), lam=lam, kappa=kappa)
    for t in range(T, s_stop, -1):
        x, lf, lb = k.reverse_step(x, t, T, kappa=kappa)
        acc = acc + lf - lb
    if s_stop == 0:
        acc = acc + base_logprob(k, x)
    return x, acc


def fwd_partial(k, u, T, s_stop, kappa, lam):
    acc = base_logprob(k, u) if s_stop == 0 else torch.zeros(u.size(0), dtype=u.dtype, device=u.device)
    x = u
    for t in range(s_stop + 1, T + 1):
        x, lf, lb = k.forward_step(x, t, T, kappa=kappa)
        acc = acc + lf - lb
    return x, acc + phi4_action(x.squeeze(1), lam=lam, kappa=kappa)


@torch.no_grad()
def paths(k, n, L, kappa, lam, T, device):
    x, logq0 = base_draw(k, (n, 1, L, L), device, kappa)
    logP = logq0
    for t in range(1, T + 1):
        x, lf, lb = k.forward_step(x, t, T, kappa=kappa)
        logP = logP + lf - lb
    return x, logP + phi4_action(x.squeeze(1), lam=lam, kappa=kappa)
