"""Exact L = 2 phi^4 reference (periodic, lambda = .022): log Z and moments by quadrature.

S(a,b,c,d) = sum_i v(phi_i) - 4 kappa (a + d)(b + c), v = (1 - 2 lam) phi^2 + lam phi^4 (each bond counted twice by
the roll sum, as targets/phi4.py and nhmc_code/action.py). Two independent quadratures:
  * 4-D trapezoid on [-5.5, 5.5]^4 (n = 101), as v6_data_phi4/tests/test_partial_rt_exact.py;
  * 2-D reduction: A = a + d, B = b + c, Z = int int g(A) g(B) exp(4 kappa A B), g = e^{-v} * e^{-v}.
"""
import math

import numpy as np
import torch


def S_closed(a, b, c, d, kappa, lam=0.022):
    v = lambda f: (1 - 2 * lam) * f ** 2 + lam * f ** 4
    return v(a) + v(b) + v(c) + v(d) - 4 * kappa * (a + d) * (b + c)


def exact_4d(kappa, lam=0.022, n=101, lo=-5.5, hi=5.5):
    g = torch.linspace(lo, hi, n, dtype=torch.float64)
    h = float(g[1] - g[0])
    gb, gc, gd = torch.meshgrid(g, g, g, indexing='ij')
    Z = 0.0
    acc = dict(absM=0.0, M2=0.0, M4=0.0, S=0.0)
    for ga in g:
        S = S_closed(ga, gb, gc, gd, kappa, lam)
        w = torch.exp(-S)
        M = (ga + gb + gc + gd) / 4
        Z += float(w.sum())
        acc['absM'] += float((w * M.abs()).sum()); acc['M2'] += float((w * M ** 2).sum())
        acc['M4'] += float((w * M ** 4).sum()); acc['S'] += float((w * S).sum())
    return dict(logZ=math.log(Z * h ** 4), absM=acc['absM'] / Z, M2=acc['M2'] / Z, M4=acc['M4'] / Z,
                action=acc['S'] / Z / 4, sign=0.0)


def logZ_2d(kappa, lam=0.022, ha=0.02, amax=9.0):
    a = np.arange(-amax, amax + ha / 2, ha)
    e = np.exp(-((1 - 2 * lam) * a ** 2 + lam * a ** 4))
    g = np.convolve(e, e) * ha
    A = 2 * a[0] + ha * np.arange(g.size)
    lw = np.log(np.maximum(g, 1e-300))
    M = lw[:, None] + lw[None, :] + 4 * kappa * A[:, None] * A[None, :]
    mx = M.max()
    return float(mx + np.log(np.exp(M - mx).sum() * ha * ha))


def check_action(model_log_gamma, kappa, lam=0.022):
    x = torch.randn(64, 4, dtype=torch.float64) * 1.5
    S1 = -model_log_gamma(x)
    S2 = S_closed(x[:, 0], x[:, 1], x[:, 2], x[:, 3], kappa, lam)
    return float((S1 - S2).abs().max())
