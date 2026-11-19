#!/usr/bin/env python3
"""
Real neural Hamiltonian Monte Carlo (NHMC) for the continuous FAB many-well
double-well target.

This is a faithful continuous-space port of the lattice phi^4 NeuralHMCKernel
(models/phi4/new_kernels.py + kernels.py):

  * base  pi_0 = N(0, I_d)
  * each of L steps:
        p0 ~ N(mu_theta(x, t), sigma_theta(x, t))                (learned momentum)
        (x', p') = leapfrog(x, p0; true force = grad log gamma)  (volume preserving)
        log q_fwd = log N(p0; mu_f, sigma_f)
        log q_bwd = log N(-p'; mu_b, sigma_b)
  * path work  W = log q_0(x_0) + sum_t (log q_fwd - log q_bwd) + S(x_L)
    with S = -log gamma  (so pi propto e^{-S} = gamma).
  * importance weight  w = exp(-W),   Z = E_fwd[w].

MixedSignKernel wraps the base kernel with a per-double-well-coordinate
stochastic sign flip (the continuous analogue of MixedZ2Kernel), enabling
tunnelling between the 2^{d/2} sign modes of the many-well target.

Only momentum/flip parameters are learned; the force is the exact target
gradient, so the leapfrog Jacobian is 1 and everything is reversible.

NOTE (recovery provenance): this file was reconstructed on 2026-07-11 from
Cursor's local editor history (.cursor-server/data/User/History/27e2def7/),
because the working copy had been deleted from disk while the compiled
__pycache__/neural_nhmc_dw.cpython-3*.pyc artifacts and several downstream
scripts (lj13_nhmc.py, train_lj13_nhmc.py, make_lj13_eval_fig.py,
make_lj13_imh.py) still imported it. Content is reproduced verbatim from the
last recorded history snapshot; no logic was changed.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_2PI = math.log(2.0 * math.pi)

# Normalizing constant of the 1D double-well factor (matches the endpoint script)
TARGET_Z_FIRST_DIM = 11784.50927
LOG_Z_FIRST = math.log(TARGET_Z_FIRST_DIM)
LOG_Z_2D = LOG_Z_FIRST + 0.5 * math.log(2.0 * math.pi)


# =========================================================================
# Target
# =========================================================================
class ManyWellTarget:
    """FAB many-well target.  Even coords: double-well; odd coords: N(0,1).

    log gamma(x) = sum_even (-x^4 + 6 x^2 + 0.5 x) + sum_odd (-0.5 x^2)
    """

    def __init__(self, dim: int):
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.n_wells = dim // 2

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        even = x[:, 0::2]
        odd = x[:, 1::2]
        return (-even.pow(4) + 6.0 * even.pow(2) + 0.5 * even).sum(1) \
            - 0.5 * odd.pow(2).sum(1)

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """grad_x log gamma(x)  (used as the leapfrog force).

        Clamped to a wide range purely for numerical safety; the wells sit at
        |x|~2 where the force is O(10), so the clamp is inactive in practice and
        leapfrog stays volume-preserving and reversible regardless.
        """
        f = torch.empty_like(x)
        even = x[:, 0::2]
        odd = x[:, 1::2]
        f[:, 0::2] = -4.0 * even.pow(3) + 12.0 * even + 0.5
        f[:, 1::2] = -odd
        return f.clamp(-1.0e3, 1.0e3)

    def true_log_z(self) -> float:
        return LOG_Z_2D * self.n_wells

    # ---- exact target sampling (rejection on the 1D double-well factor) ----
    def _sample_first_dim(self, n: int, gen: torch.Generator,
                          device, dtype) -> torch.Tensor:
        weights = torch.tensor([0.2, 0.8], dtype=dtype, device=device)
        means = torch.tensor([-1.7, 1.7], dtype=dtype, device=device)
        scales = torch.tensor([0.5, 0.5], dtype=dtype, device=device)
        k = TARGET_Z_FIRST_DIM * 3.0
        out, remaining = [], n
        while remaining > 0:
            m = max(remaining * 12, 4096)
            comp = torch.multinomial(weights, m, replacement=True, generator=gen)
            z = means[comp] + scales[comp] * torch.randn(m, dtype=dtype, device=device, generator=gen)
            log_comp = torch.log(weights) + (-0.5 * ((z[:, None] - means) / scales) ** 2
                                             - torch.log(scales) - 0.5 * LOG_2PI)
            q = torch.logsumexp(log_comp, dim=1)
            u = torch.rand(m, dtype=dtype, device=device, generator=gen)
            tgt = -z.pow(4) + 6.0 * z.pow(2) + 0.5 * z
            mask = tgt > torch.log(u) + math.log(k) + q
            if mask.any():
                new = z[mask][:remaining]
                out.append(new)
                remaining -= new.numel()
        return torch.cat(out)

    def sample(self, n: int, seed: int = 0, device='cpu', dtype=torch.float32) -> torch.Tensor:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.empty(n, self.dim, dtype=dtype, device=device)
        for w in range(self.n_wells):
            x[:, 2 * w] = self._sample_first_dim(n, gen, device, dtype)
            x[:, 2 * w + 1] = torch.randn(n, dtype=dtype, device=device, generator=gen)
        return x


# =========================================================================
# Momentum network (MLP with sinusoidal time embedding)
# =========================================================================
class MomentumNet(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, n_freq: int = 8,
                 out_bias: float = 0.0):
        super().__init__()
        self.n_freq = n_freq
        in_dim = dim + 2 * n_freq
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, out_bias)

    def _temb(self, t_norm: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(1, self.n_freq + 1, device=t_norm.device, dtype=t_norm.dtype)
        ang = 2.0 * math.pi * t_norm[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, x: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self._temb(t_norm)], dim=1))


# =========================================================================
# Continuous Neural HMC kernel (true force leapfrog + learned momentum)
# =========================================================================
class ContinuousNHMCKernel(nn.Module):
    def __init__(self, target: ManyWellTarget, n_leapfrog: int = 5,
                 hidden: int = 128, log_eps_init: float = -2.7):
        super().__init__()
        self.target = target
        self.dim = target.dim
        self.n_leapfrog = n_leapfrog
        # momentum mean starts at 0; momentum log-sigma starts near -0.7 (sigma~0.5)
        self.fwd_mu = MomentumNet(self.dim, hidden, out_bias=0.0)
        self.fwd_logsig = MomentumNet(self.dim, hidden, out_bias=-0.7)
        self.bwd_mu = MomentumNet(self.dim, hidden, out_bias=0.0)
        self.bwd_logsig = MomentumNet(self.dim, hidden, out_bias=-0.7)
        self.log_epsilon = nn.Parameter(torch.tensor(float(log_eps_init)))

    @property
    def epsilon(self):
        return torch.exp(self.log_epsilon).clamp(1e-4, 0.3)

    def _leapfrog(self, x, p, eps):
        f = self.target.force(x)
        p = p + 0.5 * eps * f
        for _ in range(self.n_leapfrog - 1):
            x = x + eps * p
            p = p + eps * self.target.force(x)
        x = x + eps * p
        p = p + 0.5 * eps * self.target.force(x)
        return x, p

    def forward_step(self, x, t, num_steps):
        B = x.size(0)
        t_norm = torch.full((B,), t / num_steps, device=x.device, dtype=x.dtype)
        mu_f = self.fwd_mu(x, t_norm)
        sig_f = torch.exp(self.fwd_logsig(x, t_norm).clamp(-5.0, 2.0))
        dist_f = Normal(mu_f, sig_f)
        p0 = dist_f.rsample()
        x_new, p_new = self._leapfrog(x, p0, self.epsilon)
        log_fwd = dist_f.log_prob(p0).sum(1)

        t_next = torch.full_like(t_norm, (t + 1) / num_steps)
        mu_b = self.bwd_mu(x_new, t_next)
        sig_b = torch.exp(self.bwd_logsig(x_new, t_next).clamp(-5.0, 2.0))
        dist_b = Normal(mu_b, sig_b)
        log_bwd = dist_b.log_prob(-p_new).sum(1)
        return x_new, log_fwd, log_bwd

    @torch.no_grad()
    def reverse_step(self, x, t, num_steps):
        """Recover x_{t-1} from x_t using the backward momentum law and the
        reversibility of leapfrog (volume preserving)."""
        B = x.size(0)
        t_next = torch.full((B,), (t + 1) / num_steps, device=x.device, dtype=x.dtype)
        mu_b = self.bwd_mu(x, t_next)
        sig_b = torch.exp(self.bwd_logsig(x, t_next).clamp(-5.0, 2.0))
        m = Normal(mu_b, sig_b).sample()
        x_prev, _ = self._leapfrog(x, m, self.epsilon)
        return x_prev


# =========================================================================
# Mixed-sign wrapper (per-double-well sign flip; continuous MixedZ2)
# =========================================================================
class MixedSignKernel(nn.Module):
    """Per-even-coordinate independent Bernoulli sign flip before the base step.

    Flip prob rho(t) is learned per step (shared across the n_wells coords).
    The flip action is recorded in the path ratio exactly like MixedZ2Kernel.
    """

    def __init__(self, base: ContinuousNHMCKernel, num_steps: int):
        super().__init__()
        self.base = base
        self.dim = base.dim
        self.n_wells = base.target.n_wells
        self.flip_logits = nn.Parameter(torch.zeros(num_steps + 1))

    def forward_step(self, x, t, num_steps):
        B = x.size(0)
        prob_f = torch.sigmoid(self.flip_logits[t])
        flip = (torch.rand(B, self.n_wells, device=x.device) < prob_f).float()  # [B,n_wells]
        sign = 1.0 - 2.0 * flip
        x_flip = x.clone()
        x_flip[:, 0::2] = x[:, 0::2] * sign

        log_rho_f = (flip * torch.log(prob_f + 1e-8)
                     + (1 - flip) * torch.log(1 - prob_f + 1e-8)).sum(1)

        x_next, log_fwd_b, log_bwd_b = self.base.forward_step(x_flip, t, num_steps)

        prob_b = torch.sigmoid(self.flip_logits[min(t + 1, num_steps)])
        log_rho_b = (flip * torch.log(prob_b + 1e-8)
                     + (1 - flip) * torch.log(1 - prob_b + 1e-8)).sum(1)

        return x_next, log_fwd_b + log_rho_f, log_bwd_b + log_rho_b

    @torch.no_grad()
    def reverse_step(self, x, t, num_steps):
        x_prev = self.base.reverse_step(x, t, num_steps)
        prob = torch.sigmoid(self.flip_logits[t])
        flip = (torch.rand(x.size(0), self.n_wells, device=x.device) < prob).float()
        sign = 1.0 - 2.0 * flip
        out = x_prev.clone()
        out[:, 0::2] = x_prev[:, 0::2] * sign
        return out


# =========================================================================
# Path-work rollout helpers
# =========================================================================
def base_log_prob(x0: torch.Tensor) -> torch.Tensor:
    return Normal(torch.zeros_like(x0), torch.ones_like(x0)).log_prob(x0).sum(1)


class DWModeAwareBase(nn.Module):
    """Factorized mode-aware base matching the ManyWell geometry.

    The ManyWell target is a product of independent 1D factors: EVEN coords are
    a tilted double well with minima near +/- 1.7 (unequal mass, +side heavier),
    ODD coords are exactly N(0,1).  The default N(0,I) base sits at 0 on every
    even coord -- exactly between the two wells -- so the NHMC annealing must
    create all 2^{n_wells} modes from scratch.

    This base seeds them directly: each even coord is an independent learnable
    2-component mixture  w*N(+m,sigma) + (1-w)*N(-m,sigma); each odd coord stays
    N(0,1) (already exact for the target).  Independent per-coord components give
    all 2^{n_wells} modes without enumeration, and the density factorizes so the
    exact mixture log-prob is a cheap per-coord logsumexp -- the NHMC path work
    W = log q(x0) + sum(log_fwd-log_bwd) + S stays exact.  m,sigma,w are learnable
    and shared across wells; at m->0, w->1/2 it reduces to N(0,I).
    """

    def __init__(self, dim: int, m_init: float = 1.7, sigma_init: float = 0.7,
                 w_init: float = 0.65, learn: bool = True):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        raw_m = math.log(math.expm1(max(m_init, 1e-4)))
        w_logit = math.log(w_init / (1.0 - w_init))
        if learn:
            self._raw_m = nn.Parameter(torch.tensor(raw_m))
            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))
            self.w_logit = nn.Parameter(torch.tensor(w_logit))
        else:
            self.register_buffer('_raw_m', torch.tensor(raw_m))
            self.register_buffer('log_sigma', torch.tensor(math.log(sigma_init)))
            self.register_buffer('w_logit', torch.tensor(w_logit))

    @property
    def m(self):
        return torch.nn.functional.softplus(self._raw_m).clamp(0.0, 4.0)

    @property
    def sigma(self):
        return torch.exp(self.log_sigma).clamp(0.15, 2.0)

    @property
    def w(self):
        return torch.sigmoid(self.w_logit).clamp(0.02, 0.98)

    def _logN(self, x, mu, sigma):
        return -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) \
            - 0.5 * ((x - mu) / sigma) ** 2

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        even = x[:, 0::2]
        odd = x[:, 1::2]
        m, sigma, w = self.m, self.sigma, self.w
        lp_plus = math.log(1.0) + torch.log(w) + self._logN(even, m, sigma)
        lp_minus = torch.log(1.0 - w) + self._logN(even, -m, sigma)
        lp_even = torch.logsumexp(torch.stack([lp_plus, lp_minus], dim=0), dim=0)
        lp_odd = self._logN(odd, 0.0, torch.ones((), device=x.device))
        return lp_even.sum(1) + lp_odd.sum(1)

    def sample(self, batch: int, device=None, dtype=torch.float32):
        device = device or self._raw_m.device
        m, sigma, w = self.m, self.sigma, self.w
        n_even = self.dim // 2
        eps_e = torch.randn(batch, n_even, device=device, dtype=dtype)
        sign = torch.where(torch.rand(batch, n_even, device=device) < w,
                           torch.ones(batch, n_even, device=device, dtype=dtype),
                           -torch.ones(batch, n_even, device=device, dtype=dtype))
        even = sigma * eps_e + sign * m
        odd = torch.randn(batch, n_even, device=device, dtype=dtype)
        x = torch.empty(batch, self.dim, device=device, dtype=dtype)
        x[:, 0::2] = even
        x[:, 1::2] = odd
        return x, self.log_prob(x)


def make_dw_base(base_type: str, dim: int, m_init: float = 1.7, learn: bool = True):
    """Factory: 'gaussian' -> None (legacy N(0,I) path); 'modeaware' -> base."""
    if base_type == 'gaussian':
        return None
    if base_type == 'modeaware':
        return DWModeAwareBase(dim, m_init=m_init, learn=learn)
    raise ValueError(f'unknown DW base_type {base_type}')


def rollout(kernel, target: ManyWellTarget, n_batch: int, num_steps: int,
            device, dtype=torch.float32, record=False, base=None):
    """Forward rollout from the base.  Returns (x_L, W) or, if record, also the
    full trajectory [num_steps+1, B, dim].

    base=None reproduces the legacy N(0,I) prior bit-for-bit; pass a learnable
    base (e.g. DWModeAwareBase) to seed the target modes from the proposal.
    """
    if base is not None:
        x0, logP = base.sample(n_batch, device=device, dtype=dtype)
    else:
        x0 = torch.randn(n_batch, target.dim, device=device, dtype=dtype)
        logP = base_log_prob(x0)
    x = x0
    traj = [x0.detach().cpu()] if record else None
    for step in range(1, num_steps + 1):
        x, lf, lb = kernel.forward_step(x, step, num_steps)
        logP = logP + lf - lb
        if record:
            traj.append(x.detach().cpu())
    S = -target.log_prob(x)
    W = logP + S
    if record:
        return x, W, torch.stack(traj, 0)
    return x, W


def ess_logz(W: torch.Tensor):
    """Return (normalized_ess, logZ_hat) from path-work W (logw = -W)."""
    logw = -W
    n = logw.numel()
    log_sum = torch.logsumexp(logw, 0)
    log_sum2 = torch.logsumexp(2 * logw, 0)
    ess = float(torch.exp(2 * log_sum - log_sum2) / n)
    logz = float(log_sum - math.log(n))
    return ess, logz
