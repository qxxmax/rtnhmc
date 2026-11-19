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

NOTE (recovery provenance): the baseline implementation was reconstructed on
2026-07-11 from Cursor's local editor history after the working copy had been
deleted. The default MLP/scalar-step configuration preserves that recovered
implementation. Optional block-equivariant momentum laws and stagewise step
sizes were added later without changing the recorded-work correction.
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Normal

from pisgrad_net_torch import PISGradMomentumNet

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


class TimeScale(nn.Module):
    """Per-dimension, time-only coefficient, zero-initialized.

    Used for the PIS-Grad score term so that training starts from the
    uninjected kernel and the score enters only as much as it earns.
    """

    def __init__(self, dim: int, hidden: int = 64, n_freq: int = 8):
        super().__init__()
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(2 * n_freq, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            1, self.n_freq + 1, device=t_norm.device, dtype=t_norm.dtype
        )
        angles = 2.0 * math.pi * t_norm[:, None] * freqs[None, :]
        return self.net(torch.cat([torch.sin(angles), torch.cos(angles)], dim=1))


class BlockEquivariantMomentumNet(nn.Module):
    """Permutation-equivariant network over the target's two-coordinate wells.

    The FAB many-well target is a product of identical two-dimensional factors.
    A shared local encoder and a pooled global context therefore preserve the
    permutation symmetry between wells while still allowing collective moves.
    """

    def __init__(self, dim: int, hidden: int = 128, n_freq: int = 8,
                 out_bias: float = 0.0):
        super().__init__()
        if dim % 2:
            raise ValueError("block-equivariant momentum laws require even dim")
        self.dim = dim
        self.n_wells = dim // 2
        self.n_freq = n_freq
        time_dim = 2 * n_freq
        self.local_encoder = nn.Sequential(
            nn.Linear(2 + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.block_head = nn.Sequential(
            nn.Linear(2 * hidden + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.block_head[-1].weight)
        nn.init.constant_(self.block_head[-1].bias, out_bias)

    def _temb(self, t_norm: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            1, self.n_freq + 1, device=t_norm.device, dtype=t_norm.dtype
        )
        angles = 2.0 * math.pi * t_norm[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(self, x: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        blocks = x.reshape(batch, self.n_wells, 2)
        time = self._temb(t_norm)
        time_blocks = time[:, None, :].expand(-1, self.n_wells, -1)
        local = self.local_encoder(torch.cat([blocks, time_blocks], dim=-1))
        global_context = local.mean(dim=1, keepdim=True).expand_as(local)
        output = self.block_head(
            torch.cat([local, global_context, time_blocks], dim=-1)
        )
        return output.reshape(batch, self.dim)


# =========================================================================
# Continuous Neural HMC kernel (true force leapfrog + learned momentum)
# =========================================================================
class ContinuousNHMCKernel(nn.Module):
    def __init__(self, target: ManyWellTarget, n_leapfrog: int = 5,
                 hidden: int = 128, log_eps_init: float = -2.7,
                 architecture: str = "mlp", num_steps: int | None = None,
                 stagewise_epsilon: bool = False,
                 sigma_mode: str = "analytic", log_mass_init: float = -0.7,
                 learn_drift: bool = False, drift_scale: float = 1.0,
                 anneal_floor: float = 1.0, anneal_power: float = 1.0,
                 learn_schedule: bool = False, score_injection: bool = False,
                 score_clip: float = 20.0, pisgrad_score: bool = True,
                 mass_precond: str = "none", stl: bool = False,
                 fixed_momentum: bool = False):
        super().__init__()
        # fixed_momentum: state-independent Gaussian momentum laws N(0, sigma_t^2) (use with
        # sigma_mode='analytic', one scale per stage shared by both directions); maps, step sizes,
        # schedule and flips are unchanged. Revision comparison B (learned vs fixed momentum laws).
        self.fixed_momentum = bool(fixed_momentum)
        self.target = target
        self.dim = target.dim
        self.n_leapfrog = n_leapfrog
        self.architecture = architecture
        self.stagewise_epsilon = bool(stagewise_epsilon)
        if sigma_mode not in ("analytic", "learned"):
            raise ValueError(f"unknown sigma mode {sigma_mode!r}")
        self.sigma_mode = sigma_mode
        if architecture == "mlp":
            network = MomentumNet
        elif architecture == "block_equivariant":
            network = BlockEquivariantMomentumNet
        elif architecture == "pisgrad":
            network = partial(PISGradMomentumNet, target=target,
                              use_target_score=bool(pisgrad_score))
        else:
            raise ValueError(
                f"unknown momentum architecture {architecture!r}; expected "
                "'mlp', 'block_equivariant' or 'pisgrad'"
            )
        # momentum mean starts at 0; momentum log-sigma starts near -0.7 (sigma~0.5)
        self.fwd_mu = network(self.dim, hidden, out_bias=0.0)
        self.bwd_mu = network(self.dim, hidden, out_bias=0.0)
        if self.sigma_mode == "learned":
            self.fwd_logsig = network(self.dim, hidden, out_bias=-0.7)
            self.bwd_logsig = network(self.dim, hidden, out_bias=-0.7)
        else:
            # One learnable mass per stage, shared by the forward and backward
            # laws. Together with the zero-initialized mean heads this makes the
            # untrained kernel exactly annealed HMC with momentum refresh, so
            # the networks only ever learn a residual on top of it.
            size = (num_steps + 2) if num_steps is not None else 1
            self.log_mass = nn.Parameter(torch.full((size,), float(log_mass_init)))
        # Learned control inside the integrator. The analytic force is the only
        # drift otherwise, so the network sees the state once per stage and the
        # remaining n_leapfrog-1 steps run on pure physics. Leapfrog stays
        # symplectic and reversible under any force that depends on x alone, so
        # adding u_theta(x, t) leaves the path work exact.
        self.learn_drift = bool(learn_drift)
        self.drift_scale = float(drift_scale)
        if self.learn_drift:
            self.drift_net = network(self.dim, hidden, out_bias=0.0)

        # Stage-wise diagonal preconditioner for the position update. 'block'
        # gives the double-well coordinates and the Gaussian ones their own
        # mass, which is the anisotropy this target actually has: the wells sit
        # at |x|~2 with curvature O(10) while the odd coordinates are unit
        # Gaussians. Starting at log 0 makes it the identity, so 'none' and an
        # untrained 'block' run are the same kernel.
        # Sticking-the-landing. The momentum is still drawn with gradients, but
        # the forward density it is scored under is detached, which removes the
        # score-function term from the gradient. That term has zero mean, so
        # the estimate of grad E[W] is unaffected while its variance drops to
        # zero once the proposal is exact.
        #
        # It is not free: Var(W) also appears in the loss, and the score term
        # does not have zero mean against W^2, so the gradient of the variance
        # penalty picks up a bias. With var_weight at 0.05 the mean term
        # dominates, which is why this is worth an arm rather than a default.
        self.stl = bool(stl)

        if mass_precond not in ("none", "scalar", "block"):
            raise ValueError(f"unknown mass preconditioner {mass_precond!r}")
        self.mass_precond = mass_precond
        if mass_precond != "none":
            size = (num_steps + 2) if num_steps is not None else 1
            width = 1 if mass_precond == "scalar" else 2
            self.log_inv_mass = nn.Parameter(torch.zeros(size, width))

        # Geometric annealing path. The default floor of 1.0 reproduces the
        # original kernel, which applied the full many-well force from the very
        # first stage with no ramp at all. Annealing only changes the kernel;
        # the work still uses the q0/gamma endpoints, so it stays exact.
        self.anneal_floor = float(anneal_floor)
        self.anneal_power = float(anneal_power)
        self.learn_schedule = bool(learn_schedule)
        if self.learn_schedule:
            if num_steps is None or num_steps < 1:
                raise ValueError("a learned schedule requires a positive num_steps")
            fractions = (torch.arange(1, num_steps + 1, dtype=torch.float64)
                         / num_steps).pow(self.anneal_power)
            cumulative = self.anneal_floor + (1.0 - self.anneal_floor) * fractions
            start = torch.tensor([self.anneal_floor], dtype=torch.float64)
            increments = torch.diff(torch.cat([start, cumulative])).clamp_min(1e-6)
            logits = torch.log(torch.expm1(increments / increments.mean()))
            self.schedule_logits = nn.Parameter(logits.to(torch.get_default_dtype()))

        # PIS-Grad style score injection: the momentum mean gets an explicit
        # score term with a learned per-dimension, time-only coefficient. The
        # momentum networks otherwise never see the force at all.
        self.score_injection = bool(score_injection)
        self.score_clip = float(score_clip)
        if self.score_injection:
            self.fwd_score_scale = TimeScale(self.dim)
            self.bwd_score_scale = TimeScale(self.dim)

        self.log_epsilon = nn.Parameter(torch.tensor(float(log_eps_init)))
        if self.stagewise_epsilon:
            if num_steps is None or num_steps < 1:
                raise ValueError("stagewise epsilon requires a positive num_steps")
            self.log_epsilon_offsets = nn.Parameter(torch.zeros(num_steps + 1))
        else:
            self.register_parameter("log_epsilon_offsets", None)

    @property
    def epsilon(self):
        return torch.exp(self.log_epsilon).clamp(1e-4, 0.3)

    def epsilon_at(self, step: int):
        if self.log_epsilon_offsets is None:
            return self.epsilon
        log_epsilon = self.log_epsilon + self.log_epsilon_offsets[step]
        return torch.exp(log_epsilon).clamp(1e-4, 0.3)

    def _log_sigma(self, x, t_norm, step, reverse):
        if self.sigma_mode == "learned":
            net = self.bwd_logsig if reverse else self.fwd_logsig
            return net(x, t_norm).clamp(-5.0, 2.0)
        index = 0 if self.log_mass.numel() == 1 else int(step)
        return self.log_mass[index].clamp(-5.0, 2.0).expand_as(x)

    def beta(self, step, num_steps):
        """Annealing weight for stage ``step`` (1-indexed)."""
        if self.learn_schedule:
            if self.schedule_logits.numel() != num_steps:
                raise ValueError(
                    f"schedule has {self.schedule_logits.numel()} entries, "
                    f"expected {num_steps}"
                )
            increments = torch.nn.functional.softplus(self.schedule_logits)
            cumulative = torch.cumsum(increments, dim=0) / increments.sum()
            return self.anneal_floor + (1.0 - self.anneal_floor) * cumulative[step - 1]
        if self.anneal_floor >= 1.0:
            return 1.0
        fraction = (step / num_steps) ** self.anneal_power
        return self.anneal_floor + (1.0 - self.anneal_floor) * fraction

    def _clipped_score(self, x):
        """Target score, norm-clipped so a single clash cannot dominate."""
        f = self.target.force(x)
        if self.score_clip <= 0:
            return f
        limit = self.score_clip * math.sqrt(self.dim)
        norm = f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return f * (limit / norm).clamp(max=1.0)

    def _drift(self, x, t_norm, beta=1.0):
        f = self.target.force(x)
        if not (isinstance(beta, float) and beta == 1.0):
            # Geometric path between the standard-normal base and the target:
            # grad log q0 = -x for the unit Gaussian.
            f = beta * f + (1.0 - beta) * (-x)
        if self.learn_drift:
            f = f + self.drift_scale * self.drift_net(x, t_norm)
        return f

    def inverse_mass(self, step):
        """Diagonal inverse mass for stage ``step``, shape [1, dim] or scalar 1.

        The integrator was written with the mass pinned to the identity even
        though the momentum is drawn with a learned scale, so the scale never
        acted as a preconditioner. A mass that depends only on the stage fixes
        that and stays safe: the position update's Jacobian is
        [[I, eps D^-1], [0, I]], whose determinant is 1 for any constant D, so
        the map is still volume preserving and the path work is still exact.
        A mass that depended on x would break exactly that and is why the
        identity was the right default until now.
        """
        if self.mass_precond == "none":
            return 1.0
        inverse = torch.exp(self.log_inv_mass[step])
        if self.mass_precond == "scalar":
            return inverse
        blocks = torch.empty(self.dim, device=inverse.device, dtype=inverse.dtype)
        blocks[0::2] = inverse[0]
        blocks[1::2] = inverse[1]
        return blocks[None, :]

    def _leapfrog(self, x, p, eps, t_norm=None, beta=1.0, step=0):
        if t_norm is None:
            t_norm = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        velocity = eps * self.inverse_mass(step)
        p = p + 0.5 * eps * self._drift(x, t_norm, beta)
        for _ in range(self.n_leapfrog - 1):
            x = x + velocity * p
            p = p + eps * self._drift(x, t_norm, beta)
        x = x + velocity * p
        p = p + 0.5 * eps * self._drift(x, t_norm, beta)
        return x, p

    def _scoring_dist(self, dist):
        """The distribution the drawn momentum is scored under.

        Only the forward law goes through this: it is the one the sample comes
        from, so it is the only one carrying a score-function term.
        """
        if not self.stl:
            return dist
        return Normal(dist.loc.detach(), dist.scale.detach())

    def _momentum_mean(self, x, t_norm, reverse):
        if getattr(self, 'fixed_momentum', False):
            return torch.zeros_like(x)
        net = self.bwd_mu if reverse else self.fwd_mu
        mu = net(x, t_norm)
        if self.score_injection:
            scale = self.bwd_score_scale if reverse else self.fwd_score_scale
            mu = mu + scale(t_norm) * self._clipped_score(x)
        return mu

    def forward_step(self, x, t, num_steps):
        B = x.size(0)
        t_norm = torch.full((B,), t / num_steps, device=x.device, dtype=x.dtype)
        mu_f = self._momentum_mean(x, t_norm, reverse=False)
        sig_f = torch.exp(self._log_sigma(x, t_norm, t, reverse=False))
        dist_f = Normal(mu_f, sig_f)
        p0 = dist_f.rsample()
        beta = self.beta(t, num_steps)
        x_new, p_new = self._leapfrog(x, p0, self.epsilon_at(t), t_norm, beta, t)
        log_fwd = self._scoring_dist(dist_f).log_prob(p0).sum(1)

        t_next = torch.full_like(t_norm, (t + 1) / num_steps)
        mu_b = self._momentum_mean(x_new, t_next, reverse=True)
        sig_b = torch.exp(self._log_sigma(x_new, t_next, t + 1, reverse=True))
        dist_b = Normal(mu_b, sig_b)
        log_bwd = dist_b.log_prob(-p_new).sum(1)
        return x_new, log_fwd, log_bwd

    @torch.no_grad()
    def reverse_step(self, x, t, num_steps):
        """Recover x_{t-1} from x_t using the backward momentum law and the
        reversibility of leapfrog (volume preserving)."""
        B = x.size(0)
        t_next = torch.full((B,), (t + 1) / num_steps, device=x.device, dtype=x.dtype)
        mu_b = self._momentum_mean(x, t_next, reverse=True)
        sig_b = torch.exp(self._log_sigma(x, t_next, t + 1, reverse=True))
        m = Normal(mu_b, sig_b).sample()
        # The reverse map must replay stage t's integrator exactly, so it uses
        # t/num_steps and beta(t) here even though the momentum law is queried
        # at t+1.
        t_stage = torch.full_like(t_next, t / num_steps)
        x_prev, _ = self._leapfrog(
            x, m, self.epsilon_at(t), t_stage, self.beta(t, num_steps), t
        )
        return x_prev


class UnderdampedNHMCKernel(nn.Module):
    """NHMC with a partial momentum refresh instead of a full one.

    The base kernel throws the momentum away after every stage, so each stage
    starts from scratch and the path has to rebuild its direction of travel
    twenty times. This carries the momentum along and only partially refreshes
    it, which is the structure of the Z2EquivUnderdampedKernel in the phi^4
    project; the difference here is that the deterministic part is the
    true-force leapfrog rather than a learned drift.

    The state becomes ``(x, p)``. The extended target is
    ``gamma(x) N(p; 0, I)`` and the extended base is ``N(x; 0, I) N(p; 0, I)``,
    both normalized in ``p``, so Z is unchanged and the estimator stays exact.

    One stage is refresh, leapfrog, momentum flip. That composition is an
    involution, which is what lets the reverse kernel have the same functional
    form as the forward one:

        forward   p1 ~ N(alpha * p  + mu_f(x, t), sigma_f)
        reverse  -p  ~ N(-alpha * p1 + mu_b(x, t), sigma_b)

    Both are normalized densities in their own variable, which is the only
    thing the path-ratio needs.
    """

    def __init__(self, base: ContinuousNHMCKernel, num_steps: int,
                 friction_init: float = 0.5, use_flip: bool = False):
        super().__init__()
        self.base = base
        self.dim = base.dim
        self.num_steps = num_steps
        # alpha = sigmoid(logit): 0 is a full refresh, 1 keeps the momentum.
        logit = math.log(friction_init / (1.0 - friction_init))
        self.friction_logits = nn.Parameter(torch.full((num_steps + 1,), logit))
        # The sign flip is what lets the chain tunnel between the 2^(d/2) modes,
        # so the underdamped arm needs it too or it loses to the baseline for a
        # reason that has nothing to do with the momentum. On the augmented
        # state the flip has to act on x and p together: it is then a linear
        # involution of unit Jacobian, and gamma(x) N(p; 0, I) is invariant
        # under it because the momentum factor is isotropic.
        self.use_flip = bool(use_flip)
        if self.use_flip:
            self.n_wells = base.target.n_wells
            self.flip_logits = nn.Parameter(torch.zeros(num_steps + 1))

    def alpha(self, step):
        return torch.sigmoid(self.friction_logits[step])

    def _flip(self, x, p, t, num_steps):
        prob_f = torch.sigmoid(self.flip_logits[t])
        flip = (torch.rand(x.size(0), self.n_wells, device=x.device) < prob_f).to(x.dtype)
        sign = 1.0 - 2.0 * flip
        x = x.clone()
        p = p.clone()
        x[:, 0::2] = x[:, 0::2] * sign
        p[:, 0::2] = p[:, 0::2] * sign
        log_rho_f = (flip * torch.log(prob_f + 1e-8)
                     + (1 - flip) * torch.log1p(-prob_f + 1e-8)).sum(1)
        prob_b = torch.sigmoid(self.flip_logits[min(t + 1, num_steps)])
        log_rho_b = (flip * torch.log(prob_b + 1e-8)
                     + (1 - flip) * torch.log1p(-prob_b + 1e-8)).sum(1)
        return x, p, log_rho_f, log_rho_b

    def forward_step(self, x, p, t, num_steps):
        base = self.base
        log_rho_f = log_rho_b = 0.0
        if self.use_flip:
            x, p, log_rho_f, log_rho_b = self._flip(x, p, t, num_steps)
        B = x.size(0)
        t_norm = torch.full((B,), t / num_steps, device=x.device, dtype=x.dtype)
        alpha = self.alpha(t)

        mu_f = base._momentum_mean(x, t_norm, reverse=False)
        sig_f = torch.exp(base._log_sigma(x, t_norm, t, reverse=False))
        dist_f = Normal(alpha * p + mu_f, sig_f)
        p1 = dist_f.rsample()
        log_fwd = base._scoring_dist(dist_f).log_prob(p1).sum(1)

        beta = base.beta(t, num_steps)
        x_new, p2 = base._leapfrog(x, p1, base.epsilon_at(t), t_norm, beta, t)

        # The reverse refresh is read at the same position the forward one was,
        # which the involution makes recoverable from the new state.
        mu_b = base._momentum_mean(x, t_norm, reverse=True)
        sig_b = torch.exp(base._log_sigma(x, t_norm, t, reverse=True))
        dist_b = Normal(-alpha * p1 + mu_b, sig_b)
        log_bwd = dist_b.log_prob(-p).sum(1)

        return x_new, -p2, log_fwd + log_rho_f, log_bwd + log_rho_b

    @torch.no_grad()
    def reverse_step(self, x, p, t, num_steps):
        """Recover ``(x_{t-1}, p_{t-1})`` from ``(x_t, p_t)``.

        The stage is refresh, leapfrog, flip, and that composition is an
        involution, so replaying the leapfrog on the current state lands on
        ``(x_{t-1}, -p1)``. Only the refresh has to be undone stochastically,
        and it is undone with the same backward law the work charges for.
        """
        base = self.base
        B = x.size(0)
        t_norm = torch.full((B,), t / num_steps, device=x.device, dtype=x.dtype)
        x_prev, minus_p1 = base._leapfrog(
            x, p, base.epsilon_at(t), t_norm, base.beta(t, num_steps), t
        )
        p1 = -minus_p1
        mu_b = base._momentum_mean(x_prev, t_norm, reverse=True)
        sig_b = torch.exp(base._log_sigma(x_prev, t_norm, t, reverse=True))
        p_prev = Normal(-self.alpha(t) * p1 + mu_b, sig_b).sample()
        if self.use_flip:
            prob = torch.sigmoid(self.flip_logits[t])
            flip = (torch.rand(B, self.n_wells, device=x.device) < prob).to(x.dtype)
            sign = 1.0 - 2.0 * flip
            x_prev = x_prev.clone()
            p_prev = p_prev.clone()
            x_prev[:, 0::2] = x_prev[:, 0::2] * sign
            p_prev[:, 0::2] = p_prev[:, 0::2] * sign
        return x_prev, p_prev


def rollout_underdamped(kernel: UnderdampedNHMCKernel, target: ManyWellTarget,
                        n_batch: int, num_steps: int, device,
                        dtype=torch.float32, stagewise=False):
    """Forward rollout of the augmented chain from N(0,I) x N(0,I).

    The momentum enters the base and the extended target through the same
    normalized N(0, I), so it cancels out of Z and only its fluctuations reach
    the work.

    With ``stagewise`` the per-stage increments are returned too. The bridge
    carries the momentum factor along with the position one, since the
    extended target is ``gamma(x) N(p; 0, I)``; dropping it would break the
    telescoping and the free-energy head would regress onto the wrong thing.
    """
    x = torch.randn(n_batch, target.dim, device=device, dtype=dtype)
    p = torch.randn(n_batch, target.dim, device=device, dtype=dtype)
    logP = base_log_prob(x) + base_log_prob(p)
    stages = [] if stagewise else None
    prev_log_gamma = base_log_prob(x) + base_log_prob(p) if stagewise else None
    for step in range(1, num_steps + 1):
        x, p, lf, lb = kernel.forward_step(x, p, step, num_steps)
        logP = logP + lf - lb
        if stagewise:
            beta = 1.0 if step == num_steps else float(kernel.base.beta(step, num_steps))
            next_log_gamma = log_gamma_tilde(target, x, beta) + base_log_prob(p)
            stages.append(prev_log_gamma - next_log_gamma + lf - lb)
            prev_log_gamma = next_log_gamma
    W = logP - target.log_prob(x) - base_log_prob(p)
    if stagewise:
        return x, W, torch.stack(stages, 0)
    return x, W


# =========================================================================
# Mixed-sign wrapper (per-double-well sign flip; continuous MixedZ2)
# =========================================================================
class MixedSignKernel(nn.Module):
    """Per-even-coordinate independent Bernoulli sign flip before the base step.

    Flip prob rho(t) is learned per step (shared across the n_wells coords).
    The flip action is recorded in the path ratio exactly like MixedZ2Kernel.

    The discrete draw itself is not differentiable. The path work already
    contains log rho, so the pathwise derivative of that term is present;
    what is missing is the score-function contribution W * grad log r of
    the Bernoulli sampling distribution. ``flip_score`` adds that term at
    training time. It is off by default so existing checkpoints and the
    current loss surface stay bit-identical.
    """

    def __init__(self, base: ContinuousNHMCKernel, num_steps: int,
                 flip_score: bool = False):
        super().__init__()
        self.base = base
        self.dim = base.dim
        self.n_wells = base.target.n_wells
        self.flip_logits = nn.Parameter(torch.zeros(num_steps + 1))
        # Flag only; not a Parameter, so state_dict keys stay unchanged.
        self.flip_score = bool(flip_score)
        self._flip_logprob = None

    def reset_flip_logprob(self):
        self._flip_logprob = None

    def consume_flip_logprob(self):
        value = self._flip_logprob
        self._flip_logprob = None
        return value

    def _accumulate_flip_logprob(self, log_rho_f):
        if not self.flip_score:
            return
        if self._flip_logprob is None:
            self._flip_logprob = log_rho_f
        else:
            self._flip_logprob = self._flip_logprob + log_rho_f

    def forward_step(self, x, t, num_steps):
        B = x.size(0)
        prob_f = torch.sigmoid(self.flip_logits[t])
        # In the path's dtype: a float32 mask times a zero-dimensional float64
        # probability stays float32, because dimensioned tensors win that
        # promotion, so a float64 evaluation would score the flip in single
        # precision while the rest of the work is double.
        flip = (torch.rand(B, self.n_wells, device=x.device,
                           dtype=x.dtype) < prob_f).to(x.dtype)  # [B,n_wells]
        sign = 1.0 - 2.0 * flip
        x_flip = x.clone()
        x_flip[:, 0::2] = x[:, 0::2] * sign

        log_rho_f = (flip * torch.log(prob_f + 1e-8)
                     + (1 - flip) * torch.log(1 - prob_f + 1e-8)).sum(1)
        self._accumulate_flip_logprob(log_rho_f)

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


def log_gamma_tilde(target, x, beta):
    """Unnormalized log density of the intermediate target at annealing level
    ``beta``: the geometric bridge from the base N(0, I) to the true target.

    ``beta=0`` gives the base exactly and ``beta=1`` gives the target exactly,
    which is what makes the stagewise work decomposition telescope.
    """
    if beta <= 0.0:
        return base_log_prob(x)
    if beta >= 1.0:
        return target.log_prob(x)
    return (1.0 - beta) * base_log_prob(x) + beta * target.log_prob(x)


def rollout(kernel, target: ManyWellTarget, n_batch: int, num_steps: int,
            device, dtype=torch.float32, record=False, stagewise=False):
    """Forward rollout from N(0,I).  Returns (x_L, W) or, if record, also the
    full trajectory [num_steps+1, B, dim].

    With ``stagewise`` the per-stage work increments are returned as well, with
    shape [num_steps, B]. They sum to W exactly, so using them in the loss
    leaves the estimator untouched; they are what the free-energy head needs,
    since E[w_t] is the free-energy difference across stage t.
    """
    reset = getattr(kernel, "reset_flip_logprob", None)
    if callable(reset):
        reset()
    x0 = torch.randn(n_batch, target.dim, device=device, dtype=dtype)
    logP = base_log_prob(x0)
    x = x0
    traj = [x0.detach().cpu()] if record else None
    stages = [] if stagewise else None
    base_kernel = getattr(kernel, "base", kernel)
    # The bridge starts at the base density regardless of the kernel's own
    # schedule, which may hold beta at 1 throughout.
    prev_log_gamma = base_log_prob(x0) if stagewise else None
    for step in range(1, num_steps + 1):
        x, lf, lb = kernel.forward_step(x, step, num_steps)
        logP = logP + lf - lb
        if stagewise:
            beta = 1.0 if step == num_steps else float(base_kernel.beta(step, num_steps))
            next_log_gamma = log_gamma_tilde(target, x, beta)
            stages.append(prev_log_gamma - next_log_gamma + lf - lb)
            prev_log_gamma = next_log_gamma
        if record:
            traj.append(x.detach().cpu())
    S = -target.log_prob(x)
    W = logP + S
    out = (x, W)
    if record:
        out = out + (torch.stack(traj, 0),)
    if stagewise:
        out = out + (torch.stack(stages, 0),)
    return out


def ess_logz(W: torch.Tensor):
    """Return (normalized_ess, logZ_hat) from path-work W (logw = -W)."""
    logw = -W
    n = logw.numel()
    log_sum = torch.logsumexp(logw, 0)
    log_sum2 = torch.logsumexp(2 * logw, 0)
    ess = float(torch.exp(2 * log_sum - log_sum2) / n)
    logz = float(log_sum - math.log(n))
    return ess, logz
