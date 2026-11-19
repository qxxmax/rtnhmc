"""NHMC stage kernels (kappa-conditional), Z2 wrapper, base law and model loading.

forward_step(x, t, T, kappa) returns (x_next, log q_F terms, log q_R terms) of one stage; reverse_step draws the
reverse stage from x_next with the same two terms, so that forward and reverse paths share one work convention.
Kernels used in the paper: 'nhmc_resid2' (native and T-scan models) and 'nhmc_learned' (kappa-conditional L = 8 model),
both wrapped by CondZ2Wrapper; odd_nets makes the force / momentum means odd and the widths even in phi.
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from networks import CondDriftNet, CondSigmaNet
from phi4 import phi4_force


def _phys_force(x, lam, kappa):
    """-grad S for x of shape [B,1,L,L]: phi4_force rolls over every dim >= 1, so the channel
    dim must be removed first (as in the original kernels' phi_sq)."""
    return phi4_force(x.squeeze(1), lam=lam, kappa=kappa).unsqueeze(1)


def _broadcast_kappa(kappa, B, device, dtype):
    """Return a (B,) tensor.  Accepts:
       * float / 0-d tensor  -> torch.full((B,), value)
       * 1-d tensor of len B -> as-is (per-sample kappa)
       * 1-d tensor of len 1 -> expand
    """
    if isinstance(kappa, torch.Tensor):
        if kappa.dim() == 0:
            return kappa.to(device=device, dtype=dtype).expand(B)
        if kappa.dim() == 1 and kappa.numel() == B:
            return kappa.to(device=device, dtype=dtype)
        if kappa.dim() == 1 and kappa.numel() == 1:
            return kappa.to(device=device, dtype=dtype).expand(B)
        raise ValueError(
            f"kappa tensor shape {tuple(kappa.shape)} incompatible with B={B}")
    return torch.full((B,), float(kappa), device=device, dtype=dtype)


class CondNeuralHMCLearnedKernel(nn.Module):
    """Learned leapfrog force f_theta(phi,t,kappa); S enters only through W = logP + S."""

    def __init__(self, embed_dim=128, channels=(32, 32), n_leapfrog=5):
        super().__init__()
        self.n_leapfrog = n_leapfrog

        self.fwd_mu = CondDriftNet(embed_dim, channels)
        self.fwd_sigma = CondSigmaNet(embed_dim, channels)
        self.bwd_mu = CondDriftNet(embed_dim, channels)
        self.bwd_sigma = CondSigmaNet(embed_dim, channels)

        self.learned_force = CondDriftNet(embed_dim, channels)

        self.log_eps = nn.Parameter(torch.tensor(-2.0))

    @property
    def epsilon(self):
        return torch.exp(self.log_eps).clamp(1e-4, 0.5)

    def _leapfrog(self, phi, p, eps, t_norm, kappa_t):
        force = self.learned_force(phi, t_norm, kappa_t)
        p = p + 0.5 * eps * force
        for _ in range(self.n_leapfrog - 1):
            phi = phi + eps * p
            force = self.learned_force(phi, t_norm, kappa_t)
            p = p + eps * force
        phi = phi + eps * p
        force = self.learned_force(phi, t_norm, kappa_t)
        p = p + 0.5 * eps * force
        return phi, p

    def forward_step(self, x, t, num_steps, kappa=0.27):
        B = x.size(0)
        device = x.device
        t_norm = torch.full((B,), t / num_steps, device=device, dtype=x.dtype)
        kappa_t = _broadcast_kappa(kappa, B, device, x.dtype)

        mu_f = self.fwd_mu(x, t_norm, kappa_t)
        sigma_f = torch.exp(self.fwd_sigma(x, t_norm, kappa_t))
        p_dist_f = Normal(mu_f, sigma_f)
        p0 = p_dist_f.rsample()
        log_fwd = p_dist_f.log_prob(p0).sum(dim=(1, 2, 3))

        phi_new, p_new = self._leapfrog(x, p0, self.epsilon, t_norm, kappa_t)

        t_next = torch.full_like(t_norm, (t + 1) / num_steps)
        mu_b = self.bwd_mu(phi_new, t_next, kappa_t)
        sigma_b = torch.exp(self.bwd_sigma(phi_new, t_next, kappa_t))
        log_bwd = Normal(mu_b, sigma_b).log_prob(-p_new).sum(dim=(1, 2, 3))

        return phi_new, log_fwd, log_bwd

    def _map(self, phi, p, t, t_norm, kappa_t):
        """Deterministic stage map (leapfrog with this kernel's force and step)."""
        return self._leapfrog(phi, p, self.epsilon, t_norm, kappa_t)

    def reverse_step(self, x_next, t, num_steps, kappa=0.27, r=None):
        """Reverse stage t: x_next -> x_in with r = -p_new ~ q_R(.|x_next) (reparametrised, so the
        reverse path is differentiable in the parameters). LF(x_next, r) = (x_in, -p0) by
        reversibility. Returns (x_in, log q_F(p0|x_in), log q_R(r|x_next)), the same two terms
        forward_step returns for the forward path x_in -> x_next."""
        B = x_next.size(0)
        device = x_next.device
        t_norm = torch.full((B,), t / num_steps, device=device, dtype=x_next.dtype)
        t_next = torch.full_like(t_norm, (t + 1) / num_steps)
        kappa_t = _broadcast_kappa(kappa, B, device, x_next.dtype)
        mu_b = self.bwd_mu(x_next, t_next, kappa_t)
        sigma_b = torch.exp(self.bwd_sigma(x_next, t_next, kappa_t))
        if r is None:
            r = mu_b + sigma_b * torch.randn_like(mu_b)
        log_bwd = Normal(mu_b, sigma_b).log_prob(r).sum(dim=(1, 2, 3))
        x_in, mp0 = self._map(x_next, r, t, t_norm, kappa_t)
        p0 = -mp0
        mu_f = self.fwd_mu(x_in, t_norm, kappa_t)
        sigma_f = torch.exp(self.fwd_sigma(x_in, t_norm, kappa_t))
        log_fwd = Normal(mu_f, sigma_f).log_prob(p0).sum(dim=(1, 2, 3))
        return x_in, log_fwd, log_bwd


class CondNHMCResidualKernel(CondNeuralHMCLearnedKernel):
    """Learned-force NHMC whose kick is an annealed physical force plus a residual.

        F_t(phi) = beta_t * (-grad S_kappa(phi)) + (1 - beta_t) * (-phi) + f_theta(phi, t, kappa)

    -phi is the score of the N(0, I) base, beta_t = cumsum(softmax(a))[t-1] is a
    learned monotone schedule with beta_T = 1 (CMCD-style geometric annealing),
    and f_theta starts at zero.  Momentum laws, path ratio and Z2 wrapper are
    unchanged, so W, SNIS, log Z and path-IMH keep their meaning.
    """

    def __init__(self, embed_dim=128, channels=(32, 32), n_leapfrog=5, lam=0.022, num_steps=16):
        super().__init__(embed_dim, channels, n_leapfrog)
        self.lam = lam
        self.beta_logits = nn.Parameter(torch.zeros(num_steps))
        with torch.no_grad():
            for prm in self.learned_force.conv_out.parameters():
                prm.zero_()

    def beta(self, t):
        b = torch.cumsum(torch.softmax(self.beta_logits, 0), 0)
        return b[min(max(t, 1), b.numel()) - 1]

    def _force(self, phi, b, t_norm, kappa_t):
        phys = _phys_force(phi, self.lam, kappa_t)
        return b * phys - (1.0 - b) * phi + self.learned_force(phi, t_norm, kappa_t)

    def _leapfrog_t(self, phi, p, eps, t, t_norm, kappa_t):
        b = self.beta(t)
        force = self._force(phi, b, t_norm, kappa_t)
        p = p + 0.5 * eps * force
        for _ in range(self.n_leapfrog - 1):
            phi = phi + eps * p
            force = self._force(phi, b, t_norm, kappa_t)
            p = p + eps * force
        phi = phi + eps * p
        force = self._force(phi, b, t_norm, kappa_t)
        p = p + 0.5 * eps * force
        return phi, p

    def _map(self, phi, p, t, t_norm, kappa_t):
        return self._leapfrog_t(phi, p, self.epsilon, t, t_norm, kappa_t)

    def forward_step(self, x, t, num_steps, kappa=0.27):
        B = x.size(0)
        device = x.device
        t_norm = torch.full((B,), t / num_steps, device=device, dtype=x.dtype)
        kappa_t = _broadcast_kappa(kappa, B, device, x.dtype)

        mu_f = self.fwd_mu(x, t_norm, kappa_t)
        sigma_f = torch.exp(self.fwd_sigma(x, t_norm, kappa_t))
        p_dist_f = Normal(mu_f, sigma_f)
        p0 = p_dist_f.rsample()
        log_fwd = p_dist_f.log_prob(p0).sum(dim=(1, 2, 3))

        phi_new, p_new = self._leapfrog_t(x, p0, self.epsilon, t, t_norm, kappa_t)

        t_next = torch.full_like(t_norm, (t + 1) / num_steps)
        mu_b = self.bwd_mu(phi_new, t_next, kappa_t)
        sigma_b = torch.exp(self.bwd_sigma(phi_new, t_next, kappa_t))
        log_bwd = Normal(mu_b, sigma_b).log_prob(-p_new).sum(dim=(1, 2, 3))

        return phi_new, log_fwd, log_bwd


def base_draw(kernel, shape, device, kappa=None):
    """Draw x0 from the kernel's base law and return (x0, log q0(x0)).

    Kernels that define base_std() use N(0, s^2) with a learned scalar s
    (reparametrised); every other kernel keeps the original N(0, I).
    """
    inner = getattr(kernel, "base", kernel)
    if hasattr(inner, "base_sample"):
        return inner.base_sample(shape, device, kappa)
    if hasattr(inner, "base_std"):
        s = inner.base_std()
        x0 = s * torch.randn(shape, device=device)
        lp = Normal(torch.zeros_like(x0), s * torch.ones_like(x0)).log_prob(x0)
    else:
        x0 = torch.randn(shape, device=device)
        lp = Normal(torch.zeros_like(x0), torch.ones_like(x0)).log_prob(x0)
    return x0, lp.sum(dim=(1, 2, 3))


class CondNHMCResidual2Kernel(CondNHMCResidualKernel):
    """Residual NHMC plus three CMCD/UDB-style parameters.

        F_t = (beta_t + g_t) * (-grad S) + (1 - beta_t) * (-phi / s^2) + f_theta

    g_t      per-stage gate on the physical force (PIS-GRAD style), init 0
    s        learned base width, q0 = N(0, s^2 I), init 1
    eps_t    per-stage leapfrog step, init exp(-2)
    """

    def __init__(self, embed_dim=128, channels=(32, 32), n_leapfrog=5, lam=0.022, num_steps=16):
        super().__init__(embed_dim, channels, n_leapfrog, lam, num_steps)
        self.gate = nn.Parameter(torch.zeros(num_steps))
        self.log_base_std = nn.Parameter(torch.zeros(()))
        self.log_eps_t = nn.Parameter(torch.full((num_steps,), -2.0))

    def base_std(self):
        return torch.exp(self.log_base_std).clamp(0.05, 20.0)

    def eps_at(self, t):
        return torch.exp(self.log_eps_t[min(max(t, 1), self.log_eps_t.numel()) - 1]).clamp(1e-4, 0.5)

    def _force2(self, phi, b, g, inv_s2, t_norm, kappa_t):
        phys = _phys_force(phi, self.lam, kappa_t)
        return (b + g) * phys - (1.0 - b) * inv_s2 * phi + self.learned_force(phi, t_norm, kappa_t)

    def _leapfrog_t(self, phi, p, eps, t, t_norm, kappa_t):
        b = self.beta(t)
        g = self.gate[min(max(t, 1), self.gate.numel()) - 1]
        inv_s2 = 1.0 / self.base_std() ** 2
        eps = self.eps_at(t)
        force = self._force2(phi, b, g, inv_s2, t_norm, kappa_t)
        p = p + 0.5 * eps * force
        for _ in range(self.n_leapfrog - 1):
            phi = phi + eps * p
            force = self._force2(phi, b, g, inv_s2, t_norm, kappa_t)
            p = p + eps * force
        phi = phi + eps * p
        force = self._force2(phi, b, g, inv_s2, t_norm, kappa_t)
        p = p + 0.5 * eps * force
        return phi, p


class _Odd(nn.Module):
    """Make a drift net odd in phi: f(phi) -> (f(phi) - f(-phi)) / 2."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, kappa):
        return 0.5 * (self.net(x, t, kappa) - self.net(-x, t, kappa))


class _Even(nn.Module):
    """Make a (log-width) net even in phi: f(phi) -> (f(phi) + f(-phi)) / 2."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, kappa):
        return 0.5 * (self.net(x, t, kappa) + self.net(-x, t, kappa))


class CondZ2Wrapper(nn.Module):
    """
    Same flip bookkeeping as kernels.MixedZ2Kernel:
      log q += log rho(a|t) - log rho(a|t+1)
    """

    def __init__(self, base_kernel, num_steps):
        super().__init__()
        self.base = base_kernel
        self.flip_logits = nn.Parameter(torch.zeros(num_steps + 1))

    def forward_step(self, x, t, num_steps, kappa=0.27):
        B = x.size(0)
        device = x.device

        logit_t = self.flip_logits[t]
        prob_flip = torch.sigmoid(logit_t)
        flip = (torch.rand(B, device=device) < prob_flip).float()
        sign = 1.0 - 2.0 * flip
        x_in = x * sign.view(-1, 1, 1, 1)

        log_rho_fwd = torch.where(
            flip.bool(),
            torch.log(prob_flip + 1e-8),
            torch.log(1.0 - prob_flip + 1e-8),
        )

        x_next, log_fwd_b, log_bwd_b = self.base.forward_step(
            x_in, t, num_steps, kappa=kappa
        )

        logit_t1 = self.flip_logits[min(t + 1, num_steps)]
        prob_flip_bwd = torch.sigmoid(logit_t1)
        log_rho_bwd = torch.where(
            flip.bool(),
            torch.log(prob_flip_bwd + 1e-8),
            torch.log(1.0 - prob_flip_bwd + 1e-8),
        )

        return x_next, log_fwd_b + log_rho_fwd, log_bwd_b + log_rho_bwd

    def reverse_step(self, x_next, t, num_steps, kappa=0.27, r=None):
        """Reverse of forward_step: base reverse stage, then the flip drawn with rho(t+1);
        returns (x_prev, log q_F-terms incl. log rho_t(a), log q_R-terms incl. log rho_{t+1}(a))."""
        x_in, log_fwd_b, log_bwd_b = self.base.reverse_step(x_next, t, num_steps, kappa=kappa, r=r)
        prob_bwd = torch.sigmoid(self.flip_logits[min(t + 1, num_steps)])
        prob_fwd = torch.sigmoid(self.flip_logits[t])
        flip = torch.rand(x_next.size(0), device=x_next.device) < prob_bwd
        x_prev = x_in * torch.where(flip, -1.0, 1.0).to(x_in.dtype).view(-1, 1, 1, 1)
        log_rho_fwd = torch.where(flip, torch.log(prob_fwd + 1e-8), torch.log(1.0 - prob_fwd + 1e-8))
        log_rho_bwd = torch.where(flip, torch.log(prob_bwd + 1e-8), torch.log(1.0 - prob_bwd + 1e-8))
        return x_prev, log_fwd_b + log_rho_fwd, log_bwd_b + log_rho_bwd


def base_logprob(kernel, x):
    """log q0(x) of the kernel's base law (N(0, I) or learned N(0, s^2 I))."""
    inner = getattr(kernel, "base", kernel)
    assert not hasattr(inner, "base_sample"), "base_logprob: kernel has a custom base sampler"
    s = inner.base_std() if hasattr(inner, "base_std") else torch.ones((), device=x.device)
    return Normal(torch.zeros_like(x), s * torch.ones_like(x)).log_prob(x).sum(dim=(1, 2, 3))


COND_KERNEL_REGISTRY = {
    "nhmc_learned": CondNeuralHMCLearnedKernel,
    "nhmc_resid": CondNHMCResidualKernel,
    "nhmc_resid2": CondNHMCResidual2Kernel,
}

COND_KERNEL_LATTICE_REGISTRY = {}


def create_cond_kernel(
    name,
    embed_dim=128,
    channels=(32, 32),
    n_leapfrog=10,
    lam=0.022,
    num_steps=10,
    use_z2=False,
    lattice_embed=False,
    lattice_ref=16.0,
    odd_nets=False,
):
    reg = COND_KERNEL_LATTICE_REGISTRY if lattice_embed else COND_KERNEL_REGISTRY
    if name not in reg:
        raise ValueError(f"Unknown kernel {name}; choose from {list(reg)}")

    cls = reg[name]
    if lattice_embed:
        if name == "neural_hmc":
            kernel = cls(embed_dim, channels, n_leapfrog, lam, lattice_ref=lattice_ref)
        elif name == "nhmc_learned":
            kernel = cls(embed_dim, channels, n_leapfrog, lattice_ref=lattice_ref)
        else:
            kernel = cls(embed_dim, channels, lattice_ref=lattice_ref)
        if use_z2:
            kernel = CondZ2LatticeWrapper(kernel, num_steps)
    else:
        if name == "neural_hmc":
            kernel = cls(embed_dim, channels, n_leapfrog, lam)
        elif name == "nhmc_learned":
            kernel = cls(embed_dim, channels, n_leapfrog)
        elif name in ("nhmc_resid", "nhmc_resid2"):
            kernel = cls(embed_dim, channels, n_leapfrog, lam, num_steps)
        elif name in ("nhmc_cont", "nhmc_cont2", "nhmc_cont3") or name.startswith("nhmc_rg"):
            return cls(embed_dim, channels, n_leapfrog, lam)   # Z2 is built in
        else:
            kernel = cls(embed_dim, channels)

        if odd_nets:
            # Z2-equivariant nets (cf. Vega et al. 2025): odd force / momentum means, even widths
            for nm in ("fwd_mu", "bwd_mu", "learned_force"):
                setattr(kernel, nm, _Odd(getattr(kernel, nm)))
            for nm in ("fwd_sigma", "bwd_sigma"):
                setattr(kernel, nm, _Even(getattr(kernel, nm)))
        if use_z2:
            kernel = CondZ2Wrapper(kernel, num_steps)
    return kernel


def load_any(run_dir, ckpt, device):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    k = create_cond_kernel(cfg["kernel"], embed_dim=cfg.get("embed_dim", 128),
                           channels=tuple(cfg.get("channels", [32, 32])), n_leapfrog=cfg["n_leapfrog"],
                           lam=cfg["lam"], num_steps=cfg["num_steps"], use_z2=True,
                           odd_nets=cfg.get("odd_nets", False))
    k.load_state_dict(torch.load(os.path.join(run_dir, ckpt), map_location="cpu"))
    return k.eval().to(device), cfg



def load_model(run_dir, ckpt, device):
    """ckpt = file name in run_dir, 'choice:<json>' written by select_ckpt.py, or 'untrained:<seed>' (the run's
    architecture from config.json exactly as load_any builds it, PyTorch default init after
    torch.manual_seed(seed), no state dict loaded)."""
    if ckpt.startswith('choice:'):
        ckpt = json.load(open(ckpt[len('choice:'):]))['ckpt']
        print('checkpoint chosen by select_ckpt:', ckpt, flush=True)
    if ckpt.startswith('untrained:'):
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        torch.manual_seed(int(ckpt.split(':')[1]))
        k = create_cond_kernel(cfg['kernel'], embed_dim=cfg.get('embed_dim', 128),
                               channels=tuple(cfg.get('channels', [32, 32])), n_leapfrog=cfg['n_leapfrog'],
                               lam=cfg['lam'], num_steps=cfg['num_steps'], use_z2=True,
                               odd_nets=cfg.get('odd_nets', False)).eval().to(device)
        b = k.base
        k.init_info = dict(eps=float(b.epsilon) if hasattr(b, 'epsilon') else None,
                           log_eps=float(b.log_eps) if hasattr(b, 'log_eps') else None,
                           flip_prob=torch.sigmoid(k.flip_logits).tolist())
        print('untrained kernel', ckpt, k.init_info, flush=True)
    else:
        k, cfg = load_any(run_dir, ckpt, device)
    for p_ in k.parameters():
        p_.requires_grad_(False)
    k.ckpt_used = ckpt
    assert hasattr(k, 'flip_logits') and hasattr(k, 'reverse_step'), 'needs the CondZ2Wrapper kernel'
    return k, cfg
