"""2D phi^4 lattice action, force and observables.

S(phi) = sum_x [ (1 - 2 lambda) phi_x^2 + lambda phi_x^4 - 2 kappa sum_mu phi_x phi_{x+mu} ], periodic boundaries,
hopping parametrisation. Configurations have shape [B, L, L2] (action, force) or [B, 1, L, L2] (observables).
"""
import torch


def _kappa_to_broadcastable(kappa, cfgs):
    """Make kappa broadcastable against cfgs of shape (B, ...) or (B, 1, ...).
    Supports kappa as: float, 0-d tensor, or 1-d (B,) tensor.
    """
    if isinstance(kappa, torch.Tensor) and kappa.dim() == 1:
        # (B,) -> (B, 1, 1, ...) to match cfgs of (B, L, L) or (B, 1, L, L)
        extra_dims = cfgs.dim() - 1
        return kappa.view(-1, *([1] * extra_dims))
    return kappa


def phi4_action_density(cfgs, lam=0.022, kappa=0.27):
    """
    Per-site action density: S_x = (1-2λ)φ² + λφ⁴ - 2κ Σ_μ φ(x)φ(x+μ̂)
    Periodic boundary conditions via torch.roll.
    """
    k = _kappa_to_broadcastable(kappa, cfgs)
    density = (1 - 2 * lam) * cfgs.pow(2) + lam * cfgs.pow(4)
    for mu in range(1, len(cfgs.shape)):
        density = density - 2 * k * cfgs * torch.roll(cfgs, shifts=1, dims=mu)
    return density


def phi4_action(cfgs, lam=0.022, kappa=0.27):
    """Total action per configuration, shape [batch]."""
    density = phi4_action_density(cfgs, lam=lam, kappa=kappa)
    sum_dims = tuple(range(1, len(cfgs.shape)))
    return density.sum(dim=sum_dims)


def phi4_force(cfgs, lam=0.022, kappa=0.27):
    """
    Force F = -∂S/∂φ.
    F(x) = -[2(1-2λ)φ + 4λφ³] + 2κ Σ_μ [φ(x+μ̂) + φ(x-μ̂)]
    """
    k = _kappa_to_broadcastable(kappa, cfgs)
    grad_S = 2 * (1 - 2 * lam) * cfgs + 4 * lam * cfgs.pow(3)
    neighbor_sum = torch.zeros_like(cfgs)
    for mu in range(1, len(cfgs.shape)):
        neighbor_sum = neighbor_sum + torch.roll(cfgs, 1, dims=mu) + torch.roll(cfgs, -1, dims=mu)
    grad_S = grad_S - 2 * k * neighbor_sum
    return -grad_S


def observables(x, lam, kappa):
    V = x[0].numel()
    M = x.mean(dim=(1, 2, 3)).double()
    return {'absM': M.abs(), 'M2': M ** 2, 'M4': M ** 4,
            'action': phi4_action(x.squeeze(1), lam=lam, kappa=kappa).double() / V, 'sign': torch.sign(M)}
