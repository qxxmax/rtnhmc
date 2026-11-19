"""Extra particle targets with the LJClusterTarget interface (n_particles, dim, rm, temperature,
osc_scale, log_prob, force) for the NHMC LJ code path.

DW-4 (Koehler et al. 2020; iDEM / Adjoint-Sampling benchmark): 4 particles in 2D,
    E(x) = sum_{i<j} a (d_ij - d0)^4 + b (d_ij - d0)^2 + c,   a = 0.9, b = -4, c = 0, d0 = 4, T = 1,
identical to bgflow MultiDoubleWellPotential as configured in iDEM configs/energy/dw4.yaml and
adjoint_samplers energies/double_well_energy.py. Requires NHMC_NSPACE=2 (set before importing lj13_nhmc).
The energy is translation invariant, so forces are zero-COM by construction.
"""
import torch

import lj13_nhmc
from lj13_nhmc import FORCE_CLAMP


class DW4Target:
    def __init__(self, n_particles=4, a=0.9, b=-4.0, c=0.0, d0=4.0, temperature=1.0):
        if lj13_nhmc.N_SPACE != 2:
            raise RuntimeError("DW-4 needs NHMC_NSPACE=2 in the environment before importing lj13_nhmc")
        self.n_particles = int(n_particles)
        self.dim = self.n_particles * 2
        self.a, self.b, self.c, self.d0 = float(a), float(b), float(c), float(d0)
        self.temperature = float(temperature)
        self.rm = 1.0             # only used by LJ diagnostics (core-pair fraction); harmless here
        self.osc_scale = 0.0
        self.lj_scale = 0.0

    def _pairs(self, x):
        p = x.view(-1, self.n_particles, 2)
        iu = torch.triu_indices(self.n_particles, self.n_particles, offset=1, device=x.device)
        diff = p[:, iu[0]] - p[:, iu[1]]                       # [B, P, 2]
        d = torch.sqrt(diff.square().sum(-1) + 1e-12)          # [B, P]
        return p, iu, diff, d

    def energy(self, x):
        _, _, _, d = self._pairs(x)
        u = d - self.d0
        return (self.a * u ** 4 + self.b * u ** 2 + self.c).sum(-1)

    def log_prob(self, x):
        return -self.energy(x) / self.temperature

    def force(self, x):
        p, iu, diff, d = self._pairs(x)
        u = d - self.d0
        dEdd = 4.0 * self.a * u ** 3 + 2.0 * self.b * u           # [B, P]
        fij = -(dEdd / d)[..., None] * diff                       # force on i from pair (i, j)
        f = torch.zeros_like(p)
        f.index_add_(1, iu[0], fij)
        f.index_add_(1, iu[1], -fij)
        f = torch.nan_to_num(f, nan=0.0, posinf=FORCE_CLAMP, neginf=-FORCE_CLAMP)
        return f.reshape(x.shape).clamp(-FORCE_CLAMP, FORCE_CLAMP) / self.temperature


def make_target(args_or_cfg):
    """Build the target from argparse args or a config dict (keys: system, n_particles, ...)."""
    g = (lambda k, d=None: args_or_cfg.get(k, d)) if isinstance(args_or_cfg, dict) else \
        (lambda k, d=None: getattr(args_or_cfg, k, d))
    system = g('system', 'lj')
    if system == 'dw4':
        return DW4Target(temperature=g('temperature', 1.0))
    return lj13_nhmc.LJClusterTarget(n_particles=g('n_particles', 13), temperature=g('temperature', 1.0),
                                     osc_scale=g('osc_scale', 1.0), lj_scale=g('lj_scale', 0.5))
