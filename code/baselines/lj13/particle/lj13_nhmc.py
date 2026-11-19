#!/usr/bin/env python3
"""
Real neural NHMC for the Lennard-Jones LJ-13 cluster (bgflow target).

Target (Boltzmann, T=1):  log gamma(x) = -E(x),
    E(x) = sum_{i<j} eps[(rm/r_ij)^12 - 2 (rm/r_ij)^6]
           + 0.5 * osc_scale * || x - mean(x) ||^2          (oscillator confinement)
matching bgflow.distribution.energy.lennard_jones.LennardJonesPotential
(eps=1, rm=1, oscillator=True).

The leapfrog uses the ANALYTIC LJ + oscillator force (differentiable, so the
path-work training backprops through it with a single backward -- no costly
double-backward), with a safety clamp on the r^-12 core and per-step
mean-removal (LJ is translation invariant; we work in the zero-COM frame).
Only the per-step momentum law is learned (same design as the phi^4 kernel).

NOTE (recovery provenance): this file was reconstructed on 2026-07-11 from
Cursor's local editor history (.cursor-server/data/User/History/-31b899da/),
because the working copy had been deleted from disk while
__pycache__/lj13_nhmc.cpython-313.pyc and several downstream scripts
(train_lj13_nhmc.py, make_lj13_eval_fig.py, make_lj13_imh.py) still imported
it. Content is reproduced verbatim from the last recorded history snapshot;
no logic was changed.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributions import Normal

from neural_nhmc_dw import MomentumNet, ess_logz   # reuse

N_PART = 13
# Spatial dimension. 3 for LJ clusters (default, all legacy runs); the DW-4 benchmark (4 particles in 2D)
# sets NHMC_NSPACE=2 in the environment BEFORE importing this module (every module reads it at import).
N_SPACE = int(os.environ.get('NHMC_NSPACE', '3'))
DIM = N_PART * N_SPACE
COM_FREE_DIM = (N_PART - 1) * N_SPACE
FORCE_CLAMP = 1.0e3
MU_CLAMP = 20.0          # bound learned momentum mean -> positions stay finite
R2_FLOOR = 0.49          # soften repulsive core at r < 0.7 rm (avoids fp32 overflow)
_EPS = 1.0e-9


def _infer_n_particles(x):
    if x.shape[-1] % N_SPACE != 0:
        raise ValueError(f"Cartesian dimension {x.shape[-1]} is not divisible by {N_SPACE}")
    return x.shape[-1] // N_SPACE


def remove_mean(x, n_particles=None):
    n_particles = _infer_n_particles(x) if n_particles is None else int(n_particles)
    p = x.view(-1, n_particles, N_SPACE)
    p = p - p.mean(dim=1, keepdim=True)
    return p.reshape(x.shape)


def icosahedral_template(dtype=torch.float32):
    """Ideal LJ13 cluster: one center plus the twelve icosahedron vertices.

    The raw regular icosahedron has edge length two. Dividing by two makes
    nearest-neighbor vertex distances one, matching the LJ equilibrium scale.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            vertices.append((0.0, a, b))
            vertices.append((a, b, 0.0))
            vertices.append((b, 0.0, a))
    points = torch.tensor([(0.0, 0.0, 0.0), *vertices], dtype=dtype) / 2.0
    return points - points.mean(dim=0, keepdim=True)


def random_rotation_matrices(batch, device, dtype):
    """Haar-uniform SO(3) matrices from uniformly sampled unit quaternions."""
    u1, u2, u3 = torch.rand(3, batch, device=device, dtype=dtype)
    x = torch.sqrt(1.0 - u1) * torch.sin(2.0 * math.pi * u2)
    y = torch.sqrt(1.0 - u1) * torch.cos(2.0 * math.pi * u2)
    z = torch.sqrt(u1) * torch.sin(2.0 * math.pi * u3)
    w = torch.sqrt(u1) * torch.cos(2.0 * math.pi * u3)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(batch, 3, 3)


class IcosahedralConditionalBase(nn.Module):
    """Tractable LJ13 base conditional on rotation and particle permutation.

    The auxiliary rotation and permutation use the same uniform laws in the
    forward and reverse extended path measures, so their probabilities cancel.
    Conditional on those auxiliaries, the position density is an isotropic
    Gaussian on the 36-dimensional zero-center-of-mass subspace.
    """

    def __init__(self, noise_scale=0.16, template_scale=1.0, learn=True):
        super().__init__()
        self.register_buffer('template', icosahedral_template())
        if learn:
            self.log_noise_scale = nn.Parameter(torch.tensor(math.log(noise_scale)))
            self.log_template_scale = nn.Parameter(torch.tensor(math.log(template_scale)))
        else:
            self.register_buffer('log_noise_scale', torch.tensor(math.log(noise_scale)))
            self.register_buffer('log_template_scale', torch.tensor(math.log(template_scale)))

    @property
    def noise_scale(self):
        return torch.exp(self.log_noise_scale).clamp(0.03, 0.60)

    @property
    def template_scale(self):
        return torch.exp(self.log_template_scale).clamp(0.70, 1.30)

    def sample(self, batch, device, dtype):
        template = self.template.to(device=device, dtype=dtype) * self.template_scale
        centers = template[None].expand(batch, -1, -1)
        permutation = torch.argsort(torch.rand(batch, N_PART, device=device), dim=1)
        centers = torch.gather(
            centers, 1, permutation[..., None].expand(-1, -1, N_SPACE)
        )
        rotation = random_rotation_matrices(batch, device, dtype)
        centers = torch.bmm(centers, rotation.transpose(1, 2))
        centers = remove_mean(centers.reshape(batch, DIM))
        noise = remove_mean(torch.randn(batch, DIM, device=device, dtype=dtype))
        x = centers + self.noise_scale * noise
        log_prob = base_log_prob_scaled(x - centers, self.noise_scale)
        return x, log_prob, centers


def load_cartesian_template(path, n_particles, coordinate_scale=1.0):
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields:
            rows.append([float(value) for value in fields[:3]])
    if len(rows) != n_particles:
        raise ValueError(f"template {path} has {len(rows)} rows, expected {n_particles}")
    template = torch.tensor(rows, dtype=torch.get_default_dtype()) * coordinate_scale
    return template - template.mean(dim=0, keepdim=True)


class MultiTemplateConditionalBase(nn.Module):
    """Zero-COM Gaussian base conditional on a template and rigid relabeling.

    The template index, Haar rotation, and particle permutation are explicit
    auxiliary variables. Their common reference laws cancel in the path ratio;
    conditional on them, the Cartesian density is a normalized Gaussian on the
    zero-center-of-mass subspace.
    """

    def __init__(self, templates, noise_scale=0.16, template_scale=1.0, learn=True):
        super().__init__()
        templates = torch.as_tensor(templates, dtype=torch.get_default_dtype())
        if templates.ndim != 3 or templates.shape[-1] != N_SPACE:
            raise ValueError("templates must have shape [components, particles, 3]")
        templates = templates - templates.mean(dim=1, keepdim=True)
        self.register_buffer('templates', templates)
        self.n_components, self.n_particles, _ = templates.shape
        if learn:
            self.log_noise_scale = nn.Parameter(torch.tensor(math.log(noise_scale)))
            self.log_template_scale = nn.Parameter(torch.tensor(math.log(template_scale)))
        else:
            self.register_buffer('log_noise_scale', torch.tensor(math.log(noise_scale)))
            self.register_buffer('log_template_scale', torch.tensor(math.log(template_scale)))

    @property
    def noise_scale(self):
        return torch.exp(self.log_noise_scale).clamp(0.02, 0.80)

    @property
    def template_scale(self):
        return torch.exp(self.log_template_scale).clamp(0.70, 1.30)

    def sample(self, batch, device, dtype):
        component = torch.randint(self.n_components, (batch,), device=device)
        templates = self.templates.to(device=device, dtype=dtype)
        centers = templates[component] * self.template_scale
        permutation = torch.argsort(
            torch.rand(batch, self.n_particles, device=device), dim=1
        )
        centers = torch.gather(
            centers, 1, permutation[..., None].expand(-1, -1, N_SPACE)
        )
        rotation = random_rotation_matrices(batch, device, dtype)
        centers = torch.bmm(centers, rotation.transpose(1, 2))
        centers = remove_mean(centers.reshape(batch, self.n_particles * N_SPACE))
        noise = remove_mean(
            torch.randn(batch, self.n_particles * N_SPACE, device=device, dtype=dtype)
        )
        x = centers + self.noise_scale * noise
        log_prob = base_log_prob_scaled(x - centers, self.noise_scale)
        return x, log_prob, centers, component


class ContextMomentumNet(nn.Module):
    """Momentum MLP with an explicit scalar base-component condition."""

    def __init__(self, dim, hidden=128, n_freq=8, out_bias=0.0):
        super().__init__()
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(dim + 2 * n_freq + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, out_bias)

    def forward(self, x, t_norm, context):
        frequencies = torch.arange(
            1, self.n_freq + 1, device=x.device, dtype=x.dtype
        )
        angle = 2.0 * math.pi * t_norm[:, None] * frequencies[None]
        time_embedding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        return self.net(torch.cat([x, time_embedding, context[:, None]], dim=1))


class EquivariantMomentumLaw(nn.Module):
    """Permutation- and E(3)-equivariant diagonal Gaussian momentum law.

    Scalar node states are built only from invariant distances and time. The
    momentum mean is a sum of relative-position vectors multiplied by
    symmetric scalar edge coefficients; the per-particle scale is invariant
    and shared across its Cartesian components. The resulting full-space
    Gaussian density remains available in closed form.
    """

    def __init__(self, n_particles=N_PART, hidden=128, n_layers=3, n_rbf=12, r_max=4.0,
                 log_sigma_bias=-0.7):
        super().__init__()
        self.n_particles = int(n_particles)
        self.hidden = hidden
        self.n_layers = n_layers
        centers = torch.linspace(0.0, r_max, n_rbf)
        self.register_buffer('rbf_centers', centers)
        self.rbf_width = float((centers[1] - centers[0]) if n_rbf > 1 else r_max)
        self.node_init = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        edge_dim = 2 * hidden + n_rbf + 1
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
            ) for _ in range(n_layers)
        ])
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(n_layers)
        ])
        # Symmetric edge features make the vector coefficients symmetric in
        # i,j, so the predicted mean has exactly zero center of mass.
        vector_edge_dim = 2 * hidden + n_rbf + 1
        self.vector_head = nn.Sequential(
            nn.Linear(vector_edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.vector_head[-1].weight)
        nn.init.zeros_(self.vector_head[-1].bias)
        nn.init.zeros_(self.log_sigma_head[-1].weight)
        nn.init.constant_(self.log_sigma_head[-1].bias, log_sigma_bias)

    def _rbf(self, distance):
        width = max(self.rbf_width, 1.0e-3)
        return torch.exp(-0.5 * ((distance[..., None] - self.rbf_centers) / width) ** 2)

    def forward(self, x, t):
        n_particles = self.n_particles
        positions = x.view(-1, n_particles, N_SPACE)
        batch = positions.shape[0]
        relative = positions[:, :, None, :] - positions[:, None, :, :]
        distance = torch.sqrt(relative.square().sum(-1) + _EPS)
        rbf = self._rbf(distance)
        mask = (~torch.eye(n_particles, device=x.device, dtype=torch.bool))[None, :, :, None]
        time_node = t[:, None, None].expand(batch, n_particles, 1)
        radius2 = positions.square().sum(-1, keepdim=True)
        h = self.node_init(torch.cat([radius2, time_node], dim=-1))

        time_edge = t[:, None, None, None].expand(batch, n_particles, n_particles, 1)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            hi = h[:, :, None, :].expand(-1, -1, n_particles, -1)
            hj = h[:, None, :, :].expand(-1, n_particles, -1, -1)
            message = edge_mlp(torch.cat([hi, hj, rbf, time_edge], dim=-1))
            aggregate = (message * mask).sum(dim=2) / (n_particles - 1)
            h = h + node_mlp(torch.cat([h, aggregate], dim=-1))

        hi = h[:, :, None, :].expand(-1, -1, n_particles, -1)
        hj = h[:, None, :, :].expand(-1, n_particles, -1, -1)
        symmetric_edge = torch.cat([hi + hj, (hi - hj).abs(), rbf, time_edge], dim=-1)
        coefficient = self.vector_head(symmetric_edge) * mask
        mu = (coefficient * relative).sum(dim=2) / (n_particles - 1)
        mu = mu.reshape(x.shape).clamp(-MU_CLAMP, MU_CLAMP)
        log_sigma = self.log_sigma_head(h).expand(-1, -1, N_SPACE).reshape(x.shape)
        return mu, log_sigma.clamp(-5.0, 2.0)


def bound_log_sigma(z, soft=False, lo=-5.0, hi=2.0):
    """Hard clamp (legacy) or smooth clamp lo + softplus(z-lo) - softplus(z-hi). The hard clamp has zero
    gradient outside [lo, hi]; on DW-4 the forward-law log sigma was driven to the upper bound (sigma = e^2)
    by the entropy term of W and could never come back (2026-10-12 collapse). The smooth clamp keeps a
    restoring gradient near the bounds."""
    if not soft:
        return z.clamp(lo, hi)
    return lo + torch.nn.functional.softplus(z - lo) - torch.nn.functional.softplus(z - hi)


class ForceEquivariantMomentumLaw(nn.Module):
    """Higher-capacity E(3)- and permutation-equivariant momentum law.

    The scalar message-passing state is built from pair distances, particle
    radii in the zero-COM frame, and invariant contractions with the target
    force.  The Gaussian mean is expanded in three equivariant vector bases:
    pair displacements, centered positions, and the target force.  The scale
    remains invariant and is shared across Cartesian components of a particle,
    so the conditional density is still available in closed form.
    """

    def __init__(self, target, hidden=128, n_layers=5, n_rbf=24, r_max=4.0,
                 n_time_freq=4, force_scale=20.0, log_sigma_bias=-0.7, use_force=True):
        super().__init__()
        self.target = target
        # use_force=False: the law never calls target.force (force features and the force basis vector
        # are zero); inputs are positions, pair distances and the time/beta embedding only.
        self.use_force = bool(use_force)
        self.n_particles = int(target.n_particles)
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_time_freq = n_time_freq
        self.force_scale = float(force_scale)
        centers = torch.linspace(0.0, r_max, n_rbf)
        self.register_buffer('rbf_centers', centers)
        self.rbf_width = float((centers[1] - centers[0]) if n_rbf > 1 else r_max)

        time_dim = 2 * n_time_freq
        self.node_init = nn.Sequential(
            nn.Linear(3 + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        edge_dim = 2 * hidden + n_rbf + time_dim
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
            ) for _ in range(n_layers)
        ])
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden, 2 * hidden), nn.SiLU(),
                nn.Linear(2 * hidden, hidden),
            ) for _ in range(n_layers)
        ])
        self.node_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        vector_edge_dim = 2 * hidden + n_rbf + time_dim
        self.vector_edge_head = nn.Sequential(
            nn.Linear(vector_edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.vector_node_head = nn.Sequential(
            nn.Linear(hidden + time_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2)
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(hidden + time_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        for head in (self.vector_edge_head, self.vector_node_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.log_sigma_head[-1].weight)
        nn.init.constant_(self.log_sigma_head[-1].bias, log_sigma_bias)

    def _rbf(self, distance):
        width = max(self.rbf_width, 1.0e-3)
        return torch.exp(-0.5 * ((distance[..., None] - self.rbf_centers) / width) ** 2)

    def _time_embedding(self, t):
        frequencies = torch.arange(
            1, self.n_time_freq + 1, device=t.device, dtype=t.dtype
        )
        angle = 2.0 * math.pi * t[:, None] * frequencies[None]
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

    def forward(self, x, t):
        n_particles = self.n_particles
        positions = remove_mean(x).view(-1, n_particles, N_SPACE)
        batch = positions.shape[0]
        if self.use_force:
            force = remove_mean(self.target.force(x)).view(-1, n_particles, N_SPACE)
        else:
            force = torch.zeros_like(positions)
        force_norm = torch.sqrt(force.square().sum(-1, keepdim=True) + _EPS)
        bounded_force = force / (1.0 + force_norm / self.force_scale)

        relative = positions[:, :, None, :] - positions[:, None, :, :]
        distance = torch.sqrt(relative.square().sum(-1) + _EPS)
        rbf = self._rbf(distance)
        mask = (~torch.eye(n_particles, device=x.device, dtype=torch.bool))[None, :, :, None]

        time = self._time_embedding(t)
        time_node = time[:, None, :].expand(batch, n_particles, -1)
        radius2 = positions.square().sum(-1, keepdim=True)
        force2 = torch.log1p(force.square().sum(-1, keepdim=True))
        position_force = (positions * bounded_force).sum(-1, keepdim=True)
        h = self.node_init(torch.cat([radius2, force2, position_force, time_node], dim=-1))

        time_edge = time[:, None, None, :].expand(batch, n_particles, n_particles, -1)
        for edge_mlp, node_mlp, node_norm in zip(
            self.edge_mlps, self.node_mlps, self.node_norms
        ):
            hi = h[:, :, None, :].expand(-1, -1, n_particles, -1)
            hj = h[:, None, :, :].expand(-1, n_particles, -1, -1)
            symmetric = torch.cat([hi + hj, (hi - hj).abs(), rbf, time_edge], dim=-1)
            message = edge_mlp(symmetric)
            aggregate = (message * mask).sum(dim=2) / (n_particles - 1)
            update = node_mlp(torch.cat([h, aggregate], dim=-1))
            h = node_norm(h + update)

        hi = h[:, :, None, :].expand(-1, -1, n_particles, -1)
        hj = h[:, None, :, :].expand(-1, n_particles, -1, -1)
        symmetric_edge = torch.cat([hi + hj, (hi - hj).abs(), rbf, time_edge], dim=-1)
        edge_coefficient = self.vector_edge_head(symmetric_edge) * mask
        pair_vector = (edge_coefficient * relative).sum(dim=2) / (n_particles - 1)

        node_input = torch.cat([h, time_node], dim=-1)
        node_coefficients = self.vector_node_head(node_input)
        mu = (pair_vector
              + node_coefficients[..., :1] * positions
              + node_coefficients[..., 1:] * bounded_force)
        mc = getattr(self, 'mu_clamp', MU_CLAMP)
        mu = remove_mean(mu.reshape(x.shape)).clamp(-mc, mc)
        log_sigma = self.log_sigma_head(node_input)
        log_sigma = log_sigma.expand(-1, -1, N_SPACE).reshape(x.shape)
        return mu, bound_log_sigma(log_sigma, getattr(self, 'soft_sigma', False))


class LJClusterTarget:
    def __init__(self, n_particles=N_PART, eps=1.0, rm=1.0, osc_scale=1.0,
                 temperature=1.0, lj_scale=0.5):
        self.n_particles = int(n_particles)
        self.dim = self.n_particles * N_SPACE
        self.eps, self.rm, self.osc_scale = eps, rm, osc_scale
        # lj_scale multiplies the LJ term summed over ORDERED pairs (each pair
        # counted twice by lj.sum over the full NxN matrix). lj_scale=0.5 = "each
        # pair once" = our historical soft potential (default, keeps every prior
        # run/artifact bit-identical). lj_scale=1.0 = the official iDEM / Adjoint
        # Sampling / Tilt Matching LennardJonesPotential (energy_factor=1.0),
        # i.e. exactly 2x deeper LJ interaction. The harmonic oscillator and
        # temperature are unchanged.
        self.lj_scale = float(lj_scale)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def _pair(self, x):
        p = x.view(-1, self.n_particles, N_SPACE)
        diff = p[:, :, None, :] - p[:, None, :, :]          # [B,N,N,3]
        r2 = (diff ** 2).sum(-1)                            # [B,N,N]
        return p, diff, r2

    def log_prob(self, x):
        p, diff, r2 = self._pair(x)
        eye = torch.eye(self.n_particles, device=x.device, dtype=torch.bool)
        r2 = r2.masked_fill(eye[None], float('inf'))
        inv6 = (self.rm ** 2 / r2.clamp_min(_EPS)) ** 3
        lj = self.eps * (inv6 ** 2 - 2.0 * inv6)            # [B,N,N]
        e_lj = lj.sum(dim=(1, 2)) * self.lj_scale           # 0.5=each pair once, 1.0=official
        pc = p - p.mean(dim=1, keepdim=True)
        e_osc = 0.5 * self.osc_scale * pc.pow(2).sum(dim=(1, 2))
        return -(e_lj + e_osc) / self.temperature

    def force(self, x, soft=None):
        """F = -grad E  (analytic, differentiable), zero-COM. soft: optional per-sample r^2 shift [B]
        (soft-core bridge force, r^2 -> r^2 + soft; soft=0 is the exact force)."""
        p, diff, r2 = self._pair(x)
        if soft is not None:
            r2 = r2 + soft.to(r2.dtype).view(-1, 1, 1)
        eye = torch.eye(self.n_particles, device=x.device, dtype=torch.bool)
        # Floor r^2 at a physical core (~0.7 rm) so the repulsive 1/r^14 term cannot
        # overflow fp32 -> inf -> NaN on near-collisions. This only softens the force
        # at sub-physical separations (Boltzmann weight ~ 0 there); the exact LJ energy
        # used for S / evaluation is untouched, so the path-work W stays exact.
        r2 = r2.masked_fill(eye[None], float('inf')).clamp_min(R2_FLOOR * self.rm ** 2)
        inv2 = self.rm ** 2 / r2
        inv6 = inv2 ** 3
        inv12 = inv6 ** 2
        # g(r) such that F_i = sum_j g * (x_i - x_j). For lj_scale=0.5 (each pair
        # once) g = 12 eps (rm^12/r^14 - rm^6/r^8); the general force is that base
        # scaled by lj_scale/0.5, i.e. coeff = 24 eps lj_scale (so lj_scale=1.0
        # official gives exactly 2x the force, consistent with the 2x energy).
        g = 24.0 * self.eps * self.lj_scale * (inv12 - inv6) / r2   # [B,N,N]
        f_lj = (g[..., None] * diff).sum(dim=2)             # [B,N,3]
        pc = p - p.mean(dim=1, keepdim=True)
        f = f_lj - self.osc_scale * pc                      # oscillator force
        f = torch.nan_to_num(f, nan=0.0, posinf=FORCE_CLAMP, neginf=-FORCE_CLAMP)
        f = f.reshape(x.shape).clamp(-FORCE_CLAMP, FORCE_CLAMP)
        return f / self.temperature


class LJ13Target(LJClusterTarget):
    def __init__(self, eps=1.0, rm=1.0, osc_scale=1.0, temperature=1.0,
                 lj_scale=0.5):
        super().__init__(
            N_PART, eps=eps, rm=rm, osc_scale=osc_scale, temperature=temperature,
            lj_scale=lj_scale,
        )


class LJ13NHMCKernel(nn.Module):
    """Learned-momentum leapfrog with the true LJ force, zero-COM frame."""

    def __init__(self, target: LJ13Target, n_leapfrog=5, hidden=128,
                 log_eps_init=-6.2, anneal_power=1.0, anneal_floor=0.05,
                 anneal_base=True, learn_base=True, base_scale=1.2,
                 num_steps=None, stagewise_eps=False, learn_schedule=False,
                 architecture='mlp', equiv_layers=3, equiv_rbf=12,
                 equiv_time_freq=4, equiv_force_scale=20.0,
                 equiv_r_max=4.0,
                 checkpoint_equiv=True, schedule_type='power',
                 base_type='gaussian', structured_noise_scale=0.16,
                 structured_template_scale=1.0, structured_mix_prob=0.5,
                 condition_on_component=False, template_files=None,
                 template_coordinate_scale=1.0, schedule_increment_floor=0.0,
                 momentum_refresh_alpha=None, learn_mass=False, mass_init=1.0,
                 momentum_mode='full', eps_max=1e-2, drift='exact', law_force_input=True,
                 drift_hidden=64, drift_layers=3, soft_sigma=False, drift_arch='egnn',
                 drift_lmax=2, drift_corr=3, drift_pair=False, drift_clamp=MU_CLAMP,
                 softcore_alpha=0.0, softcore_power=1.0, softcore_learn=0):
        super().__init__()
        self.target = target
        # Soft-core bridge (2026-10-18): the leapfrog force uses LJ with r^2 -> r^2 + alpha (1-beta)^power
        # (alchemical soft core, Beutler et al. 1994). W only contains the endpoint energy, so any such
        # volume-preserving dynamics keeps the weights exact; it removes the r^-12 singularity that the
        # geometric Gaussian->LJ bridge hits at small beta. alpha=0 = legacy exact force.
        self.softcore_alpha = float(softcore_alpha)
        self.softcore_power = float(softcore_power)
        # Learned core (NHMC-specific): because W never evaluates intermediate densities, the core width is
        # a free dynamics parameter. alpha(beta) = alpha0 (1-beta)^power exp(h(beta)), h piecewise linear on
        # softcore_learn knots (zero-init = the fixed soft core), trained by the same path objective.
        self.softcore_learn = int(softcore_learn)
        if self.softcore_learn > 0:
            self.softcore_knots = nn.Parameter(torch.zeros(self.softcore_learn))
        self.eps_max = float(eps_max)     # upper clamp of the leapfrog step (1e-2 = legacy LJ value)
        self.dim = target.dim
        self.n_particles = target.n_particles
        self.n_leapfrog = n_leapfrog
        # ----- momentum-law ablation (NHMC mechanism study) -----
        # 'full'         : learned forward AND backward momentum laws (default,
        #                  the complete NHMC; reproduces all legacy checkpoints).
        # 'forward_only' : learned forward law, backward law fixed to N(0, I).
        # 'fixed'        : both laws fixed to standard normal N(0, I); only the
        #                  annealed leapfrog / step-size / base are learned.
        # A fixed law returns mu=0, log_sigma=0 (sigma=1), which is a proper
        # normalized density, so the path-work importance weight stays exact.
        if momentum_mode not in ('full', 'forward_only', 'fixed'):
            raise ValueError(f"unknown momentum_mode: {momentum_mode}")
        self.momentum_mode = momentum_mode
        # annealing schedule for the force: beta(t) ramps from ~floor to 1 over
        # the L steps so the base -> target transport is gradual (key for LJ).
        self.anneal_power = anneal_power
        self.anneal_floor = anneal_floor
        self.num_steps = int(num_steps) if num_steps is not None else None
        self.stagewise_eps = bool(stagewise_eps)
        self.learn_schedule = bool(learn_schedule)
        self.architecture = architecture
        self.checkpoint_equiv = bool(checkpoint_equiv)
        self.schedule_type = schedule_type
        self.base_type = base_type
        # Anti-collapse floor for the LEARNED schedule: each stage is
        # guaranteed at least `floor/num_steps` of the total beta ramp, so the
        # softmax-style increments cannot concentrate all annealing progress
        # into a few stages (the historical U(1) failure mode).  floor=0
        # reproduces the previous behaviour exactly.
        self.schedule_increment_floor = float(schedule_increment_floor)
        # ----- underdamped / partial momentum refresh (direction 1) -----
        # None  -> legacy code path (full refresh every stage), untouched.
        # alpha in [0,1) -> persistent momentum with Crank-Nicolson refresh
        # anchored at the learned mean (see UNDERDAMPED_NHMC_NOTES.md):
        #   forward:  p~_t = mu_f + alpha (p'_{t-1} - mu_f) + sqrt(1-a^2) sig_f eps
        #   backward: -p'_{t-1} ~ N((1-a) mu_b + a (-p~_t), (1-a^2) sig_b^2)
        # alpha = 0 reproduces the legacy full-refresh densities exactly.
        if momentum_refresh_alpha is not None:
            momentum_refresh_alpha = float(momentum_refresh_alpha)
            if not (0.0 <= momentum_refresh_alpha < 1.0):
                raise ValueError("momentum_refresh_alpha must be in [0, 1)")
        self.momentum_refresh_alpha = momentum_refresh_alpha
        # ----- learned scalar mass (direction 2) -----
        # Permutation + E(3) equivariance restrict any constant mass matrix on
        # LJ13 to a global scalar; leapfrog uses x += eps p / m in forward AND
        # reverse, so the path weight stays exact for any m.
        self.learn_mass = bool(learn_mass)
        if self.learn_mass:
            self.log_mass = nn.Parameter(torch.tensor(math.log(float(mass_init))))
        else:
            # persistent=False keeps legacy kernel_best.pt checkpoints loadable
            # with strict=True (they predate log_mass).
            self.register_buffer(
                'log_mass', torch.tensor(math.log(float(mass_init))),
                persistent=False,
            )
        self.structured_mix_prob = float(structured_mix_prob)
        self.condition_on_component = bool(condition_on_component)
        if self.n_particles != N_PART and base_type in ('icosahedral', 'mixture'):
            raise ValueError("icosahedral and mixture bases are currently defined only for LJ13")
        # geometric-path dynamics: force = (1-beta) grad log q0 + beta grad log gamma.
        # The (1-beta) base-gradient term makes early steps a stable OU relaxation
        # toward the base instead of slamming the full LJ force on a clashing gas.
        # This only changes the kernel; W still uses the q0/gamma endpoints, so the
        # path-work estimator stays unbiased for any choice here.
        self.anneal_base = anneal_base
        self.learn_base = learn_base
        self._base_scale_const = float(base_scale)
        if base_type in ('gaussian', 'mixture'):
            if learn_base:
                self.log_base_scale = nn.Parameter(torch.tensor(math.log(base_scale)))
        if base_type in ('icosahedral', 'mixture'):
            self.structured_base = IcosahedralConditionalBase(
                noise_scale=structured_noise_scale,
                template_scale=structured_template_scale,
                learn=learn_base,
            )
        if base_type == 'templates':
            if not template_files:
                raise ValueError("template_files are required for base_type='templates'")
            templates = torch.stack([
                load_cartesian_template(
                    path, self.n_particles, coordinate_scale=template_coordinate_scale
                )
                for path in template_files
            ])
            self.structured_base = MultiTemplateConditionalBase(
                templates,
                noise_scale=structured_noise_scale,
                template_scale=structured_template_scale,
                learn=learn_base,
            )
        if base_type not in ('gaussian', 'icosahedral', 'mixture', 'templates'):
            raise ValueError(f'unknown base type: {base_type}')
        if architecture == 'mlp':
            net = ContextMomentumNet if self.condition_on_component else MomentumNet
            self.fwd_mu = net(self.dim, hidden, out_bias=0.0)
            self.fwd_logsig = net(self.dim, hidden, out_bias=-0.7)
            self.bwd_mu = net(self.dim, hidden, out_bias=0.0)
            self.bwd_logsig = net(self.dim, hidden, out_bias=-0.7)
        elif architecture == 'equivariant':
            self.fwd_law = EquivariantMomentumLaw(
                self.n_particles, hidden, equiv_layers, equiv_rbf,
                r_max=equiv_r_max,
            )
            self.bwd_law = EquivariantMomentumLaw(
                self.n_particles, hidden, equiv_layers, equiv_rbf,
                r_max=equiv_r_max,
            )
        elif architecture == 'force_equivariant':
            self.fwd_law = ForceEquivariantMomentumLaw(
                target, hidden, equiv_layers, equiv_rbf, r_max=equiv_r_max,
                n_time_freq=equiv_time_freq, force_scale=equiv_force_scale,
                use_force=law_force_input,
            )
            self.bwd_law = ForceEquivariantMomentumLaw(
                target, hidden, equiv_layers, equiv_rbf, r_max=equiv_r_max,
                n_time_freq=equiv_time_freq, force_scale=equiv_force_scale,
                use_force=law_force_input,
            )
        else:
            raise ValueError(f'unknown momentum architecture: {architecture}')
        # ----- learned deterministic drift (2026-10-12) -----
        # 'exact'         : leapfrog kick = exact bridge force (legacy).
        # 'learned'       : kick = (1-beta) grad log q0 + b_phi(x, beta); the target force is NEVER called
        #                   inside the integrator. b_phi is an E(n)- and permutation-equivariant, zero-COM
        #                   vector field (pair-displacement + radial basis, no force input). Any b_phi(x)
        #                   that depends on position (and the stage) only keeps kick-drift-kick a composition
        #                   of shears: volume preserving and reversible (Phi^-1 = R Phi R), so the path
        #                   work W and round-trip MH stay exact. It is NOT a gradient field in general.
        # 'exact+learned' : exact bridge force + b_phi (learned residual).
        for law in (getattr(self, 'fwd_law', None), getattr(self, 'bwd_law', None)):
            if law is not None:
                law.soft_sigma = bool(soft_sigma)
        if drift not in ('exact', 'learned', 'exact+learned'):
            raise ValueError(f"unknown drift: {drift}")
        self.drift = drift
        self.law_force_input = bool(law_force_input)
        # drift_arch (2026-10-14): 'egnn' = ForceEquivariantMomentumLaw backbone (default, legacy);
        # 'painn' / 'mace' = equiv_drift.PaiNNDrift / MACEDrift (same interface, no force input).
        self.drift_arch = drift_arch
        if drift != 'exact':
            if drift_arch == 'egnn':
                self.drift_net = ForceEquivariantMomentumLaw(
                    target, drift_hidden, drift_layers, equiv_rbf, r_max=equiv_r_max,
                    n_time_freq=equiv_time_freq, force_scale=equiv_force_scale, use_force=False,
                )
            elif drift_arch == 'painn':
                from equiv_drift import PaiNNDrift
                self.drift_net = PaiNNDrift(target.n_particles, drift_hidden, drift_layers, equiv_rbf,
                                            r_max=equiv_r_max, n_time_freq=equiv_time_freq)
            elif drift_arch == 'mace':
                from equiv_drift import MACEDrift
                self.drift_net = MACEDrift(target.n_particles, drift_hidden, drift_layers, max_ell=drift_lmax,
                                           correlation=drift_corr, r_max=equiv_r_max, n_time_freq=equiv_time_freq)
            else:
                raise ValueError(f'unknown drift_arch: {drift_arch}')
            # drift_clamp (2026-10-17): MU_CLAMP=20 was inherited from the momentum-mean bound; an LJ
            # pair at r=0.9 already needs ~140, so for singular targets the drift bound is raised.
            # drift_pair: + unbounded radial pair term (equiv_drift.PairRadialDrift), no target calls.
            self.drift_clamp = float(drift_clamp)
            if hasattr(self.drift_net, 'log_sigma_head'):
                self.drift_net.mu_clamp = self.drift_clamp
            self.drift_pair = bool(drift_pair)
            if self.drift_pair:
                from equiv_drift import PairRadialDrift
                self.pair_drift = PairRadialDrift(target.n_particles, n_rbf=equiv_rbf, r_max=equiv_r_max,
                                                  n_time_freq=equiv_time_freq)
        if self.stagewise_eps:
            if self.num_steps is None:
                raise ValueError("num_steps is required for stage-wise step sizes")
            self.log_epsilon = nn.Parameter(
                torch.full((self.num_steps,), float(log_eps_init))
            )
        else:
            self.log_epsilon = nn.Parameter(torch.tensor(float(log_eps_init)))

        if self.learn_schedule:
            if self.num_steps is None:
                raise ValueError("num_steps is required for a learned schedule")
            fractions = torch.arange(1, self.num_steps + 1, dtype=torch.float64)
            fractions = (fractions / self.num_steps).pow(self.anneal_power)
            cumulative = self.anneal_floor + (1.0 - self.anneal_floor) * fractions
            increments = torch.diff(
                torch.cat([torch.tensor([self.anneal_floor], dtype=torch.float64), cumulative])
            ).clamp_min(1.0e-6)
            # Positive increments normalized by their sum give a monotone schedule.
            schedule_logits = torch.log(torch.expm1(increments / increments.mean()))
            self.schedule_logits = nn.Parameter(schedule_logits.to(torch.get_default_dtype()))

    @property
    def epsilon(self):
        return torch.exp(self.log_epsilon).clamp(1e-5, getattr(self, 'eps_max', 1e-2))

    def epsilon_at(self, t, num_steps):
        eps = self.epsilon
        if eps.ndim == 0:
            return eps
        if eps.numel() != num_steps:
            raise ValueError(f"step-size vector has {eps.numel()} entries, expected {num_steps}")
        return eps[t - 1]

    @property
    def base_scale(self):
        if self.base_type in ('icosahedral', 'templates'):
            return self.structured_base.noise_scale
        if self.learn_base:
            return torch.exp(self.log_base_scale).clamp(0.3, 5.0)
        return self._base_scale_const

    def sample_base(self, batch, device, dtype, base_noise=None):
        if self.base_type == 'icosahedral':
            x, log_prob, center = self.structured_base.sample(batch, device, dtype)
            component = torch.ones(batch, device=device, dtype=dtype)
            return x, log_prob, center, self.structured_base.noise_scale, component
        if self.base_type == 'templates':
            x, log_prob, center, component = self.structured_base.sample(
                batch, device, dtype
            )
            return (
                x,
                log_prob,
                center,
                self.structured_base.noise_scale,
                component.to(dtype),
            )
        scale = self.base_scale
        if base_noise is None:
            base_noise = torch.randn(batch, self.dim, device=device, dtype=dtype)
        else:
            base_noise = base_noise.to(device=device, dtype=dtype)
            if base_noise.shape != (batch, self.dim):
                raise ValueError(
                    f"base noise has shape {tuple(base_noise.shape)}, expected {(batch, self.dim)}"
                )
        x_gaussian = remove_mean(base_noise * scale)
        logq_gaussian = base_log_prob_scaled(x_gaussian, scale)
        if self.base_type == 'gaussian':
            component = torch.zeros(batch, device=device, dtype=dtype)
            return x_gaussian, logq_gaussian, None, scale, component

        x_structured, logq_structured, center = self.structured_base.sample(
            batch, device, dtype
        )
        use_structured = (
            torch.rand(batch, device=device) < self.structured_mix_prob
        )
        choose = use_structured[:, None]
        x = torch.where(choose, x_structured, x_gaussian)
        log_prob = torch.where(use_structured, logq_structured, logq_gaussian)
        center = torch.where(choose, center, torch.zeros_like(center))
        structured_scale = self.structured_base.noise_scale.to(dtype=dtype)
        scale_context = torch.where(
            use_structured[:, None],
            structured_scale.expand(batch, 1),
            torch.as_tensor(scale, device=device, dtype=dtype).expand(batch, 1),
        )
        return x, log_prob, center, scale_context, use_structured.to(dtype)

    def _beta(self, t, num_steps):
        if self.learn_schedule:
            if self.schedule_logits.numel() != num_steps:
                raise ValueError(
                    f"schedule has {self.schedule_logits.numel()} entries, expected {num_steps}"
                )
            increments = torch.nn.functional.softplus(self.schedule_logits)
            fractions = increments / increments.sum()
            if self.schedule_increment_floor > 0.0:
                # Convex mix with the uniform schedule: every stage keeps at
                # least floor/num_steps of the ramp, learned part fills the rest.
                floor = min(self.schedule_increment_floor, 1.0)
                fractions = (floor / num_steps
                             + (1.0 - floor) * fractions)
            cumulative = torch.cumsum(fractions, dim=0)
            return self.anneal_floor + (1.0 - self.anneal_floor) * cumulative[t - 1]
        if self.schedule_type == 'lj13_focus':
            # The v2 stage audit places the largest log-ratio fluctuations at
            # beta=0.24--0.40. Allocate 45% of the stages to beta=0.2--0.5.
            u = t / num_steps
            if u <= 0.30:
                return self.anneal_floor + (0.20 - self.anneal_floor) * (u / 0.30)
            if u <= 0.75:
                return 0.20 + 0.30 * ((u - 0.30) / 0.45)
            return 0.50 + 0.50 * ((u - 0.75) / 0.25)
        if self.schedule_type != 'power':
            raise ValueError(f'unknown annealing schedule: {self.schedule_type}')
        frac = t / num_steps
        return self.anneal_floor + (1.0 - self.anneal_floor) * frac ** self.anneal_power

    def beta_schedule(self, num_steps):
        return torch.stack([torch.as_tensor(self._beta(t, num_steps)) for t in range(1, num_steps + 1)])

    def _drift_raw(self, x, b):
        mu, _ = self.drift_net(x, b)
        if getattr(self, 'drift_pair', False):        # inside the checkpoint: the [B,N,N,F] pair MLP is large
            mu = mu + self.pair_drift(x, b)
        return mu

    def _drift(self, x, beta):
        b = torch.as_tensor(beta, device=x.device, dtype=x.dtype).expand(x.shape[0])
        if self.training and self.checkpoint_equiv and torch.is_grad_enabled():
            mu = checkpoint(self._drift_raw, x, b, use_reentrant=False)
        else:
            mu = self._drift_raw(x, b)
        dc = getattr(self, 'drift_clamp', MU_CLAMP)
        if dc != MU_CLAMP or getattr(self, 'drift_pair', False):
            mu = mu.clamp(-dc, dc)
        return mu

    def force_calls_per_path(self, num_steps):
        """True target-force evaluations per forward path (integrator + momentum-law inputs)."""
        per_stage = (self.n_leapfrog + 1) * (getattr(self, 'drift', 'exact') != 'learned')
        per_stage += 2 * int(self.architecture == 'force_equivariant' and getattr(self, 'law_force_input', True))
        return num_steps * per_stage

    def _force(self, x, beta, base_center=None, base_scale_context=None):
        mode = getattr(self, 'drift', 'exact')
        if mode == 'learned':
            f = self._drift(x, beta)
        else:
            soft = None
            if getattr(self, 'softcore_alpha', 0.0) > 0:
                b = torch.as_tensor(beta, device=x.device, dtype=x.dtype).expand(x.shape[0])
                soft = self.softcore_alpha * (1.0 - b).clamp_min(0).pow(self.softcore_power)
                if getattr(self, 'softcore_learn', 0) > 0:
                    k = self.softcore_knots
                    u = b.clamp(0, 1) * (k.numel() - 1)
                    i0 = u.floor().long().clamp(max=k.numel() - 2) if k.numel() > 1 else torch.zeros_like(u).long()
                    w = (u - i0.to(u.dtype)) if k.numel() > 1 else torch.zeros_like(u)
                    h = k[i0] * (1 - w) + k[(i0 + 1).clamp(max=k.numel() - 1)] * w
                    soft = soft * torch.exp(h.clamp(-5.0, 3.0))
            f = beta * self.target.force(x, soft)
            if mode == 'exact+learned':
                f = f + self._drift(x, beta)
        if self.anneal_base:
            s = self.base_scale if base_scale_context is None else base_scale_context
            s2 = (s * s) if torch.is_tensor(s) else s ** 2
            if torch.is_tensor(s2) and s2.ndim == 2:
                s2 = s2.expand(-1, x.shape[1])
            center = 0.0 if base_center is None else base_center
            f = f + (1.0 - beta) * (-(x - center) / s2)
        return f

    def _leapfrog(self, x, p, eps, beta, base_center=None, base_scale_context=None):
        # Scalar mass m: x-updates use eps * p / m (H = S + 0.5 |p|^2 / m).  With
        # the default m = 1 the division by 1.0 is IEEE-exact, so the legacy path
        # is reproduced bitwise; leapfrog remains a unit-Jacobian bijection for
        # any fixed m, so the path work stays exact (UNDERDAMPED_NHMC_NOTES.md #3).
        m = torch.exp(self.log_mass).to(dtype=x.dtype, device=x.device)
        x = remove_mean(x)
        p = p + 0.5 * eps * self._force(x, beta, base_center, base_scale_context)
        for _ in range(self.n_leapfrog - 1):
            x = remove_mean(x + eps * p / m)
            p = p + eps * self._force(x, beta, base_center, base_scale_context)
        x = remove_mean(x + eps * p / m)
        p = p + 0.5 * eps * self._force(x, beta, base_center, base_scale_context)
        return x, p

    def _momentum_params(self, x, t_norm, reverse=False, base_component=None):
        # Ablation: return a fixed standard-normal law (mu=0, sigma=1) when the
        # corresponding direction is not learned.  Bypasses the learned net so
        # its (untrained/unused) parameters never affect the momentum density.
        mode = getattr(self, 'momentum_mode', 'full')
        if mode == 'fixed' or (mode == 'forward_only' and reverse):
            zeros = torch.zeros_like(x)
            return zeros, zeros
        if self.architecture in ('equivariant', 'force_equivariant'):
            law = self.bwd_law if reverse else self.fwd_law
            if self.training and self.checkpoint_equiv and torch.is_grad_enabled():
                return checkpoint(law, x, t_norm, use_reentrant=False)
            return law(x, t_norm)
        mu_net = self.bwd_mu if reverse else self.fwd_mu
        logsig_net = self.bwd_logsig if reverse else self.fwd_logsig
        if self.condition_on_component:
            if base_component is None:
                base_component = torch.zeros_like(t_norm)
            mu = mu_net(x, t_norm, base_component).clamp(-MU_CLAMP, MU_CLAMP)
            log_sigma = logsig_net(x, t_norm, base_component).clamp(-5.0, 2.0)
        else:
            mu = mu_net(x, t_norm).clamp(-MU_CLAMP, MU_CLAMP)
            log_sigma = logsig_net(x, t_norm).clamp(-5.0, 2.0)
        return mu, log_sigma

    def forward_step(self, x, t, num_steps, base_center=None,
                     base_scale_context=None, base_component=None,
                     momentum_noise=None):
        B = x.size(0)
        t_norm = torch.full((B,), t / num_steps, device=x.device, dtype=x.dtype)
        beta = self._beta(t, num_steps)
        mu_f, log_sig_f = self._momentum_params(
            x, t_norm, reverse=False, base_component=base_component
        )
        sig_f = torch.exp(log_sig_f)
        dist_f = Normal(mu_f, sig_f)
        if momentum_noise is None:
            p0 = dist_f.rsample()
        else:
            momentum_noise = momentum_noise.to(device=x.device, dtype=x.dtype)
            if momentum_noise.shape != x.shape:
                raise ValueError(
                    f"momentum noise has shape {tuple(momentum_noise.shape)}, "
                    f"expected {tuple(x.shape)}"
                )
            p0 = mu_f + sig_f * momentum_noise
        x_new, p_new = self._leapfrog(
            x, p0, self.epsilon_at(t, num_steps), beta,
            base_center, base_scale_context
        )
        log_fwd = dist_f.log_prob(p0).sum(1)

        t_next = torch.full_like(t_norm, (t + 1) / num_steps)
        mu_b, log_sig_b = self._momentum_params(
            x_new, t_next, reverse=True, base_component=base_component
        )
        sig_b = torch.exp(log_sig_b)
        log_bwd = Normal(mu_b, sig_b).log_prob(-p_new).sum(1)
        return x_new, log_fwd, log_bwd

    @torch.no_grad()
    def reverse_step(self, x, t, num_steps, base_center=None,
                     base_scale_context=None, base_component=None):
        B = x.size(0)
        t_next = torch.full((B,), (t + 1) / num_steps, device=x.device, dtype=x.dtype)
        beta = self._beta(t, num_steps)
        mu_b, log_sig_b = self._momentum_params(
            x, t_next, reverse=True, base_component=base_component
        )
        sig_b = torch.exp(log_sig_b)
        m = Normal(mu_b, sig_b).sample()
        # The reverse proposal inverts forward stage t, so it must use the
        # same stage-specific integrator parameters.
        x_prev, _ = self._leapfrog(
            x, m, self.epsilon_at(t, num_steps), beta,
            base_center, base_scale_context
        )
        return x_prev


def base_log_prob_scaled(x0, scale):
    """Normalized isotropic Gaussian density on the zero-COM subspace.

    For ``N`` particles, ``remove_mean`` maps ``3N`` Cartesian coordinates
    onto a ``3(N-1)``-dimensional orthonormal subspace.  The density is
    expressed with respect to Lebesgue measure on that subspace.
    """
    x0 = remove_mean(x0)
    com_free_dim = (_infer_n_particles(x0) - 1) * N_SPACE
    scale_t = torch.as_tensor(scale, device=x0.device, dtype=x0.dtype)
    quadratic = x0.square().sum(dim=1) / scale_t.square()
    log_normalizer = com_free_dim * (
        torch.log(scale_t) + 0.5 * math.log(2.0 * math.pi)
    )
    return -0.5 * quadratic - log_normalizer


def _rollout_persistent(kernel, target, num_steps, x0, logP,
                        base_center, base_scale_context, base_component,
                        momentum_noise=None, base_noise=None,
                        weight_dtype=None, record=False,
                        record_diagnostics=False, return_trace_noise=False,
                        return_stage_increments=False):
    """Underdamped NHMC rollout with persistent (Crank-Nicolson) momentum.

    Implements UNDERDAMPED_NHMC_NOTES.md sec.2.  The momentum p' produced by
    stage t's leapfrog is carried into stage t+1 instead of being discarded;
    the forward refresh is  p~_t = mu_f + a (p'_{t-1}-mu_f) + sqrt(1-a^2) sig_f e
    (stage 1 is a full refresh, a=0 reproduces the legacy densities exactly).

    The per-stage paired increment (f_{t-1}-b_{t-1}) is accumulated ONE STAGE
    LATE, in the same order and as the same (lf-lb) subtraction the legacy loop
    uses, so with alpha=0 the returned W is bitwise-identical to `rollout`'s.
    """
    a = float(kernel.momentum_refresh_alpha)
    sqrt1ma2 = math.sqrt(max(0.0, 1.0 - a * a))
    dim = int(kernel.dim)
    n_particles = int(target.n_particles)
    dev = x0.device
    dtype = x0.dtype

    def _mnoise(step):
        # step is 1-based; momentum_noise is [num_steps, batch, dim] if given.
        if momentum_noise is not None:
            return momentum_noise[step - 1].to(device=dev, dtype=dtype)
        return torch.randn(x0.shape, device=dev, dtype=dtype)

    x = x0
    traj = [x0.detach().cpu()] if record else None
    stage_increments = [] if return_stage_increments else None
    stage_log_ratios = [] if record_diagnostics else None
    stage_actions = [] if record_diagnostics else None
    stage_min_dist = [] if record_diagnostics else None
    stage_core_frac = [] if record_diagnostics else None

    p_prev = None          # p'_{t-1}, the momentum leaving the previous stage
    f_prev = None          # forward log-density f_{t-1}, paired one stage late
    # NOTE: paired increments are produced one stage late, so the k-th entry of
    # stage_log_ratio / stage_increments is the increment of stage k (same as
    # legacy), it is just COMPUTED during stage k+1.

    for step in range(1, num_steps + 1):
        t_norm = torch.full((x.size(0),), step / num_steps, device=dev, dtype=dtype)
        beta = kernel._beta(step, num_steps)
        eps = kernel.epsilon_at(step, num_steps)

        # forward momentum law at (x_{t-1}, t/T)
        mu_f, log_sig_f = kernel._momentum_params(
            x, t_norm, reverse=False, base_component=base_component
        )
        sig_f = torch.exp(log_sig_f)
        noise = _mnoise(step)
        if step == 1 or p_prev is None:
            mean_f = mu_f
            sig_eff_f = sig_f
        else:
            mean_f = (1.0 - a) * mu_f + a * p_prev
            sig_eff_f = sqrt1ma2 * sig_f
        ptilde = mean_f + sig_eff_f * noise
        f_t = Normal(mean_f, sig_eff_f).log_prob(ptilde).sum(1)

        # backward law of the PREVIOUS stage, computable now that p~_t exists.
        # b_{t-1} lives at (x_{t-1}, t/T); mu_b evaluated at the current x (which
        # is x_{t-1}) BEFORE this stage's leapfrog overwrites it.
        if step >= 2:
            mu_b, log_sig_b = kernel._momentum_params(
                x, t_norm, reverse=True, base_component=base_component
            )
            sig_b = torch.exp(log_sig_b)
            mean_b = (1.0 - a) * mu_b + a * (-ptilde)
            sig_eff_b = sqrt1ma2 * sig_b
            b_prev = Normal(mean_b, sig_eff_b).log_prob(-p_prev).sum(1)
            increment = f_prev - b_prev
            if weight_dtype is not None:
                increment = increment.to(weight_dtype)
            logP = logP + increment
            if return_stage_increments:
                stage_increments.append(increment)
            if record_diagnostics:
                stage_log_ratios.append(increment.detach().cpu())

        # leapfrog stage t: (x_{t-1}, p~_t) -> (x_t, p'_t); KEEP p'_t.
        x_new, p_new = kernel._leapfrog(
            x, ptilde, eps, beta, base_center, base_scale_context
        )

        if record_diagnostics:
            with torch.no_grad():
                positions = x_new.detach().view(-1, n_particles, N_SPACE)
                distances = torch.cdist(positions, positions)
                eye = torch.eye(n_particles, device=x.device, dtype=torch.bool)
                distances = distances.masked_fill(eye[None], float('inf'))
                pair_mask = torch.triu(
                    torch.ones(n_particles, n_particles, device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
                stage_actions.append((-target.log_prob(x_new.detach())).cpu())
                stage_min_dist.append(distances.amin(dim=(1, 2)).cpu())
                stage_core_frac.append(
                    (distances[:, pair_mask] < math.sqrt(R2_FLOOR) * target.rm)
                    .to(torch.float32).mean(dim=1).cpu()
                )

        x = x_new
        p_prev = p_new
        f_prev = f_t
        if record:
            traj.append(x.detach().cpu())

    # final backward density b_T (full refresh, unscaled) at (x_T, (T+1)/T).
    t_last = torch.full((x.size(0),), (num_steps + 1) / num_steps, device=dev, dtype=dtype)
    mu_b, log_sig_b = kernel._momentum_params(
        x, t_last, reverse=True, base_component=base_component
    )
    sig_b = torch.exp(log_sig_b)
    b_last = Normal(mu_b, sig_b).log_prob(-p_prev).sum(1)
    increment = f_prev - b_last
    if weight_dtype is not None:
        increment = increment.to(weight_dtype)
    logP = logP + increment
    if return_stage_increments:
        stage_increments.append(increment)
    if record_diagnostics:
        stage_log_ratios.append(increment.detach().cpu())

    endpoint_x = x.to(weight_dtype) if weight_dtype is not None else x
    S = -target.log_prob(endpoint_x)
    W = logP + S

    diagnostics = None
    if record_diagnostics:
        diagnostics = {
            "stage_log_ratio": torch.stack(stage_log_ratios, dim=0),
            "stage_action": torch.stack(stage_actions, dim=0),
            "stage_min_pair_distance": torch.stack(stage_min_dist, dim=0),
            "stage_core_pair_fraction": torch.stack(stage_core_frac, dim=0),
            "beta": kernel.beta_schedule(num_steps).detach().cpu(),
            "epsilon": kernel.epsilon.detach().cpu(),
        }
    outputs = [x, W]
    if record:
        outputs.append(torch.stack(traj, 0))
    if record_diagnostics:
        outputs.append(diagnostics)
    if return_trace_noise:
        outputs.append({
            'base_noise': None if base_noise is None else base_noise.detach(),
            'momentum_noise': None if momentum_noise is None else momentum_noise.detach(),
        })
    if return_stage_increments:
        outputs.append(torch.stack(stage_increments, dim=0))
    return tuple(outputs)


def rollout(kernel, target, n_batch, num_steps, dev, base_scale=1.2,
            dtype=torch.float32, record=False, weight_dtype=None,
            record_diagnostics=False, trace_noise=None,
            return_trace_noise=False, return_stage_increments=False):
    if trace_noise is not None and getattr(kernel, 'base_type', 'gaussian') != 'gaussian':
        raise ValueError("path-noise replay currently supports the Gaussian base only")
    if return_trace_noise and getattr(kernel, 'base_type', 'gaussian') != 'gaussian':
        raise ValueError("path-noise capture currently supports the Gaussian base only")

    dim = int(kernel.dim)
    n_particles = int(target.n_particles)
    base_noise = None
    momentum_noise = None
    if trace_noise is not None:
        base_noise = trace_noise['base_noise'].to(device=dev, dtype=dtype)
        momentum_noise = trace_noise['momentum_noise'].to(device=dev, dtype=dtype)
        if base_noise.shape != (n_batch, dim):
            raise ValueError("replayed base noise does not match the requested batch")
        if momentum_noise.shape != (num_steps, n_batch, dim):
            raise ValueError("replayed momentum noise does not match steps/batch/dimension")
    elif return_trace_noise:
        base_noise = torch.randn(n_batch, dim, device=dev, dtype=dtype)
        momentum_noise = torch.randn(num_steps, n_batch, dim, device=dev, dtype=dtype)

    if hasattr(kernel, 'sample_base'):
        x0, logP, base_center, base_scale_context, base_component = kernel.sample_base(
            n_batch, dev, dtype, base_noise=base_noise
        )
    else:
        scale = getattr(kernel, 'base_scale', base_scale)
        x0 = remove_mean(torch.randn(n_batch, dim, device=dev, dtype=dtype) * scale)
        logP = base_log_prob_scaled(x0, scale)
        base_center = None
        base_scale_context = scale
        base_component = torch.zeros(n_batch, device=dev, dtype=dtype)
    if weight_dtype is not None:
        logP = logP.to(weight_dtype)
    x = x0

    # ------------------------------------------------------------------ #
    # Underdamped / persistent-momentum branch (direction 1).            #
    # alpha=None -> fall through to the legacy full-refresh loop below,   #
    # which is left byte-for-byte untouched.  See UNDERDAMPED_NHMC_NOTES. #
    # ------------------------------------------------------------------ #
    if getattr(kernel, 'momentum_refresh_alpha', None) is not None:
        return _rollout_persistent(
            kernel, target, num_steps, x0, logP,
            base_center, base_scale_context, base_component,
            momentum_noise=momentum_noise, base_noise=base_noise,
            weight_dtype=weight_dtype, record=record,
            record_diagnostics=record_diagnostics,
            return_trace_noise=return_trace_noise,
            return_stage_increments=return_stage_increments,
        )

    traj = [x0.detach().cpu()] if record else None
    stage_increments = [] if return_stage_increments else None
    stage_log_ratios = [] if record_diagnostics else None
    stage_actions = [] if record_diagnostics else None
    stage_min_dist = [] if record_diagnostics else None
    stage_core_frac = [] if record_diagnostics else None
    for step in range(1, num_steps + 1):
        x, lf, lb = kernel.forward_step(
            x, step, num_steps,
            base_center=base_center,
            base_scale_context=base_scale_context,
            base_component=base_component,
            momentum_noise=(None if momentum_noise is None else momentum_noise[step - 1]),
        )
        increment = lf - lb
        if weight_dtype is not None:
            increment = increment.to(weight_dtype)
        logP = logP + increment
        if return_stage_increments:
            stage_increments.append(increment)
        if record:
            traj.append(x.detach().cpu())
        if record_diagnostics:
            with torch.no_grad():
                positions = x.detach().view(-1, n_particles, N_SPACE)
                distances = torch.cdist(positions, positions)
                eye = torch.eye(n_particles, device=x.device, dtype=torch.bool)
                distances = distances.masked_fill(eye[None], float('inf'))
                pair_mask = torch.triu(
                    torch.ones(
                        n_particles, n_particles, device=x.device, dtype=torch.bool
                    ),
                    diagonal=1,
                )
                stage_log_ratios.append(increment.detach().cpu())
                stage_actions.append((-target.log_prob(x.detach())).cpu())
                stage_min_dist.append(distances.amin(dim=(1, 2)).cpu())
                stage_core_frac.append(
                    (distances[:, pair_mask] < math.sqrt(R2_FLOOR) * target.rm)
                    .to(torch.float32).mean(dim=1).cpu()
                )
    endpoint_x = x.to(weight_dtype) if weight_dtype is not None else x
    S = -target.log_prob(endpoint_x)
    W = logP + S
    diagnostics = None
    if record_diagnostics:
        diagnostics = {
            "stage_log_ratio": torch.stack(stage_log_ratios, dim=0),
            "stage_action": torch.stack(stage_actions, dim=0),
            "stage_min_pair_distance": torch.stack(stage_min_dist, dim=0),
            "stage_core_pair_fraction": torch.stack(stage_core_frac, dim=0),
            "beta": kernel.beta_schedule(num_steps).detach().cpu(),
            "epsilon": kernel.epsilon.detach().cpu(),
        }
    outputs = [x, W]
    if record:
        outputs.append(torch.stack(traj, 0))
    if record_diagnostics:
        outputs.append(diagnostics)
    if return_trace_noise:
        outputs.append({
            'base_noise': base_noise.detach(),
            'momentum_noise': momentum_noise.detach(),
        })
    if return_stage_increments:
        # [num_steps, batch] tensor of differentiable per-stage log-ratio
        # increments (log q_fwd - log q_bwd); sums to W - S - logq0 over stages.
        outputs.append(torch.stack(stage_increments, dim=0))
    return tuple(outputs)
