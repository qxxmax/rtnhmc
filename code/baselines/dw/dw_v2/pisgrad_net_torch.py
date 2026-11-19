"""PyTorch port of the PIS-Grad network used by UDB / sde_sampler.

The upstream implementation is JAX/Flax and lives in this repository at
``paper_repeat/others_gnn/pisgrad_net.py``. This port keeps its structure and
its initialization constants so the comparison is against their architecture
rather than a reinterpretation of it:

  * Fourier time features with a learnable phase and fixed coefficients
    ``linspace(0.1, 100, num_hid)``;
  * a state branch ``[Dense, GELU] * num_layers -> Dense(dim)`` whose output
    layer starts at ``1e-8`` so the network begins as a no-op;
  * a separate time-only branch producing a per-dimension coefficient for the
    target score, also starting at ``1e-8``;
  * the two clips, ``outer_clip`` on the state branch and ``inner_clip`` on the
    score, which is what keeps a near-collision from dominating the output.

Only the wrapper that adapts it to the NHMC momentum-law signature
``(x, t_norm) -> (B, dim)`` is ours; it evaluates the target force internally.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PISGradNet(nn.Module):
    """Faithful port of ``PISGRADNet``."""

    def __init__(self, dim: int, num_hid: int = 64, num_layers: int = 2,
                 outer_clip: float = 1.0e4, inner_clip: float = 1.0e2,
                 weight_init: float = 1.0e-8, bias_init: float = 0.0,
                 use_target_score: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.num_hid = int(num_hid)
        self.outer_clip = float(outer_clip)
        self.inner_clip = float(inner_clip)
        self.use_target_score = bool(use_target_score)

        self.timestep_phase = nn.Parameter(torch.zeros(1, num_hid))
        self.register_buffer(
            'timestep_coeff', torch.linspace(0.1, 100.0, num_hid)[None]
        )

        self.time_coder_state = nn.Sequential(
            nn.Linear(2 * num_hid, num_hid), nn.GELU(), nn.Linear(num_hid, num_hid),
        )

        grad_layers: list[nn.Module] = [nn.Linear(2 * num_hid, num_hid)]
        for _ in range(num_layers):
            grad_layers += [nn.GELU(), nn.Linear(num_hid, num_hid)]
        grad_layers.append(nn.Linear(num_hid, dim))
        self.time_coder_grad = nn.Sequential(*grad_layers)
        nn.init.constant_(self.time_coder_grad[-1].weight, weight_init)
        nn.init.constant_(self.time_coder_grad[-1].bias, bias_init)

        state_layers: list[nn.Module] = []
        in_features = dim + num_hid
        for _ in range(num_layers):
            state_layers += [nn.Linear(in_features, num_hid), nn.GELU()]
            in_features = num_hid
        state_layers.append(nn.Linear(in_features, dim))
        self.state_time_net = nn.Sequential(*state_layers)
        nn.init.constant_(self.state_time_net[-1].weight, 1.0e-8)
        nn.init.zeros_(self.state_time_net[-1].bias)

    def fourier_features(self, t: torch.Tensor) -> torch.Tensor:
        angle = self.timestep_coeff * t[:, None] + self.timestep_phase
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                target_score: torch.Tensor | None = None) -> torch.Tensor:
        time_embedding = self.fourier_features(t)
        state_time = self.time_coder_state(time_embedding)
        out = self.state_time_net(torch.cat([x, state_time], dim=-1))
        out = out.clamp(-self.outer_clip, self.outer_clip)
        if self.use_target_score and target_score is not None:
            coefficient = self.time_coder_grad(time_embedding)
            score = target_score.clamp(-self.inner_clip, self.inner_clip)
            out = out + coefficient * score
        return out


class PISGradMomentumNet(nn.Module):
    """Adapter to the NHMC momentum-law signature ``(x, t_norm) -> (B, dim)``.

    The target force is evaluated here so the kernel does not have to thread it
    through; ``out_bias`` shifts the output the way the plain ``MomentumNet``
    does, which the scale heads rely on.
    """

    def __init__(self, dim: int, hidden: int = 128, out_bias: float = 0.0,
                 target=None, num_layers: int = 2, use_target_score: bool = True):
        super().__init__()
        if target is None:
            raise ValueError('PISGradMomentumNet needs the target for its score branch')
        self.target = target
        self.out_bias = float(out_bias)
        self.net = PISGradNet(
            dim, num_hid=hidden, num_layers=num_layers,
            use_target_score=use_target_score,
        )

    def forward(self, x: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        score = self.target.force(x) if self.net.use_target_score else None
        return self.net(x, t_norm, score) + self.out_bias
