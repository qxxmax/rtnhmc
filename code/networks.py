"""Conditional networks of the NHMC stage kernel: periodic convolution and the (t, kappa)-conditioned
drift / log-width nets (FiLM-style conditioning on a joint sinusoidal embedding of the stage time and kappa).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CyclicConv2d(nn.Module):
    """Conv2d with circular padding for periodic lattice."""

    def __init__(self, in_ch, out_ch, kernel_size=3, bias=False):
        super().__init__()
        self.pad_size = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, [self.pad_size] * 4, mode='circular')
        return self.conv(x)


# ---------------------------------------------------------------------------
# Drift network: predicts drift field on 2D lattice


class TimeKappaEmbedding(nn.Module):
    """Joint (t, kappa) sinusoidal embedding -> cond vector (half embed_dim)."""

    def __init__(self, embed_dim=128):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.dense = nn.Linear(embed_dim * 2, half)
        scale = math.log(10000) / max(half - 1, 1)
        self.register_buffer(
            "freq", torch.exp(-scale * torch.arange(half, dtype=torch.float32))
        )

    def _sinusoidal(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.freq.device)
        x = x.to(dtype=self.freq.dtype, device=self.freq.device)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        inner = x.unsqueeze(-1) * self.freq.unsqueeze(0)
        return torch.cat([torch.sin(inner), torch.cos(inner)], dim=-1)

    def forward(self, t, kappa=None):
        t_emb = self._sinusoidal(t)
        if kappa is not None:
            kappa_emb = self._sinusoidal(kappa * 3.0)
            return F.silu(self.dense(torch.cat([t_emb, kappa_emb], dim=-1)))
        return F.silu(self.dense(torch.cat([t_emb, torch.zeros_like(t_emb)], dim=-1)))


class CondDriftNet(nn.Module):
    """[B,1,L,L] -> [B,1,L,L] with (t, kappa) conditioning."""

    def __init__(self, embed_dim=128, channels=(32, 32)):
        super().__init__()
        self.tk_emb = TimeKappaEmbedding(embed_dim)
        half = embed_dim // 2

        self.conv1 = CyclicConv2d(1, channels[0])
        self.dense1 = nn.Linear(half, channels[0])
        self.norm1 = nn.GroupNorm(min(4, channels[0]), channels[0], eps=1e-3)

        self.conv2 = CyclicConv2d(channels[0], channels[1])
        self.dense2 = nn.Linear(half, channels[1])
        self.norm2 = nn.GroupNorm(min(4, channels[1]), channels[1], eps=1e-3)

        self.conv_out = CyclicConv2d(channels[1], 1)

    def forward(self, x, t, kappa=None):
        cond = self.tk_emb(t, kappa)
        h = F.silu(
            self.norm1(self.conv1(x) + self.dense1(cond).unsqueeze(-1).unsqueeze(-1))
        )
        h = F.silu(
            self.norm2(self.conv2(h) + self.dense2(cond).unsqueeze(-1).unsqueeze(-1))
        )
        return self.conv_out(h)


class CondSigmaNet(nn.Module):
    """Log-sigma branch; forward returns clamped logits (exp applied outside)."""

    def __init__(self, embed_dim=128, channels=(32, 32)):
        super().__init__()
        self.tk_emb = TimeKappaEmbedding(embed_dim)
        half = embed_dim // 2

        self.conv1 = CyclicConv2d(1, channels[0])
        self.dense1 = nn.Linear(half, channels[0])
        self.norm1 = nn.GroupNorm(min(4, channels[0]), channels[0], eps=1e-3)

        self.conv2 = CyclicConv2d(channels[0], channels[1])
        self.dense2 = nn.Linear(half, channels[1])
        self.norm2 = nn.GroupNorm(min(4, channels[1]), channels[1], eps=1e-3)

        self.conv_out = CyclicConv2d(channels[1], 1)

    def forward(self, x, t, kappa=None):
        cond = self.tk_emb(t, kappa)
        h = F.silu(
            self.norm1(self.conv1(x) + self.dense1(cond).unsqueeze(-1).unsqueeze(-1))
        )
        h = F.silu(
            self.norm2(self.conv2(h) + self.dense2(cond).unsqueeze(-1).unsqueeze(-1))
        )
        return self.conv_out(h).clamp(-4.0, 2.0)
