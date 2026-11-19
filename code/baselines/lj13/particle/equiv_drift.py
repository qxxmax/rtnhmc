"""Alternative equivariant backbones for the learned NHMC drift b_phi(x, beta) (2026-10-14).

Both modules have the drift_net interface of ForceEquivariantMomentumLaw(use_force=False):
    forward(x [B, N*d], t [B]) -> (mu [B, N*d] zero-COM, None)
and never call the target force. The last layer is zero-initialised (initial drift = 0, as for the EGNN-style net).

  PaiNNDrift : PaiNN (Schuett, Unke, Gastegger, ICML 2021, arXiv 2102.03150) re-implemented after
               schnetpack.representation.PaiNN, fully connected, scalar + l=1 vector channels, gated
               equivariant vector readout. Pure torch; O(d)-equivariant in any d (works for 2D DW-4).
  MACEDrift  : MACE (Batatia et al., NeurIPS 2022, arXiv 2206.07697) built from mace-torch 0.3.16 blocks
               (RealAgnosticResidualInteractionBlock + EquivariantProductBasisBlock, e3nn 0.4.4), fully
               connected, l=1 readout via o3.Linear(hidden -> 1x1o). The stage/time embedding is fed as the
               MACE node attributes (which normally hold the species one-hot), so node embedding, skip
               connections and the element-dependent product weights all become time-dependent. 3D only;
               2D inputs are embedded at z=0 (O(3)-equivariance with parity forces the z output to 0).
"""
import math

import torch
import torch.nn as nn

from lj13_nhmc import MU_CLAMP, N_SPACE, R2_FLOOR, remove_mean


def time_embedding(t, n_freq):
    f = torch.arange(1, n_freq + 1, device=t.device, dtype=t.dtype)
    a = 2.0 * math.pi * t[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class PaiNNDrift(nn.Module):
    def __init__(self, n_particles, hidden=64, n_layers=3, n_rbf=16, r_max=8.0, n_time_freq=4):
        super().__init__()
        F = hidden
        self.n, self.F, self.n_layers, self.n_time_freq, self.r_max = int(n_particles), F, n_layers, n_time_freq, r_max
        self.register_buffer('rbf_freq', torch.arange(1, n_rbf + 1, dtype=torch.float32) * math.pi / r_max)
        self.time_mlp = nn.Sequential(nn.Linear(2 * n_time_freq, F), nn.SiLU(), nn.Linear(F, F))
        self.film = nn.ModuleList([nn.Linear(F, 2 * F) for _ in range(n_layers)])
        self.phi = nn.ModuleList([nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 3 * F))
                                  for _ in range(n_layers)])
        self.filt = nn.ModuleList([nn.Linear(n_rbf, 3 * F) for _ in range(n_layers)])
        self.U = nn.ModuleList([nn.Linear(F, F, bias=False) for _ in range(n_layers)])
        self.V = nn.ModuleList([nn.Linear(F, F, bias=False) for _ in range(n_layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * F, F), nn.SiLU(), nn.Linear(F, 3 * F))
                                  for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(F) for _ in range(n_layers)])
        # gated equivariant readout (Schuett et al. 2021, Fig. 3)
        self.out_W1 = nn.Linear(F, F, bias=False)
        self.out_W2 = nn.Linear(F, F, bias=False)
        self.out_gate = nn.Sequential(nn.Linear(2 * F, F), nn.SiLU(), nn.Linear(F, F))
        self.out_vec = nn.Linear(F, 1, bias=False)
        nn.init.zeros_(self.out_vec.weight)

    def _rbf(self, r):
        # sin(n pi r / r_max) / r with cosine cutoff, as in PaiNN
        cut = 0.5 * (torch.cos(math.pi * r / self.r_max) + 1.0) * (r < self.r_max)
        return torch.sin(r[..., None] * self.rbf_freq) / r[..., None] * cut[..., None]

    def forward(self, x, t):
        n, F = self.n, self.F
        pos = remove_mean(x).view(-1, n, N_SPACE)
        B = pos.shape[0]
        rel = pos[:, None, :, :] - pos[:, :, None, :]                 # [B,i,j,d] = x_j - x_i
        r = torch.sqrt(rel.square().sum(-1) + 1e-8)
        rhat = rel / r[..., None]
        mask = (~torch.eye(n, device=x.device, dtype=torch.bool)).to(x.dtype)[None, :, :, None]
        rbf = self._rbf(r)
        temb = self.time_mlp(time_embedding(t, self.n_time_freq))       # [B,F]
        s = temb[:, None, :].expand(B, n, F)
        v = pos.new_zeros(B, n, N_SPACE, F)
        for l in range(self.n_layers):
            g, b = self.film[l](temb)[:, None, :].chunk(2, -1)
            s = s * (1 + g) + b
            m = self.phi[l](s)[:, None, :, :] * self.filt[l](rbf) * mask  # [B,i,j,3F]
            a, bv, c = m.split(F, -1)
            s = s + a.sum(2) / (n - 1)
            v = v + (bv[:, :, :, None, :] * v[:, None, :, :, :]
                     + c[:, :, :, None, :] * rhat[..., None]).sum(2) / (n - 1)
            Uv, Vv = self.U[l](v), self.V[l](v)
            vn = torch.sqrt(Vv.square().sum(2) + 1e-8)
            avv, asv, ass = self.upd[l](torch.cat([s, vn], -1)).split(F, -1)
            v = v + avv[:, :, None, :] * Uv
            s = self.norms[l](s + asv * (Uv * Vv).sum(2) + ass)
        w1 = self.out_W1(v)
        w2n = torch.sqrt(self.out_W2(v).square().sum(2) + 1e-8)
        gate = self.out_gate(torch.cat([s, w2n], -1))
        mu = self.out_vec(gate[:, :, None, :] * w1).squeeze(-1)          # [B,n,d]
        mu = remove_mean(mu.reshape(x.shape)).clamp(-MU_CLAMP, MU_CLAMP)
        return mu, None


class MACEDrift(nn.Module):
    def __init__(self, n_particles, hidden=32, n_layers=2, max_ell=2, correlation=3, r_max=8.0,
                 num_bessel=8, n_time_freq=4, radial_mlp=(64, 64, 64)):
        super().__init__()
        # e3nn 0.4.4 loads its Wigner constants with torch.load; torch >= 2.6 defaults to weights_only=True,
        # which rejects the pickled `slice` objects in that (packaged, trusted) file.
        torch.serialization.add_safe_globals([slice])
        from e3nn import o3
        from mace.modules.blocks import (EquivariantProductBasisBlock, LinearNodeEmbeddingBlock,
                                         RadialEmbeddingBlock, RealAgnosticResidualInteractionBlock)
        self.n, self.n_time_freq = int(n_particles), n_time_freq
        na = 1 + 2 * n_time_freq
        attr_irreps = o3.Irreps(f'{na}x0e')
        hidden_irreps = o3.Irreps(f'{hidden}x0e+{hidden}x1o')
        feats0 = o3.Irreps(f'{hidden}x0e')
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=attr_irreps, irreps_out=feats0)
        self.radial = RadialEmbeddingBlock(r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=5)
        edge_feats_irreps = o3.Irreps(f'{self.radial.out_dim}x0e')
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.sh = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization='component')
        inter_irreps = (sh_irreps * hidden).sort()[0].simplify()
        self.interactions, self.products = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            self.interactions.append(RealAgnosticResidualInteractionBlock(
                node_attrs_irreps=attr_irreps, node_feats_irreps=feats0 if i == 0 else hidden_irreps,
                edge_attrs_irreps=sh_irreps, edge_feats_irreps=edge_feats_irreps, target_irreps=inter_irreps,
                hidden_irreps=hidden_irreps, avg_num_neighbors=float(self.n - 1), radial_MLP=list(radial_mlp)))
            self.products.append(EquivariantProductBasisBlock(
                node_feats_irreps=inter_irreps, target_irreps=hidden_irreps, correlation=correlation,
                use_sc=True, num_elements=na))
        self.readout = o3.Linear(hidden_irreps, o3.Irreps('1x1o'))
        with torch.no_grad():
            self.readout.weight.zero_()
        src, dst = torch.where(~torch.eye(self.n, dtype=torch.bool))
        self.register_buffer('edge_template', torch.stack([src, dst]))

    def forward(self, x, t):
        n = self.n
        pos = remove_mean(x).view(-1, n, N_SPACE)
        B = pos.shape[0]
        if N_SPACE < 3:
            pos = torch.cat([pos, pos.new_zeros(B, n, 3 - N_SPACE)], -1)
        pos = pos.reshape(B * n, 3)
        off = (torch.arange(B, device=x.device) * n).repeat_interleave(self.edge_template.shape[1])
        ei = self.edge_template.repeat(1, B) + off[None]
        sender, receiver = ei[0], ei[1]
        vec = pos[receiver] - pos[sender]
        length = torch.sqrt(vec.square().sum(-1, keepdim=True) + 1e-8)
        attrs = torch.cat([torch.ones_like(t)[:, None], time_embedding(t, self.n_time_freq)], -1)
        attrs = attrs.repeat_interleave(n, 0)
        feats = self.node_embedding(attrs)
        edge_attrs = self.sh(vec)
        edge_feats, cutoff = self.radial(length, attrs, ei, None)
        for i, (inter, prod) in enumerate(zip(self.interactions, self.products)):
            feats, sc = inter(node_attrs=attrs, node_feats=feats, edge_attrs=edge_attrs, edge_feats=edge_feats,
                              edge_index=ei, cutoff=cutoff, first_layer=(i == 0))
            feats = prod(node_feats=feats, sc=sc, node_attrs=attrs)
        mu = self.readout(feats).view(B, n, 3)[..., :N_SPACE]
        mu = remove_mean(mu.reshape(x.shape)).clamp(-MU_CLAMP, MU_CLAMP)
        return mu, None


class PairRadialDrift(nn.Module):
    """b_i = sum_j g(r_ij, beta) (x_i - x_j),  g = a(beta) r^-14 - c(beta) r^-8 + h(rbf(r), beta)  (2026-10-17).

    Pure radial pair term: E(n)- and permutation-equivariant, zero-COM by antisymmetry, O(N^2), never calls
    the target. Unlike the EGNN head it is unbounded at short range, so it can represent a singular
    (LJ-like) repulsive core. a, c >= 0 via 24*softplus (init ~3); h zero-initialised.
    r^2 is floored at R2_FLOOR (same fp32 guard as the target force)."""
    def __init__(self, n_particles, n_rbf=16, r_max=4.0, n_time_freq=4, hidden=64):
        super().__init__()
        self.n, self.n_time_freq = int(n_particles), n_time_freq
        self.unit = 24.0
        self.register_buffer('centers', torch.linspace(0.0, r_max, n_rbf))
        self.width = r_max / n_rbf
        self.coef = nn.Sequential(nn.Linear(2 * n_time_freq, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.coef[-1].weight)
        nn.init.constant_(self.coef[-1].bias, -2.0)   # softplus(-2)=0.13: weak repulsive prior, non-vanishing gradient
        self.h = nn.Sequential(nn.Linear(n_rbf + 2 * n_time_freq, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.h[-1].weight); nn.init.zeros_(self.h[-1].bias)

    def forward(self, x, t):
        B = x.shape[0]
        pos = x.view(B, self.n, N_SPACE)
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        r2 = diff.square().sum(-1)
        eye = torch.eye(self.n, device=x.device, dtype=torch.bool)[None]
        r2s = r2.masked_fill(eye, 1.0).clamp_min(R2_FLOOR)
        inv2 = 1.0 / r2s
        te = time_embedding(t, self.n_time_freq)
        ac = nn.functional.softplus(self.coef(te))                       # [B, 2]
        # fixed unit scale 24 (LJ prefactor in reduced units) keeps the logits O(1): with Adam at lr~5e-4 an
        # unscaled softplus coefficient could not reach O(10-100) within a few hundred steps.
        g = self.unit * (ac[:, 0, None, None] * inv2 ** 7 - ac[:, 1, None, None] * inv2 ** 4)
        r = r2s.sqrt()
        rbf = torch.exp(-((r[..., None] - self.centers) / self.width) ** 2)
        h_in = torch.cat([rbf, te[:, None, None, :].expand(B, self.n, self.n, -1)], -1)
        g = (g + self.h(h_in).squeeze(-1)).masked_fill(eye, 0.0)
        return (g[..., None] * diff).sum(2).reshape(x.shape)
