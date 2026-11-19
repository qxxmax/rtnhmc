#!/usr/bin/env python3
"""Stationary round-trip flow on the v4 many-well targets: accepted flow F, ideal flow Omega and the
conditional mismatch E[delta(u)] of the flow-comparison theorem (v4, Thm. "Stationary flow comparison").

For x ~ pi (exact draws), a reverse path gives the bridge u ~ rho. From u, K fresh forward paths y_j with
complete-path works W_j are drawn. Self-normalised weights v_j ~ exp(-W_j) (log q0(u) is common) estimate
the target-induced conditional record law h(.|u) against the proposal f(.|u):
  m_k(u)     ~ sum_j v_j 1[y_j in A_k]                 (region mass under h)
  Omega_kl   = E_rho[m_k(u) m_l(u)],   Omega_tot = 1 - E_rho[sum_k m_k(u)^2]
  delta(u)   ~ 1/2 sum_j |v_j - 1/K|                   (TV(h, f) at fixed u)
The accepted flow uses y_1 as the round-trip proposal: F_kl = E[1(x in A_k) a 1(y_1 in A_l)],
a = min(1, exp(W(Gamma_x) - W_1)). Nested SNIS is biased at finite K, so Omega and delta are reported for
several K (nested subsets of the same K_max draws). Regions: the sign of the first double-well coordinate
(two regions) and all 2^n sign modes.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dw_v2'))
sys.path.insert(0, str(HERE))

from dw_chains import forward_rollout_from, reverse_rollout  # noqa: E402
from dw_compare import exact_moments, mode_index  # noqa: E402
from eval_fixed_flip_ablation import set_fixed_flip_probability  # noqa: E402
from neural_nhmc_dw import ManyWellTarget  # noqa: E402
from train_block_dw_nhmc import build_kernel, seed_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--checkpoint', default='kernel_best.pt')
    ap.add_argument('--p_flip', default='0.05')
    ap.add_argument('--n_outer', type=int, default=16384)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--Ks', type=int, nargs='+', default=[8, 32, 128, 512])
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / 'config.json').read_text())
    ns = argparse.Namespace(**cfg)
    seed_all(a.seed, dev)
    target = ManyWellTarget(ns.dim)
    _, kernel = build_kernel(target, ns)
    kernel.load_state_dict(torch.load(rd / a.checkpoint, map_location=dev))
    kernel = kernel.to(dev).eval()
    if a.p_flip != 'learned':
        set_fixed_flip_probability(kernel, float(a.p_flip))
    T, n_wells = int(cfg['num_steps']), ns.dim // 2
    n_modes = 2 ** n_wells
    _, p_pos = exact_moments(n_wells)
    Kmax = max(a.Ks)
    acc_F2, acc_Ftot, acc_acc = [], [], []
    per_K = {K: dict(omega2=[], omega_tot=[], delta=[], m_pos=[]) for K in a.Ks}
    torch.manual_seed(a.seed)
    with torch.no_grad():
        for b0 in range(0, a.n_outer, a.batch):
            B = min(a.batch, a.n_outer - b0)
            x = target.sample(B, seed=a.seed + b0, device=dev, dtype=torch.float32)
            u, Wx = reverse_rollout(kernel, target, x, T)
            Ws, pos, mode = [], [], []
            for j in range(Kmax):
                y, Wy = forward_rollout_from(kernel, target, u, T)
                Ws.append(Wy.double())
                pos.append((y[:, 0] > 0).double())
                mode.append(mode_index(y))
                if j == 0:
                    al = torch.clamp(Wx.double() - Wy.double(), max=0.0).exp()
                    al = torch.nan_to_num(al, nan=0.0)
                    xp = (x[:, 0] > 0)
                    acc_F2.append((al * (xp != (y[:, 0] > 0)).double()).cpu())
                    acc_Ftot.append((al * (mode_index(x) != mode_index(y)).double()).cpu())
                    acc_acc.append(al.cpu())
            W = torch.stack(Ws, 1)                       # (B, Kmax)
            P = torch.stack(pos, 1)
            Mi = torch.stack(mode, 1)
            for K in a.Ks:
                v = torch.softmax(-W[:, :K], 1)
                v = torch.nan_to_num(v, nan=0.0)
                mp = (v * P[:, :K]).sum(1)
                mk = torch.zeros(B, n_modes, dtype=torch.float64, device=dev)
                mk.scatter_add_(1, Mi[:, :K], v)
                per_K[K]['omega2'].append((mp * (1 - mp)).cpu())
                per_K[K]['omega_tot'].append((1 - (mk ** 2).sum(1)).cpu())
                per_K[K]['delta'].append((0.5 * (v - 1.0 / K).abs().sum(1)).cpu())
                per_K[K]['m_pos'].append(mp.cpu())
            print(f'outer {b0 + B}/{a.n_outer}', flush=True)
    cat = lambda L_: torch.cat(L_).numpy()
    F2, Ft, A = cat(acc_F2), cat(acc_Ftot), cat(acc_acc)
    se = lambda z: float(z.std() / np.sqrt(z.size))
    # F_{+-} for the two-region partition: half of the symmetric two-way accepted flow
    res = dict(args=vars(a), n_outer=a.n_outer, w_pos=p_pos, w_pos_w_neg=p_pos * (1 - p_pos),
               acceptance=float(A.mean()), acceptance_se=se(A),
               F_two_region=float(F2.mean() / 2), F_two_region_se=se(F2) / 2,
               F_total_mode_change=float(Ft.mean()), F_total_se=se(Ft), per_K={})
    for K in a.Ks:
        o2, ot, de, mp = (cat(per_K[K][k]) for k in ('omega2', 'omega_tot', 'delta', 'm_pos'))
        res['per_K'][K] = dict(Omega_two_region=float(o2.mean()), Omega_two_region_se=se(o2),
                               Var_rho_m_pos=float(mp.var()), Omega_total=float(ot.mean()), Omega_total_se=se(ot),
                               E_delta=float(de.mean()), E_delta_se=se(de))
        r = res['per_K'][K]
        print(f'K={K:4d}  Omega+- {r["Omega_two_region"]:.4f}  Omega_tot {r["Omega_total"]:.4f}  '
              f'E[delta] {r["E_delta"]:.4f}', flush=True)
    print(f'accept {res["acceptance"]:.4f}  F+- {res["F_two_region"]:.4f}  F_tot {res["F_total_mode_change"]:.4f}  '
          f'w+w- {res["w_pos_w_neg"]:.4f}', flush=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
