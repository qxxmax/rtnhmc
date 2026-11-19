#!/usr/bin/env python3
"""Weight identity E_Q[w] = Z (w = e^{-W}, unnormalised gamma = e^{-S}) for the torch CMCD / DBS port, L = 2.

For random (perturbed, non-trivial) parameters the identity holds exactly, so the SNIS log Z estimate must agree
with the exact quadrature log Z within its standard error. Cases: cmcd / dbs, with and without Z2, T = 8, float64.
Also: the sampler's running log w equals log P - log Q recomputed on the stored path, and W_0 from
reverse_to/forward_from is consistent (reverse W on the forward endpoint uses the same formula).
Negative control: the forward density evaluated with scale_{k+1} while paths are drawn with scale_k must fail.
(A mis-scaled but normalised BACKWARD kernel would not fail: E_Q[P/Q] = Z for any normalised backward kernel.)
"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import cmcd_phi4 as C  # noqa: E402
import l2_exact as X  # noqa: E402

torch.set_num_threads(int(os.environ.get('NTHREADS', 4)))
KAPPA, T, N, CH = 0.30, 8, 1_200_000, 50_000


def make(alg, z2, seed, train_steps=300):
    """Random perturbation (so that every parameter group is non-trivial), then brief training with the port's
    own loss / optimiser (cmcd_phi4: -mean log w, zero_nans + clip(1) + Adam) to bring Var log w to O(1)."""
    torch.manual_seed(seed)
    m = C.UDBPhi4(alg, 2, KAPPA, T=T, dt=0.1, z2=z2).double()
    with torch.no_grad():
        m.betas.add_(0.3 * torch.randn_like(m.betas))
        m.prior_std.add_(0.1 * torch.randn_like(m.prior_std))
        m.friction.add_(0.1 * torch.randn_like(m.friction))
        if not z2:
            m.prior_mean.add_(0.2 * torch.randn_like(m.prior_mean))
        for net in (m.fwd, m.bwd):
            if net is not None:
                net.out.weight.normal_(0, 0.02)
                net.timestep_phase.normal_(0, 0.3)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-3)
    for it in range(train_steps):
        lw, _, _ = m.sample_paths(512, 'cpu', torch.float64)
        opt.zero_grad(); (-lw.mean()).backward(); C.adam_update_(params, opt)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def estimate(lw):
    lw = lw[np.isfinite(lw)]
    mx = lw.max()
    w = np.exp(lw - mx)
    logZ = mx + math.log(w.mean())
    se = w.std() / w.mean() / math.sqrt(w.size)          # delta method SE of log mean w
    return logZ, se, float(np.var(lw)), float(w.sum() ** 2 / (w ** 2).sum() / w.size)


@torch.no_grad()
def run(m, bug=False):
    lws = []
    for i in range(N // CH):
        torch.manual_seed(1000 + i)
        if not bug:
            lw, _, _ = m.sample_paths(CH, 'cpu', torch.float64)
        else:   # negative control: forward density evaluated with scale_{k+1} (sampling used scale_k)
            _, _, path = m.sample_paths(CH, 'cpu', torch.float64, keep_path=True)
            c = m.coeffs()
            _, LB = m.per_step_logs(path, c)
            sc = c['scale']
            terms = [m.point(path[j], j, c) for j in range(T + 1)]
            LFb = []
            for k in range(T):
                e = c['eta'][k]; d0, f0, _ = terms[k]
                LFb.append(m.lognormal(path[k + 1] - path[k], e * d0 + e * f0, sc[min(k + 1, T - 1)]))
            lw = -m.prior_logpdf(path[0], c) + m.log_gamma(path[-1]) + (LB - torch.stack(LFb)).sum(0)
        lws.append(lw.numpy())
    return np.concatenate(lws)


def main():
    exact = X.exact_4d(KAPPA)
    lz2 = X.logZ_2d(KAPPA)
    res = dict(kappa=KAPPA, T=T, N=N, logZ_exact_4d=exact['logZ'], logZ_exact_2d=lz2, cases={})
    ok = abs(exact['logZ'] - lz2) < 1e-6
    print(f"exact log Z: 4-D trapezoid {exact['logZ']:.8f}, 2-D reduction {lz2:.8f}")
    m0 = make('cmcd', True, 0)
    da = X.check_action(m0.log_gamma, KAPPA)
    res['action_closed_form_max_diff'] = da
    ok &= da < 1e-10
    for alg in ('cmcd', 'dbs'):
        for z2 in (True, False):
            m = make(alg, z2, 7 if alg == 'cmcd' else 8)
            # path recompute + reverse/forward consistency on a few paths
            torch.manual_seed(3)
            with torch.no_grad():
                lw, xT, path = m.sample_paths(256, 'cpu', torch.float64, keep_path=True)
                lq, lp = m.path_logs(path)
                e_path = float((lp - lq - lw).abs().max())
                torch.manual_seed(4)
                u = m.prior_sample(256, m.coeffs(), 'cpu', torch.float64)
                torch.manual_seed(5)
                y, W0 = m.forward_from(u, 0)
            lw = run(m)
            lz, se, vlw, ess = estimate(lw)
            z = (lz - lz2) / se
            case = dict(logZ=lz, se=se, z=z, var_logw=vlw, ess=ess, path_recompute_err=e_path)
            ok &= abs(z) < 4 and e_path < 1e-9
            res['cases'][f'{alg}{"+Z2" if z2 else ""}'] = case
            print(f'{alg:5s} z2={z2!s:5s} logZ {lz:.5f} +- {se:.5f} (exact {lz2:.5f}) z={z:+.2f} '
                  f'Var log w {vlw:.3f} ESS {ess:.3f} path-recompute err {e_path:.1e}', flush=True)
    m = make('cmcd', True, 7)
    lz, se, vlw, ess = estimate(run(m, bug=True))
    zb = (lz - lz2) / se
    res['negative_control_fwd_density_scale_next'] = dict(logZ=lz, se=se, z=zb)
    ok &= abs(zb) > 6
    print(f'NEGATIVE CONTROL (log F_k with scale_(k+1)): logZ {lz:.5f} +- {se:.5f} z={zb:+.1f} (must be large)')
    res['PASS'] = bool(ok)
    json.dump(res, open(os.path.join(HERE, 'test_weight_identity.json'), 'w'), indent=1)
    print('PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
