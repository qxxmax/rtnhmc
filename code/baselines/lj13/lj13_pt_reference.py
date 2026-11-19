#!/usr/bin/env python3
"""Independent LJ13 reference by parallel tempering (PT) with HMC moves, CPU, float64.

Target: exactly the LJ13 target of lj13_compare.py / lj13_hmc.py, built with particle_targets.make_target from the
NHMC run config (lj13sc_a05T56_s0: official iDEM / AS potential, lj_scale 1, osc_scale 1, temperature 1):
    E(x) = sum_{i != j} [(1/r_ij)^12 - 2 (1/r_ij)^6] + 0.5 |x - xbar|^2,     pi(x) ~ exp(-E(x) / T).
E is taken from target.log_prob (E = -T log_prob); the HMC force is the exact analytic gradient of the same E
(checked against autograd of target.log_prob at start-up). No clamped / soft force is used.

E is translation invariant, so the sampler works on the centre-of-mass-free subspace (as the Klein split, whose
COM is 0): states start COM-free, momenta and gradients are projected onto the subspace, so the leapfrog is a
reversible, volume-preserving map of the subspace and the MH test with E + |p|^2 / 2 is exact there.

PT: C independent ladders, K inverse temperatures beta_k = beta_0 * r^k (beta_0 = 1/T, geometric to --beta-min * beta_0
unless --betas is given). Each sweep: one HMC trajectory per replica (per-rung / per-ladder step size, adapted to
--target-acc during burn-in only, then frozen), then neighbour swaps (even / odd pairs alternating), accepted with
min(1, exp((beta_k - beta_{k+1}) (E_k - E_{k+1}))). Initial states are independent of the Klein data
(Gaussian cloud relaxed by clipped steepest descent).

Output (--out DIR):
  lj13_pt_report.json   betas, swap acceptance per rung pair, HMC acceptance and step size per rung, replica round
                        trips, cold-chain E / Rg / mean pair / min pair: mean, SE from the C ladder means, SE from tau,
                        split-R-hat across ladders, tau per ladder and pooled (tau = 1 + 2 sum rho, Sokal c = 5,
                        revision/chainstats.py), tau(E) per rung; Stein checks; comparison with the Klein test split
  lj13_pt_cold.npz      x [C, T_keep, 39] float64 cold-chain states (post burn-in, thinned), E, rg traces per sweep
  test_split_LJ13-PT.npy  10000 states drawn without replacement (seed --split-seed) from the thinned cold chains,
                        float32, shape (10000, 39), COM-free: same layout as Klein test_split_LJ13-1000.npy
Rg = sqrt(mean_i |x_i - xbar|^2) as in lj13_compare.observables.
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # code/ (chainstats, chain_ckpt)
LJ_CODE = os.environ.get('LJ_CODE', str(HERE / 'particle'))
sys.path.insert(0, LJ_CODE)

import chainstats as cs  # noqa: E402
from particle_targets import make_target  # noqa: E402

N, D = 13, 3
IU = torch.triu_indices(N, N, offset=1)


def com_free(x):
    p = x.view(*x.shape[:-1], N, D)
    return (p - p.mean(-2, keepdim=True)).reshape(x.shape)


class Energy:
    """E = -T log_prob of the repo target (exact), analytic gradient of the same function."""

    def __init__(self, target):
        self.t = target
        self.T = float(target.temperature)
        self.lj, self.osc, self.eps, self.rm = target.lj_scale, target.osc_scale, target.eps, target.rm

    def E(self, x):
        sh = x.shape[:-1]
        return (-self.t.log_prob(x.reshape(-1, N * D)) * self.T).reshape(sh)

    def grad(self, x):
        p = x.view(*x.shape[:-1], N, D)
        diff = p[..., IU[0], :] - p[..., IU[1], :]
        r2 = diff.square().sum(-1)
        inv6 = (self.rm ** 2 / r2) ** 3
        # per unordered pair the energy is 2 * lj_scale * eps * (inv6^2 - 2 inv6) (ordered-pair sum x lj_scale);
        # dE/dr2 = 2 lj eps (-6 inv6^2 + 6 inv6) / r2, dE/dx_i = 2 dE/dr2 * diff
        c = 2.0 * self.lj * self.eps * 12.0 * (inv6 - inv6 * inv6) / r2
        f = c[..., None] * diff
        g = torch.zeros_like(p)
        g.index_add_(-2, IU[0], f)
        g.index_add_(-2, IU[1], -f)
        g = g + self.osc * (p - p.mean(-2, keepdim=True))
        return com_free(g.reshape(x.shape))


def observables(x):
    p = x.view(*x.shape[:-1], N, D)
    p = p - p.mean(-2, keepdim=True)
    r = (p[..., IU[0], :] - p[..., IU[1], :]).norm(dim=-1)
    return dict(rg=p.square().sum(-1).mean(-1).sqrt(), mean_pair=r.mean(-1), min_pair=r.min(-1).values)


def init_state(en, n, gen):
    x = com_free(torch.randn(n, N * D, dtype=torch.float64, generator=gen) * 0.6 * N ** (1 / 3))
    for _ in range(2000):
        x = com_free(x - (0.01 * en.grad(x)).clamp(-0.02, 0.02))
    return x


def hmc(en, x, ex, beta, eps, n_lf, gen):
    """x [M, 39] COM-free, ex = E(x) [M], beta / eps [M]. Returns (x, E, accepted)."""
    p = com_free(torch.randn(x.shape, dtype=x.dtype, generator=gen))
    b, e = beta[:, None], eps[:, None]
    h0 = beta * ex + 0.5 * p.square().sum(-1)
    q = x.clone()
    p = p - 0.5 * e * b * en.grad(q)
    for i in range(n_lf):
        q = q + e * p
        if i < n_lf - 1:
            p = p - e * b * en.grad(q)
    p = p - 0.5 * e * b * en.grad(q)
    eq = en.E(q)
    h1 = beta * eq + 0.5 * p.square().sum(-1)
    la = torch.nan_to_num(h0 - h1, nan=-math.inf)
    a = (torch.rand(la.shape, dtype=la.dtype, generator=gen).log() < la) & torch.isfinite(eq)
    return torch.where(a[:, None], q, x), torch.where(a, eq, ex), a


def check_target(en, refs):
    gen = torch.Generator().manual_seed(7)
    x = init_state(en, 64, gen) + 0.05 * com_free(torch.randn(64, N * D, dtype=torch.float64, generator=gen))
    xs = [x]
    if refs:
        xs.append(torch.from_numpy(np.load(os.path.join(refs, 'test_split_LJ13-1000.npy')).astype(np.float64))[:256])
    out = {}
    for i, z in enumerate(xs):
        z = com_free(z.clone()).requires_grad_(True)
        e = en.E(z)
        ga = torch.autograd.grad(e.sum(), z)[0]
        gn = en.grad(z.detach())
        # closed form of the official potential, unordered pairs x 2 (independent of target.log_prob)
        p = z.detach().view(-1, N, D)
        r = (p[:, IU[0]] - p[:, IU[1]]).norm(dim=-1)
        e_cf = (2.0 * (r ** -12 - 2.0 * r ** -6)).sum(1) + 0.5 * (p - p.mean(1, keepdim=True)).square().sum((1, 2))
        out['random' if i == 0 else 'klein_test'] = dict(
            max_rel_grad_err=float(((ga - gn).norm(dim=1) / ga.norm(dim=1)).max()),
            max_abs_energy_vs_closed_form=float((e.detach() - e_cf).abs().max()))
    ok = all(v['max_rel_grad_err'] < 1e-9 and v['max_abs_energy_vs_closed_form'] < 1e-9 for v in out.values())
    if en.lj != 1.0 or en.osc != 1.0 or en.T != 1.0 or en.eps != 1.0 or en.rm != 1.0:
        ok = False
    print('target check', json.dumps(out), flush=True)
    if not ok:
        raise SystemExit('target check failed: energy / gradient differ from the official LJ13 potential '
                         f'(lj_scale {en.lj}, osc {en.osc}, T {en.T})')
    return out


def stein(en, x):
    """Reference-free checks on the COM-free subspace (dim (N-1) D): E[x.gradE] = (N-1) D T and
    E[|gradE|^2] = T E[Lap E] for pi ~ exp(-E/T). 20 batch means."""
    x = com_free(x.double())
    g = en.grad(x)
    p = x.view(-1, N, D)
    diff = p[:, IU[0]] - p[:, IU[1]]
    r = diff.norm(dim=-1)
    # pair energy phi(r) = 2 lj eps (rm^12 r^-12 - 2 rm^6 r^-6); Lap_x of sum phi(r_ij) = sum 2 (phi'' + (D-1) phi'/r)
    k = 2.0 * en.lj * en.eps
    d1 = k * (-12.0 * en.rm ** 12 * r ** -13 + 12.0 * en.rm ** 6 * r ** -7)
    d2 = k * (156.0 * en.rm ** 12 * r ** -14 - 84.0 * en.rm ** 6 * r ** -8)
    lap = (2.0 * (d2 + (D - 1) * d1 / r)).sum(1) + en.osc * (N - 1) * D
    out = {}
    for nm, v in (('virial', (x * g).sum(-1) - (N - 1) * D * en.T), ('grad2_lap', g.square().sum(-1) - en.T * lap)):
        n = v.numel() // 20 * 20
        m = v[:n].view(20, -1).mean(1)
        mu, se = float(v.mean()), float(m.std() / math.sqrt(20))
        out[nm] = dict(mean=mu, se=se, z=mu / se if se > 0 else float('nan'))
    return out


def summary(tr):
    """tr [C, T] cold-chain trace of one observable (post burn-in)."""
    C = tr.shape[0]
    lm = tr.mean(1)
    tau_pool, win = cs.tau_int(tr.T)
    tau_lad = [cs.tau_int(tr[c][:, None])[0] for c in range(C)]
    n_eff = tr.size / tau_pool
    return dict(mean=float(tr.mean()), var=float(tr.var()), se_ladders=float(lm.std(ddof=1) / math.sqrt(C)),
                se_tau=float(math.sqrt(tr.var() / n_eff)), rhat_split=cs_split_rhat(tr),
                tau=tau_pool, window=win, n_sweeps=int(tr.shape[1]), window_ok=bool(win < tr.shape[1] / 50),
                tau_per_ladder=dict(min=float(np.min(tau_lad)), median=float(np.median(tau_lad)),
                                    max=float(np.max(tau_lad))))


def cs_split_rhat(y):
    T = y.shape[1] // 2
    z = np.concatenate([y[:, :T], y[:, T:2 * T]], 0)
    W, B = z.var(1, ddof=1).mean(), T * z.mean(1).var(ddof=1)
    return float(math.sqrt(((T - 1) / T * W + B / T) / W))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='config.json of the LJ13 NHMC run (target definition)')
    ap.add_argument('--refs', default='', help='dir with test_split_LJ13-1000.npy (Klein) for the checks / comparison')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ladders', type=int, default=128)
    ap.add_argument('--temps', type=int, default=8)
    ap.add_argument('--beta-min', type=float, default=0.3, help='hottest beta / beta_0 (geometric ladder)')
    ap.add_argument('--betas', type=float, nargs='*', default=None, help='explicit beta / beta_0 list, first = 1')
    ap.add_argument('--sweeps', type=int, default=100_000)
    ap.add_argument('--burn', type=int, default=10_000)
    ap.add_argument('--thin', type=int, default=20)
    ap.add_argument('--n-lf', type=int, default=10)
    ap.add_argument('--target-acc', type=float, default=0.75)
    ap.add_argument('--n-split', type=int, default=10_000)
    ap.add_argument('--split-seed', type=int, default=20261103)
    ap.add_argument('--seed', type=int, default=20261103)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(a.config))
    en = Energy(make_target(cfg))
    rep = dict(args=vars(a), target=dict(lj_scale=en.lj, osc_scale=en.osc, temperature=en.T, eps=en.eps, rm=en.rm,
                                         n_particles=cfg.get('n_particles', 13)))
    rep['target_check'] = check_target(en, a.refs)

    rel = np.array(a.betas if a.betas else np.geomspace(1.0, a.beta_min, a.temps), dtype=np.float64)
    assert rel[0] == 1.0 and np.all(np.diff(rel) < 0), 'betas must start at 1 and decrease'
    K, C = len(rel), a.ladders
    betas = torch.tensor(rel / en.T, dtype=torch.float64)
    gen = torch.Generator().manual_seed(a.seed)
    M = K * C                                                   # replica slot m = k * C + c (rung k, ladder c)
    x = init_state(en, M, gen)
    ex = en.E(x)
    bvec = betas.repeat_interleave(C)
    eps = 0.005 / bvec.sqrt()
    rate = torch.full((M,), a.target_acc, dtype=torch.float64)
    label = torch.arange(K).repeat_interleave(C).view(K, C).clone()   # which replica (by start rung) sits where
    rep_dir = torch.zeros(K, C, dtype=torch.long)    # per (replica id, ladder): 0 start, 1 left cold end, 2 then hot end
    trips = torch.zeros(C, dtype=torch.long)
    sw_acc = torch.zeros(K - 1, dtype=torch.float64)
    sw_n = torch.zeros(K - 1, dtype=torch.float64)
    hmc_acc = torch.zeros(K, dtype=torch.float64)
    n_keep = (a.sweeps - a.burn + a.thin - 1) // a.thin
    keep = np.empty((C, n_keep, N * D))
    n_prod = a.sweeps - a.burn
    tr_e = np.empty((C, n_prod))
    tr_o = {k: np.empty((C, n_prod)) for k in ('rg', 'mean_pair', 'min_pair')}
    rung_e_all = np.empty((K, C, min(n_prod, 20000)), dtype=np.float32)   # energy per rung slot (tau per rung)
    t0, ik = time.time(), 0
    for s in range(a.sweeps):
        x, ex, acc = hmc(en, x, ex, bvec, eps, a.n_lf, gen)
        if s < a.burn:
            rate = 0.95 * rate + 0.05 * acc.double()
            eps = (eps * torch.exp(0.05 * (rate - a.target_acc))).clamp(1e-5, 0.5)
        else:
            hmc_acc += acc.double().view(K, C).mean(1)
        xv, ev = x.view(K, C, N * D), ex.view(K, C)
        for k in range(s % 2, K - 1, 2):
            la = (betas[k] - betas[k + 1]) * (ev[k] - ev[k + 1])
            sw = torch.rand(C, dtype=torch.float64, generator=gen).log() < la
            xk, xk1 = xv[k].clone(), xv[k + 1].clone()
            xv[k], xv[k + 1] = torch.where(sw[:, None], xk1, xk), torch.where(sw[:, None], xk, xk1)
            ek, ek1 = ev[k].clone(), ev[k + 1].clone()
            ev[k], ev[k + 1] = torch.where(sw, ek1, ek), torch.where(sw, ek, ek1)
            lk, lk1 = label[k].clone(), label[k + 1].clone()
            label[k], label[k + 1] = torch.where(sw, lk1, lk), torch.where(sw, lk, lk1)
            if s >= a.burn:
                sw_acc[k] += sw.double().sum()
                sw_n[k] += C
        # replica round trips: a replica that visited the cold end and then the hot end and returns to cold
        cols = torch.arange(C)
        cold_id, hot_id = label[0], label[K - 1]
        back = rep_dir[cold_id, cols] == 2
        if s >= a.burn:                                  # count trips completed after burn-in only
            trips += back.long()
        rep_dir[cold_id, cols] = 1                       # visited cold: now heading hot
        hh = rep_dir[hot_id, cols] == 1
        rep_dir[hot_id, cols] = torch.where(hh, torch.full_like(hot_id, 2), rep_dir[hot_id, cols])
        if s >= a.burn:
            j = s - a.burn
            tr_e[:, j] = ev[0].numpy()
            for kk, v in observables(xv[0]).items():
                tr_o[kk][:, j] = v.numpy()
            if j < rung_e_all.shape[2]:
                rung_e_all[:, :, j] = ev.numpy()
            if j % a.thin == 0:
                keep[:, ik] = xv[0].numpy()
                ik += 1
        if s % max(a.sweeps // 20, 1) == 0:
            print(f'[sweep {s}] {time.time() - t0:.0f}s  E_cold {float(ev[0].mean()):.3f}  '
                  f'eps_cold {float(eps.view(K, C)[0].mean()):.4f}  trips {int(trips.sum())}', flush=True)
    rep['seconds'] = time.time() - t0
    rep['force_evals'] = int(a.sweeps * (a.n_lf + 1) * M)
    rep['betas'] = betas.tolist()
    rep['swap_acc'] = (sw_acc / sw_n.clamp_min(1)).tolist()
    rep['swap_acc_min'] = float(min(rep['swap_acc']))
    rep['hmc_acc'] = (hmc_acc / max(n_prod, 1)).tolist()
    rep['eps_per_rung'] = eps.view(K, C).mean(1).tolist()
    rep['round_trips'] = dict(total=int(trips.sum()), per_ladder_mean=float(trips.double().mean()),
                              ladders_without_trip=int((trips == 0).sum()))
    rep['per_rung_stats_sweeps'] = int(rung_e_all.shape[2])   # first post-burn sweeps used below
    rep['tau_E_per_rung'] = [cs.tau_int(rung_e_all[k].T.astype(np.float64))[0] for k in range(K)]
    rep['E_mean_per_rung'] = rung_e_all.astype(np.float64).mean((1, 2)).tolist()
    rep['cold'] = dict(energy=summary(tr_e), **{k: summary(v) for k, v in tr_o.items()})
    xs = torch.from_numpy(keep[:, :ik].reshape(-1, N * D))
    rep['stein_pt'] = stein(en, xs)
    idx = np.random.default_rng(a.split_seed).choice(xs.shape[0], a.n_split, replace=False)
    split = com_free(xs[idx]).numpy().astype(np.float32)
    np.save(os.path.join(a.out, 'test_split_LJ13-PT.npy'), split)
    sp = torch.from_numpy(split.astype(np.float64))
    rep['split'] = dict(n=int(split.shape[0]), shape=list(split.shape), dtype=str(split.dtype),
                        n_source=int(xs.shape[0]), energy_mean=float(en.E(sp).mean()),
                        rg_mean=float(observables(sp)['rg'].mean()))
    if a.refs:
        kt = torch.from_numpy(np.load(os.path.join(a.refs, 'test_split_LJ13-1000.npy')).astype(np.float64))
        ek, rk = en.E(com_free(kt)), observables(kt)['rg']
        rep['klein_test'] = dict(n=int(kt.shape[0]), energy_mean=float(ek.mean()),
                                 energy_se_iid=float(ek.std() / math.sqrt(len(ek))), rg_mean=float(rk.mean()),
                                 rg_se_iid=float(rk.std() / math.sqrt(len(rk))), stein=stein(en, kt))
        for k, kv in (('energy', ek), ('rg', rk)):
            c = rep['cold'][k]
            se = math.hypot(max(c['se_ladders'], c['se_tau']), float(kv.std() / math.sqrt(len(kv))))
            rep['klein_test'][f'z_{k}_klein_minus_pt'] = (float(kv.mean()) - c['mean']) / se
    np.savez_compressed(os.path.join(a.out, 'lj13_pt_cold.npz'), x=keep[:, :ik], energy=tr_e.astype(np.float32),
                        rg=tr_o['rg'].astype(np.float32), betas=betas.numpy())
    json.dump(rep, open(os.path.join(a.out, 'lj13_pt_report.json'), 'w'), indent=1, default=float)
    print(json.dumps(dict(swap_acc=rep['swap_acc'], hmc_acc=rep['hmc_acc'], trips=rep['round_trips'],
                          energy={k: rep['cold']['energy'][k] for k in ('mean', 'se_ladders', 'se_tau', 'rhat_split',
                                                                       'tau', 'window_ok')},
                          rg={k: rep['cold']['rg'][k] for k in ('mean', 'se_ladders', 'se_tau', 'rhat_split', 'tau')},
                          klein=rep.get('klein_test', {}), stein=rep['stein_pt'])), flush=True)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
