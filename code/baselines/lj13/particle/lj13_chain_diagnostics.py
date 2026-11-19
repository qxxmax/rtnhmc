#!/usr/bin/env python3
"""LJ-13 inference-chain diagnostics for the trained v7a NHMC kernel.

Two Metropolis chains built on the frozen kernel (no retraining):

  1. path-IMH      -- independent NHMC path proposals + MH acceptance
                      min(1, exp(W_x - W_y)) (same kernel construction as
                      make_lj13_imh.py, but run long enough for tau_int).
                      Proposals are i.i.d., so they are pre-generated in
                      large GPU batches and consumed in order by the chains;
                      this is mathematically identical to sequential IMH.
  2. roundtrip MH  -- current-state kernel: from x, sample an actual reverse
                      NHMC path to a bridge point u, then a forward NHMC path
                      u -> y, and accept y with min(1, exp(W_x - W_y)) where
                      W = log q0(u) + sum_t(log f_t - log r_t) + S(endpoint).
                      Mechanism ported from the phi4 reference implementation
                      evaluation_conditional_20260719/phi4_roundtrip_mh.py.

Observables: energy E(x) = -log gamma(x) and minimum pair distance.
tau_int via the Sokal windowed estimator (c=5.0), matching
imh_tau_int_phi4_fullsave.sokal_tau.  Output format is aligned with the phi4
three-corrections table (conditional_phi4_three_corrections.csv).

Cost accounting: exact counted target.force() calls per chain step
(instrumented, not formula-trusted).  For v7a one NHMC path costs
56 stages x (8+1 leapfrog + 2 momentum-law force features) = 616 force
evals; the dynamics-only convention 56 x 9 = 504 is reported alongside.

GPU safety: waits for >= --min-free-mib free GPU memory before starting and
uses small batches (well under 2 GB).

Outputs (under results/neural_nhmc_dw/lj13_chain_diagnostics/):
  path_imh/trace.npz, path_imh/result.json
  roundtrip/trace.npz, roundtrip/result.json
  lj13_chain_diagnostics.csv   (three_corrections-style table incl. SNIS row)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from lj13_nhmc import (
    LJ13Target, LJ13NHMCKernel, rollout, remove_mean, base_log_prob_scaled,
    N_PART, N_SPACE,
)
from neural_nhmc_dw import ess_logz

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results', 'neural_nhmc_dw')


# ---------------------------------------------------------------------------
# GPU memory guard (other trainings share this GPU)
# ---------------------------------------------------------------------------
def wait_for_gpu(min_free_mib: float, poll_s: float = 30.0):
    if not torch.cuda.is_available():
        return
    while True:
        free_b, _ = torch.cuda.mem_get_info()
        free_mib = free_b / (1024 ** 2)
        if free_mib >= min_free_mib:
            print(f'[gpu-guard] free={free_mib:.0f} MiB >= {min_free_mib:.0f} MiB, proceeding',
                  flush=True)
            return
        print(f'[gpu-guard] free={free_mib:.0f} MiB < {min_free_mib:.0f} MiB, waiting {poll_s}s',
              flush=True)
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# Kernel loading (full v7 config, unlike the older make_lj13_imh loader)
# ---------------------------------------------------------------------------
def load_kernel(run_name: str, dev):
    run_dir = os.path.join(RESULTS, run_name)
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    target = LJ13Target()
    kernel = LJ13NHMCKernel(
        target, n_leapfrog=cfg['n_leapfrog'], hidden=cfg['hidden'],
        anneal_power=cfg.get('anneal_power', 1.0),
        anneal_floor=cfg.get('anneal_floor', 0.05),
        anneal_base=bool(cfg.get('anneal_base', 0)),
        learn_base=bool(cfg.get('learn_base', 0)),
        base_scale=cfg.get('base_scale', 1.2),
        num_steps=cfg.get('num_steps'),
        stagewise_eps=bool(cfg.get('stagewise_eps', 0)),
        learn_schedule=bool(cfg.get('learn_schedule', 0)),
        architecture=cfg.get('architecture', 'mlp'),
        equiv_layers=cfg.get('equiv_layers', 3),
        equiv_rbf=cfg.get('equiv_rbf', 12),
        equiv_time_freq=cfg.get('equiv_time_freq', 4),
        equiv_force_scale=cfg.get('equiv_force_scale', 20.0),
        checkpoint_equiv=bool(cfg.get('checkpoint_equiv', 1)),
        schedule_type=cfg.get('schedule_type', 'power'),
        base_type=cfg.get('base_type', 'gaussian'),
        structured_noise_scale=cfg.get('structured_noise_scale', 0.16),
        structured_template_scale=cfg.get('structured_template_scale', 1.0),
        structured_mix_prob=cfg.get('structured_mix_prob', 0.5),
        condition_on_component=bool(cfg.get('condition_on_component', 0)),
    ).to(dev)
    kernel.load_state_dict(
        torch.load(os.path.join(run_dir, 'kernel_best.pt'), map_location=dev))
    kernel.eval()
    return run_dir, target, kernel, cfg


# ---------------------------------------------------------------------------
# tau_int (Sokal, c=5.0) -- copied from imh_tau_int_phi4_fullsave.py
# ---------------------------------------------------------------------------
def autocorr_fft(values: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    x = x - x.mean()
    var = np.dot(x, x) / n
    lag = min(max_lag, n - 1)
    if var <= 0 or lag <= 0:
        out = np.zeros(lag + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n]
    acf = acf / np.arange(n, 0, -1)
    return acf[:lag + 1] / acf[0]


def sokal_tau(values: np.ndarray, c: float = 5.0) -> tuple[float, float]:
    rho = autocorr_fft(values, min(len(values) // 4, 4096))
    tau = 0.5
    window = 0
    for t in range(1, len(rho)):
        tau += rho[t]
        window = t
        if t > c * max(2.0 * tau, 1.0):
            break
    err = tau * np.sqrt(2.0 * (2 * window + 1) / max(len(values), 1))
    return float(max(tau, 0.5)), float(err)


def per_chain_trace_stats(traces: np.ndarray) -> dict:
    """traces: [n_steps, n_chain] post-burn observable trace."""
    n_steps, n_chain = traces.shape
    taus, dtaus = [], []
    for c in range(n_chain):
        tau, dtau = sokal_tau(traces[:, c])
        taus.append(tau)
        dtaus.append(dtau)
    taus = np.asarray(taus)
    tau_mean = float(taus.mean())
    n_eff = float(n_steps * n_chain / (2.0 * tau_mean))
    flat = traces.reshape(-1)
    stderr = float(np.std(flat, ddof=1) / math.sqrt(max(n_eff, 1.0)))
    return {
        'mean': float(flat.mean()),
        'stderr_autocorr': stderr,
        'tau_int_per_chain': [float(t) for t in taus],
        'dtau_int_per_chain': [float(t) for t in dtaus],
        'tau_int_mean': tau_mean,
        'tau_int_spread': float(taus.std(ddof=1)) if n_chain > 1 else 0.0,
        'n_eff': n_eff,
        'n_samples': int(n_steps * n_chain),
    }


def min_pair_dist(x: torch.Tensor) -> torch.Tensor:
    p = x.view(-1, N_PART, N_SPACE)
    d = torch.cdist(p, p)
    eye = torch.eye(N_PART, device=x.device, dtype=torch.bool)
    return d.masked_fill(eye[None], float('inf')).amin(dim=(1, 2))


# ---------------------------------------------------------------------------
# Force-eval counting (instrumented)
# ---------------------------------------------------------------------------
class ForceCounter:
    def __init__(self, target):
        self.target = target
        self.n = 0
        self._orig = None

    def __enter__(self):
        self._orig = self.target.force

        def counting_force(x):
            self.n += 1
            return self._orig(x)

        self.target.force = counting_force
        return self

    def __exit__(self, *exc):
        self.target.force = self._orig
        return False


# ---------------------------------------------------------------------------
# Round-trip machinery (ported from phi4_roundtrip_mh.py to the LJ13 kernel)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_path(kernel, target, u: torch.Tensor, num_steps: int):
    """u -> y with forward-path log densities. Returns y, log_f, log_r."""
    x = u
    log_f = torch.zeros(x.size(0), device=x.device, dtype=torch.float64)
    log_r = torch.zeros_like(log_f)
    for t in range(1, num_steps + 1):
        x, lf, lb = kernel.forward_step(x, t, num_steps)
        log_f += lf.to(torch.float64)
        log_r += lb.to(torch.float64)
    return x, log_f, log_r


@torch.no_grad()
def reverse_step_with_work(kernel, x_next: torch.Tensor, t: int, num_steps: int,
                           check_inverse: bool = False):
    """Sample the actual reverse law at stage t and reconstruct the
    forward-oriented step densities.

    Forward stage t maps x_{t-1} -> x_t via p0 ~ N(mu_f(x_{t-1}, t/S)),
    leapfrog(eps_t, beta_t).  The reverse move samples m ~ N(mu_b(x_t,(t+1)/S))
    (m = -p_new in forward notation); the leapfrog is an involution under
    momentum flip, so leapfrog(x_t, m) = (x_{t-1}, -p0).
    """
    B = x_next.size(0)
    t_next = torch.full((B,), (t + 1) / num_steps,
                        device=x_next.device, dtype=x_next.dtype)
    t_norm = torch.full((B,), t / num_steps,
                        device=x_next.device, dtype=x_next.dtype)
    beta = kernel._beta(t, num_steps)
    eps = kernel.epsilon_at(t, num_steps)

    mu_b, log_sig_b = kernel._momentum_params(x_next, t_next, reverse=True)
    sig_b = torch.exp(log_sig_b)
    dist_b = torch.distributions.Normal(mu_b, sig_b)
    m = dist_b.sample()
    log_r = dist_b.log_prob(m).sum(1)

    x_prev, minus_p0 = kernel._leapfrog(x_next, m, eps, beta)
    p0 = -minus_p0
    mu_f, log_sig_f = kernel._momentum_params(x_prev, t_norm, reverse=False)
    sig_f = torch.exp(log_sig_f)
    log_f = torch.distributions.Normal(mu_f, sig_f).log_prob(p0).sum(1)

    inverse_error = None
    if check_inverse:
        x_check, p_check = kernel._leapfrog(x_prev, p0, eps, beta)
        inverse_error = float(
            torch.maximum((x_check - x_next).abs().amax(),
                          (p_check + m).abs().amax()))
    return x_prev, log_f, log_r, inverse_error


@torch.no_grad()
def reverse_path(kernel, target, x: torch.Tensor, num_steps: int,
                 check_inverse: bool = False):
    current = x
    log_f = torch.zeros(x.size(0), device=x.device, dtype=torch.float64)
    log_r = torch.zeros_like(log_f)
    max_inverse_error = 0.0
    for t in range(num_steps, 0, -1):
        current, lf, lr, inv = reverse_step_with_work(
            kernel, current, t, num_steps, check_inverse=check_inverse)
        log_f += lf.to(torch.float64)
        log_r += lr.to(torch.float64)
        if inv is not None:
            max_inverse_error = max(max_inverse_error, inv)
    return current, log_f, log_r, max_inverse_error


@torch.no_grad()
def path_work_forward(kernel, target, batch: int, num_steps: int, dev):
    """Independent proposal with decomposed work (float64 accumulation)."""
    x, W = rollout(kernel, target, batch, num_steps, dev,
                   weight_dtype=torch.float64)
    return x, W


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_path_imh(kernel, target, cfg, dev, n_chain, n_steps, gen_batch,
                 min_free_mib, seed):
    """Pre-generate the i.i.d. proposal stream in GPU batches, then run the
    vectorized IMH scan.  Identical law to sequential IMH."""
    torch.manual_seed(seed)
    num_steps = cfg['num_steps']
    n_total = n_chain * (n_steps + 1)          # +1 for the initial states
    W_all = torch.empty(n_total, dtype=torch.float64)
    E_all = torch.empty(n_total, dtype=torch.float64)
    D_all = torch.empty(n_total, dtype=torch.float64)

    t0 = time.time()
    done = 0
    while done < n_total:
        wait_for_gpu(min_free_mib, poll_s=30.0)
        b = min(gen_batch, n_total - done)
        x, W = path_work_forward(kernel, target, b, num_steps, dev)
        W_all[done:done + b] = W.cpu()
        E_all[done:done + b] = (-target.log_prob(x)).to(torch.float64).cpu()
        D_all[done:done + b] = min_pair_dist(x).to(torch.float64).cpu()
        done += b
        if done % (gen_batch * 5) == 0 or done == n_total:
            print(f'[imh] generated {done}/{n_total} proposals '
                  f'({time.time() - t0:.0f}s)', flush=True)

    # deterministic MH scan on CPU
    rng = np.random.default_rng(seed + 1)
    W_np = W_all.numpy().reshape(n_steps + 1, n_chain)
    E_np = E_all.numpy().reshape(n_steps + 1, n_chain)
    D_np = D_all.numpy().reshape(n_steps + 1, n_chain)

    W_cur, E_cur, D_cur = W_np[0].copy(), E_np[0].copy(), D_np[0].copy()
    bad0 = ~np.isfinite(W_cur)
    W_cur[bad0] = np.inf                      # forces first finite proposal in
    acc_tr = np.empty((n_steps, n_chain), dtype=bool)
    E_tr = np.empty((n_steps, n_chain))
    D_tr = np.empty((n_steps, n_chain))
    W_tr = np.empty((n_steps, n_chain))
    for step in range(n_steps):
        W_p, E_p, D_p = W_np[step + 1], E_np[step + 1], D_np[step + 1]
        log_alpha = np.minimum(W_cur - W_p, 0.0)
        acc = (np.log(rng.random(n_chain).clip(1e-300)) <= log_alpha) \
            & np.isfinite(W_p)
        W_cur = np.where(acc, W_p, W_cur)
        E_cur = np.where(acc, E_p, E_cur)
        D_cur = np.where(acc, D_p, D_cur)
        acc_tr[step] = acc
        E_tr[step], D_tr[step], W_tr[step] = E_cur, D_cur, W_cur
    return {
        'accept': acc_tr, 'energy': E_tr, 'min_dist': D_tr, 'W_current': W_tr,
        'W_proposal': W_np[1:], 'wall_sec': time.time() - t0,
    }


@torch.no_grad()
def run_roundtrip(kernel, target, cfg, dev, n_chain, n_steps, min_free_mib,
                  seed, print_every=200):
    torch.manual_seed(seed)
    num_steps = cfg['num_steps']
    wait_for_gpu(min_free_mib, poll_s=30.0)

    # init from independent forward paths (target-side endpoints)
    x, _ = path_work_forward(kernel, target, n_chain, num_steps, dev)

    acc_tr = np.empty((n_steps, n_chain), dtype=bool)
    E_tr = np.empty((n_steps, n_chain))
    D_tr = np.empty((n_steps, n_chain))
    Wx_tr = np.empty((n_steps, n_chain))
    Wy_tr = np.empty((n_steps, n_chain))
    esjd_tr = np.empty((n_steps, n_chain))
    inverse_error0 = None

    t0 = time.time()
    for step in range(n_steps):
        check = (step == 0)
        u, log_fx, log_rx, inv = reverse_path(
            kernel, target, x, num_steps, check_inverse=check)
        if check:
            inverse_error0 = inv
            print(f'[roundtrip] step-0 leapfrog inverse check: '
                  f'max abs error = {inv:.3e}', flush=True)
        y, log_fy, log_ry = forward_path(kernel, target, u, num_steps)

        log_q0_u = base_log_prob_scaled(u, kernel.base_scale).to(torch.float64)
        s_x = (-target.log_prob(x)).to(torch.float64)
        s_y = (-target.log_prob(y)).to(torch.float64)
        work_x = log_q0_u + s_x + log_fx - log_rx
        work_y = log_q0_u + s_y + log_fy - log_ry
        log_alpha = (work_x - work_y).clamp(max=0.0)
        accept = (torch.log(torch.rand_like(log_alpha).clamp_min(1e-30))
                  <= log_alpha) & torch.isfinite(work_y)
        squared_jump = (y - x).square().sum(1)
        x = torch.where(accept[:, None], y, x)

        acc_tr[step] = accept.cpu().numpy()
        E_cur = (-target.log_prob(x)).to(torch.float64)
        E_tr[step] = E_cur.cpu().numpy()
        D_tr[step] = min_pair_dist(x).cpu().numpy()
        Wx_tr[step] = work_x.cpu().numpy()
        Wy_tr[step] = work_y.cpu().numpy()
        esjd_tr[step] = (squared_jump * accept).cpu().numpy()

        if (step + 1) % print_every == 0 or step + 1 == n_steps:
            print(f'[roundtrip] step {step + 1}/{n_steps} '
                  f'acc={acc_tr[:step + 1].mean():.4f} '
                  f'E_mean={E_tr[step].mean():.2f} '
                  f'wall={time.time() - t0:.0f}s', flush=True)
        if (step + 1) % 50 == 0:
            wait_for_gpu(min_free_mib, poll_s=30.0)

    return {
        'accept': acc_tr, 'energy': E_tr, 'min_dist': D_tr,
        'work_x': Wx_tr, 'work_y': Wy_tr, 'esjd': esjd_tr,
        'inverse_error_step0': inverse_error0, 'wall_sec': time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------
@torch.no_grad()
def count_costs(kernel, target, cfg, dev):
    num_steps = cfg['num_steps']
    with ForceCounter(target) as fc:
        path_work_forward(kernel, target, 1, num_steps, dev)
    imh_step = fc.n
    with ForceCounter(target) as fc:
        x, _ = path_work_forward(kernel, target, 1, num_steps, dev)
        fc.n = 0
        reverse_path(kernel, target, x, num_steps)
        forward_path(kernel, target, x, num_steps)
    roundtrip_step = fc.n
    dynamics_only = num_steps * (cfg['n_leapfrog'] + 1)
    return {
        'force_evals_per_imh_step_counted': imh_step,
        'force_evals_per_roundtrip_step_counted': roundtrip_step,
        'force_evals_per_path_dynamics_only': dynamics_only,
        'note': ('counted includes the 2 target.force() calls per stage made '
                 'by the force-equivariant momentum law (features), on top of '
                 'the (n_leapfrog+1) leapfrog force evals per stage'),
    }


# ---------------------------------------------------------------------------
def summarize(name, res, burn, costs, cost_per_step, snis_ref):
    post = slice(burn, None)
    acc = res['accept'][post]
    stats = {
        'energy': per_chain_trace_stats(res['energy'][post]),
        'min_dist': per_chain_trace_stats(res['min_dist'][post]),
    }
    n_steps, n_chain = res['accept'].shape
    total_force = n_steps * n_chain * cost_per_step
    n_eff_energy = stats['energy']['n_eff']
    out = {
        'method': name,
        'n_chains': int(n_chain),
        'n_steps': int(n_steps),
        'burn_in': int(burn),
        'n_post_burn': int(acc.shape[0] * n_chain),
        'acceptance_rate': float(acc.mean()),
        'trace_statistics': stats,
        'force_evals_per_chain_step': int(cost_per_step),
        'total_force_evals': int(total_force),
        'n_eff_energy_per_1e6_force_evals':
            float(n_eff_energy * 1.0e6 / total_force),
        'n_eff_mindist_per_1e6_force_evals':
            float(stats['min_dist']['n_eff'] * 1.0e6 / total_force),
        'snis_reference_eff_samples_per_1e6_force_evals': snis_ref,
    }
    if 'esjd' in res:
        out['mean_esjd'] = float(res['esjd'][post].mean())
        out['mean_delta_work'] = float(
            (res['work_y'][post] - res['work_x'][post]).mean())
        out['std_delta_work'] = float(
            (res['work_y'][post] - res['work_x'][post]).std(ddof=1))
        out['inverse_error_step0'] = res['inverse_error_step0']
    out['wall_sec'] = res['wall_sec']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='lj13_v7a_force_equiv')
    ap.add_argument('--n-steps', type=int, default=10000)
    ap.add_argument('--burn-in', type=int, default=1000)
    ap.add_argument('--n-chains', type=int, default=4)
    ap.add_argument('--gen-batch', type=int, default=500)
    ap.add_argument('--min-free-mib', type=float, default=2000.0)
    ap.add_argument('--seed', type=int, default=20260720)
    ap.add_argument('--only', choices=('imh', 'roundtrip', 'both'),
                    default='both')
    ap.add_argument('--out-name', default='lj13_chain_diagnostics')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile the momentum laws (~4.5x faster '
                         'roundtrip stepping; numerics unchanged)')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    wait_for_gpu(args.min_free_mib, poll_s=30.0)
    run_dir, target, kernel, cfg = load_kernel(args.run, dev)
    out_root = os.path.join(RESULTS, args.out_name)
    os.makedirs(out_root, exist_ok=True)

    # Count costs BEFORE compiling: the counter monkey-patches target.force,
    # which compiled graphs would not see.
    costs = count_costs(kernel, target, cfg, dev)
    print(json.dumps(costs, indent=2), flush=True)

    if args.compile and dev.type == 'cuda' \
            and cfg.get('architecture') in ('equivariant', 'force_equivariant'):
        kernel.fwd_law = torch.compile(kernel.fwd_law, dynamic=False)
        kernel.bwd_law = torch.compile(kernel.bwd_law, dynamic=False)
        print('[compile] momentum laws compiled', flush=True)

    # SNIS reference from the frozen unified metrics (ESS per 1e6 force evals)
    snis_ref = None
    metrics_path = os.path.join(run_dir, 'lj13_unified_metrics.json')
    if os.path.exists(metrics_path):
        um = json.load(open(metrics_path))
        snis_ref = um['ess']['ess_per_1e6_force_evals']

    rows = []
    common = {
        'run': args.run, 'config': cfg, 'costs': costs,
        'seed': args.seed, 'device': str(dev),
        'tau_estimator': 'sokal windowed, c=5.0 '
                         '(imh_tau_int_phi4_fullsave.sokal_tau)',
    }

    if args.only in ('imh', 'both'):
        print('\n=== path-IMH ===', flush=True)
        res = run_path_imh(kernel, target, cfg, dev, args.n_chains,
                           args.n_steps, args.gen_batch, args.min_free_mib,
                           args.seed)
        odir = os.path.join(out_root, 'path_imh')
        os.makedirs(odir, exist_ok=True)
        np.savez_compressed(os.path.join(odir, 'trace.npz'),
                            accept=res['accept'], energy=res['energy'],
                            min_dist=res['min_dist'],
                            W_current=res['W_current'],
                            W_proposal=res['W_proposal'])
        summary = summarize('path-IMH', res, args.burn_in, costs,
                            costs['force_evals_per_imh_step_counted'],
                            snis_ref)
        result = {**common, 'n_chains': args.n_chains,
                  'n_steps': args.n_steps, 'burn_in': args.burn_in,
                  'observables': summary}
        json.dump(result, open(os.path.join(odir, 'result.json'), 'w'),
                  indent=2)
        rows.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    if args.only in ('roundtrip', 'both'):
        print('\n=== roundtrip MH ===', flush=True)
        res = run_roundtrip(kernel, target, cfg, dev, args.n_chains,
                            args.n_steps, args.min_free_mib, args.seed + 7)
        odir = os.path.join(out_root, 'roundtrip')
        os.makedirs(odir, exist_ok=True)
        np.savez_compressed(os.path.join(odir, 'trace.npz'),
                            accept=res['accept'], energy=res['energy'],
                            min_dist=res['min_dist'], work_x=res['work_x'],
                            work_y=res['work_y'], esjd=res['esjd'])
        summary = summarize('round-trip NHMC-MH', res, args.burn_in, costs,
                            costs['force_evals_per_roundtrip_step_counted'],
                            snis_ref)
        result = {**common, 'n_chains': args.n_chains,
                  'n_steps': args.n_steps, 'burn_in': args.burn_in,
                  'observables': summary}
        json.dump(result, open(os.path.join(odir, 'result.json'), 'w'),
                  indent=2)
        rows.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    # -------- three_corrections-style CSV (incl. SNIS reference row) --------
    csv_rows = []
    for s in rows:
        st = s['trace_statistics']
        csv_rows.append({
            'method': s['method'],
            'target': 'LJ13',
            'n_generated': s['n_steps'] * s['n_chains'],
            'n_valid_or_postburn': s['n_post_burn'],
            'acceptance': s['acceptance_rate'],
            'n_eff': st['energy']['n_eff'],
            'ess_frac': '',
            'tau_int_energy': st['energy']['tau_int_mean'],
            'tau_int_energy_spread': st['energy']['tau_int_spread'],
            'tau_int_mindist': st['min_dist']['tau_int_mean'],
            'energy_mean': st['energy']['mean'],
            'energy_err': st['energy']['stderr_autocorr'],
            'mindist_mean': st['min_dist']['mean'],
            'mindist_err': st['min_dist']['stderr_autocorr'],
            'force_evals_per_step': s['force_evals_per_chain_step'],
            'total_force_evals': s['total_force_evals'],
            'n_eff_per_1e6_force_evals':
                s['n_eff_energy_per_1e6_force_evals'],
            'source': os.path.join(out_root, s['method']),
        })
    if snis_ref is not None:
        um = json.load(open(metrics_path))
        csv_rows.append({
            'method': 'path-SNIS (reference)',
            'target': 'LJ13',
            'n_generated': um['force_eval_accounting']['proposal_pool_size'],
            'n_valid_or_postburn':
                um['force_eval_accounting']['proposal_pool_size'],
            'acceptance': '',
            'n_eff': um['ess']['normalized_ess_frac_full_pool']
                * um['force_eval_accounting']['proposal_pool_size'],
            'ess_frac': um['ess']['normalized_ess_frac_full_pool'],
            'tau_int_energy': '', 'tau_int_energy_spread': '',
            'tau_int_mindist': '',
            'energy_mean': '', 'energy_err': '',
            'mindist_mean': '', 'mindist_err': '',
            'force_evals_per_step':
                um['force_eval_accounting']['per_sample_counted'],
            'total_force_evals':
                um['force_eval_accounting']['proposal_pool_total_force_evals'],
            'n_eff_per_1e6_force_evals': snis_ref,
            'source': metrics_path,
        })
    csv_path = os.path.join(out_root, 'lj13_chain_diagnostics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'\nsaved -> {csv_path}', flush=True)


if __name__ == '__main__':
    main()
