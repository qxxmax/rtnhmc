#!/usr/bin/env python3
"""
Train the real neural NHMC on the bgflow Lennard-Jones LJ-13 cluster.

Same path-work objective as the double-well / phi^4 trainers:
    loss = E[W] + var_weight * Var(W),   W = logP + S,  S = -log gamma = E(x).

LJ-13 has no analytic logZ, so we monitor ESS, energy contraction, and the
nearest-neighbour distance (the order parameter of the cluster).

RT-BUFFER VARIANT (2026-10-10, phi4_bench/lj_buffer): copy of lj13_seeds/code/train_lj13_nhmc.py plus
  * --rt_buffer: exact round-trip MH replay buffer (as in the phi^4 trainer). Chains are advanced with the
    round-trip MH kernel of the CURRENT model (pi-invariant for any parameters), their states fill a FIFO, and
    each step adds the forward-KL term  -E[W_rev] + var_weight Var(W_rev)  of reverse records generated
    from buffer states with the CURRENT parameters (reparametrised, no stale weights; not the FAB/SNIS or
    high-weight path-replay buffers below). Training only; evaluation never sees the buffer.
  * exact periodic checkpoint last.pt (weights, Adam, LR schedule, RNG, chains, buffer, counters) every
    --ckpt_minutes, resumed automatically when present.

NOTE (recovery provenance): this file was reconstructed on 2026-07-11 from
Cursor's local editor history (.cursor-server/data/User/History/-667b1383/),
because the working copy had been deleted from disk. Content is reproduced
verbatim from the last recorded history snapshot; no logic was changed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.nn as nn

from lj13_nhmc import (LJClusterTarget, LJ13NHMCKernel, rollout, remove_mean,
                       ess_logz, DIM, N_PART, N_SPACE, base_log_prob_scaled)
from lj13_chain_diagnostics import reverse_path, forward_path


@torch.no_grad()
def rt_step(kernel, target, x, num_steps):
    """One round-trip MH step x -> u -> y (current parameters). Returns new states and acceptance."""
    u, log_fx, log_rx, _ = reverse_path(kernel, target, x, num_steps)
    y, log_fy, log_ry = forward_path(kernel, target, u, num_steps)
    lq0 = base_log_prob_scaled(u, kernel.base_scale).to(torch.float64)
    wx = lq0 + (-target.log_prob(x)).to(torch.float64) + log_fx - log_rx
    wy = lq0 + (-target.log_prob(y)).to(torch.float64) + log_fy - log_ry
    acc = (torch.log(torch.rand_like(wx).clamp_min(1e-30)) <= (wx - wy).clamp(max=0.0)) & torch.isfinite(wy)
    return torch.where(acc[:, None], y, x), acc


def reverse_record_work(kernel, target, x, num_steps):
    """Reparametrised reverse record from x with the current parameters; returns
    W_rev = log q0(u) + S(x) + sum_t (log f_t - log r_t)  (= -log w of the record)."""
    cur = x
    lf = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    lr = torch.zeros_like(lf)
    B = x.size(0)
    for t in range(num_steps, 0, -1):
        tn1 = torch.full((B,), (t + 1) / num_steps, device=x.device, dtype=x.dtype)
        tn = torch.full((B,), t / num_steps, device=x.device, dtype=x.dtype)
        beta = kernel._beta(t, num_steps)
        eps = kernel.epsilon_at(t, num_steps)
        mu_b, ls_b = kernel._momentum_params(cur, tn1, reverse=True)
        sb = torch.exp(ls_b)
        m = mu_b + sb * torch.randn_like(mu_b)
        lr = lr + torch.distributions.Normal(mu_b, sb).log_prob(m).sum(1)
        x_prev, minus_p0 = kernel._leapfrog(cur, m, eps, beta)
        mu_f, ls_f = kernel._momentum_params(x_prev, tn, reverse=False)
        lf = lf + torch.distributions.Normal(mu_f, torch.exp(ls_f)).log_prob(-minus_p0).sum(1)
        cur = x_prev
    return base_log_prob_scaled(cur, kernel.base_scale) + (-target.log_prob(x)) + lf - lr

ROOT = os.path.dirname(os.path.abspath(__file__))


def min_pair_dist(x):
    n_particles = x.shape[-1] // N_SPACE
    p = x.view(-1, n_particles, N_SPACE)
    d = torch.cdist(p, p)
    eye = torch.eye(n_particles, device=x.device).bool()
    return d.masked_fill(eye[None], float('inf')).amin(dim=(1, 2))


def logmeanexp(x):
    return torch.logsumexp(x, dim=0) - torch.log(
        torch.as_tensor(x.numel(), device=x.device, dtype=x.dtype)
    )


class PathNoiseReplay:
    """CPU replay bank for random paths that previously carried large weight.

    The bank stores base and momentum standard-normal draws, not detached
    endpoints. Replaying a draw therefore reruns the complete current NHMC path
    and preserves gradients through the current proposal parameters.
    """

    def __init__(self, capacity, num_steps, dim, rank_alpha=0.5):
        self.capacity = int(capacity)
        self.num_steps = int(num_steps)
        self.dim = int(dim)
        self.rank_alpha = float(rank_alpha)
        self.base_noise = torch.empty(0, dim)
        self.momentum_noise = torch.empty(0, num_steps, dim)
        self.priority = torch.empty(0, dtype=torch.float64)

    def __len__(self):
        return int(self.priority.numel())

    def add(self, trace_noise, work, count):
        if self.capacity <= 0 or count <= 0:
            return
        finite = torch.isfinite(work.detach())
        if not finite.any():
            return
        valid_indices = torch.nonzero(finite, as_tuple=False).flatten()
        scores = -work.detach()[valid_indices].to(torch.float64)
        take = min(int(count), scores.numel())
        chosen_local = torch.topk(scores, k=take, largest=True).indices
        chosen = valid_indices[chosen_local]
        new_base = trace_noise['base_noise'][chosen].detach().cpu().to(torch.float32)
        new_momentum = (
            trace_noise['momentum_noise'][:, chosen]
            .permute(1, 0, 2).detach().cpu().to(torch.float32)
        )
        new_priority = scores[chosen_local].cpu()
        self.base_noise = torch.cat([self.base_noise, new_base], dim=0)
        self.momentum_noise = torch.cat([self.momentum_noise, new_momentum], dim=0)
        self.priority = torch.cat([self.priority, new_priority], dim=0)
        if len(self) > self.capacity:
            keep = torch.topk(self.priority, k=self.capacity, largest=True).indices
            self.base_noise = self.base_noise[keep]
            self.momentum_noise = self.momentum_noise[keep]
            self.priority = self.priority[keep]

    def sample(self, count, device, dtype):
        if len(self) == 0:
            raise RuntimeError("cannot sample an empty path replay bank")
        count = min(int(count), len(self))
        order = torch.argsort(self.priority, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(1, len(self) + 1, dtype=torch.float64)
        probabilities = ranks.pow(-self.rank_alpha)
        probabilities = probabilities / probabilities.sum()
        indices = torch.multinomial(probabilities, count, replacement=False)
        trace = {
            'base_noise': self.base_noise[indices].to(device=device, dtype=dtype),
            'momentum_noise': (
                self.momentum_noise[indices].permute(1, 0, 2)
                .to(device=device, dtype=dtype)
            ),
        }
        return trace, indices

    def update(self, indices, work):
        finite = torch.isfinite(work.detach())
        if finite.any():
            idx = indices[finite.cpu()]
            self.priority[idx] = -work.detach()[finite].to(torch.float64).cpu()


class SNISPathBuffer:
    """FAB-style path buffer with self-normalized importance reweighting.

    Motivated by the FAB buffer (Midgley et al. 2023): the buffer stores path
    noise traces together with their last-known log importance weight
    logw = -W.  Because the noise trace is standard normal under the proposal
    path measure FOR ANY parameter value, a stored trace can be replayed
    under the current parameters and still defines a valid proposal path.

    Differences from the naive PathNoiseReplay that hurt v7b:
      * sampling is WITH replacement with probability q_i ∝ exp(logw_i)
        (SNIS resampling toward the TARGET path measure), not rank-based
        prioritized sampling mixed into a proposal-side penalty;
      * the loss side applies a per-draw importance correction
        c_i ∝ exp(-W_i^new) / q_i (self-normalized, clipped), so the buffer
        term estimates the target-side expectation E_pi[W] (a forward-KL
        anchor) instead of biasing the proposal-side ESS estimate;
      * eviction is FIFO (recency), not priority, so the buffer tracks the
        current proposal instead of accumulating stale extreme-weight paths.
    """

    def __init__(self, capacity, num_steps, dim):
        self.capacity = int(capacity)
        self.num_steps = int(num_steps)
        self.dim = int(dim)
        self.base_noise = torch.empty(0, dim)
        self.momentum_noise = torch.empty(0, num_steps, dim)
        self.logw = torch.empty(0, dtype=torch.float64)

    def __len__(self):
        return int(self.logw.numel())

    def add(self, trace_noise, work):
        finite = torch.isfinite(work.detach())
        if not finite.any():
            return
        idx = torch.nonzero(finite, as_tuple=False).flatten()
        new_base = trace_noise['base_noise'][idx].detach().cpu().to(torch.float32)
        new_momentum = (
            trace_noise['momentum_noise'][:, idx]
            .permute(1, 0, 2).detach().cpu().to(torch.float32)
        )
        new_logw = (-work.detach()[idx]).to(torch.float64).cpu()
        self.base_noise = torch.cat([self.base_noise, new_base], dim=0)
        self.momentum_noise = torch.cat([self.momentum_noise, new_momentum], dim=0)
        self.logw = torch.cat([self.logw, new_logw], dim=0)
        if len(self) > self.capacity:
            keep = len(self) - self.capacity
            self.base_noise = self.base_noise[keep:]
            self.momentum_noise = self.momentum_noise[keep:]
            self.logw = self.logw[keep:]

    def sample(self, count, device, dtype):
        if len(self) == 0:
            raise RuntimeError("cannot sample an empty SNIS path buffer")
        count = int(count)
        log_q = self.logw - torch.logsumexp(self.logw, dim=0)
        probabilities = torch.exp(log_q)
        indices = torch.multinomial(probabilities, count, replacement=True)
        trace = {
            'base_noise': self.base_noise[indices].to(device=device, dtype=dtype),
            'momentum_noise': (
                self.momentum_noise[indices].permute(1, 0, 2)
                .to(device=device, dtype=dtype)
            ),
        }
        return trace, indices, log_q[indices]

    def update(self, indices, work):
        finite = torch.isfinite(work.detach())
        if finite.any():
            idx = indices[finite.cpu()]
            self.logw[idx] = (-work.detach()[finite]).to(torch.float64).cpu()


@torch.no_grad()
def fixed_validation(kernel, target, args, dev, eval_seed):
    """Evaluate every checkpoint on the same random proposal pool."""
    devices = [dev.index if dev.index is not None else 0] if dev.type == 'cuda' else []
    xs, ws = [], []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(eval_seed)
        if dev.type == 'cuda':
            torch.cuda.manual_seed_all(eval_seed)
        remaining = args.eval_samples
        while remaining > 0:
            batch = min(args.eval_batch, remaining)
            x, w = rollout(
                kernel, target, batch, args.num_steps, dev,
                base_scale=args.base_scale,
                weight_dtype=torch.float64 if args.weight_float64_eval else None,
            )
            finite = torch.isfinite(w)
            xs.append(x[finite].cpu())
            ws.append(w[finite].cpu())
            remaining -= batch
    x = torch.cat(xs)
    w = torch.cat(ws)
    ess, _ = ess_logz(w)
    normalized_weight = torch.softmax(-w.to(torch.float64), dim=0)
    top_count = max(1, math.ceil(0.01 * normalized_weight.numel()))
    top1_mass = float(torch.topk(normalized_weight, top_count).values.sum())
    energy = -target.log_prob(x.to(dev)).cpu()
    distance = min_pair_dist(x.to(dev)).cpu()
    return {
        'ess': float(ess),
        'n_valid': int(w.numel()),
        'valid_fraction': float(w.numel() / args.eval_samples),
        'energy_median': float(energy.median()),
        'min_distance_median': float(distance.median()),
        'work_std': float(w.std()),
        'top1_mass': top1_mass,
        'eval_seed': int(eval_seed),
    }


@torch.no_grad()
def multi_pool_validation(kernel, target, args, dev):
    seeds = [int(item) for item in args.eval_seeds.split(',') if item.strip()]
    if not seeds:
        seeds = [args.eval_seed]
    pools = [fixed_validation(kernel, target, args, dev, seed) for seed in seeds]
    ess_values = [row['ess'] for row in pools]
    top1_values = [row['top1_mass'] for row in pools]
    return {
        'ess': statistics.median(ess_values),
        'selection_score': min(ess_values),
        'worst_top1_mass': max(top1_values),
        'valid_fraction': min(row['valid_fraction'] for row in pools),
        'work_std': statistics.median(row['work_std'] for row in pools),
        'energy_median': statistics.median(row['energy_median'] for row in pools),
        'min_distance_median': statistics.median(
            row['min_distance_median'] for row in pools
        ),
        'pools': pools,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_particles', type=int, default=N_PART)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--osc_scale', type=float, default=1.0)
    p.add_argument('--lj_scale', type=float, default=0.5,
                   help='LJ term scale over ordered pairs: 0.5=historical soft '
                        'potential (each pair once), 1.0=official iDEM/Adjoint '
                        '(energy_factor=1.0, 2x deeper).')
    p.add_argument('--num_steps', type=int, default=24)
    p.add_argument('--n_leapfrog', type=int, default=5)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--architecture',
                   choices=['mlp', 'equivariant', 'force_equivariant'],
                   default='mlp')
    p.add_argument('--equiv_layers', type=int, default=3)
    p.add_argument('--equiv_rbf', type=int, default=12)
    p.add_argument('--equiv_time_freq', type=int, default=4)
    p.add_argument('--equiv_force_scale', type=float, default=20.0)
    p.add_argument('--equiv_r_max', type=float, default=4.0,
                   help='RBF distance cutoff of the equivariant momentum law '
                        '(raise for clusters larger than LJ13)')
    p.add_argument('--checkpoint_equiv', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--target_lr', type=float, default=1e-5)
    p.add_argument('--var_weight', type=float, default=0.01)
    p.add_argument('--base_scale', type=float, default=1.2)
    p.add_argument('--base_type',
                   choices=['gaussian', 'icosahedral', 'mixture', 'templates'],
                   default='gaussian')
    p.add_argument('--template_files', type=str, default='')
    p.add_argument('--template_coordinate_scale', type=float, default=1.0)
    p.add_argument('--structured_noise_scale', type=float, default=0.16)
    p.add_argument('--structured_template_scale', type=float, default=1.0)
    p.add_argument('--structured_mix_prob', type=float, default=0.5)
    p.add_argument('--condition_on_component', type=int, default=0)
    p.add_argument('--log_eps_init', type=float, default=-6.2)
    p.add_argument('--anneal_power', type=float, default=1.0)
    p.add_argument('--anneal_floor', type=float, default=0.05)
    p.add_argument('--anneal_base', type=int, default=1)
    p.add_argument('--learn_base', type=int, default=1)
    p.add_argument('--stagewise_eps', type=int, default=0)
    p.add_argument('--learn_schedule', type=int, default=0)
    p.add_argument('--schedule_type', choices=['power', 'lj13_focus'], default='power')
    p.add_argument('--schedule_increment_floor', type=float, default=0.0,
                   help='anti-collapse floor for the learned schedule: fraction '
                        'of the beta ramp distributed uniformly across stages '
                        '(0 = previous behaviour)')
    p.add_argument('--momentum_refresh_alpha', type=float, default=None,
                   help='persistent-momentum (underdamped) NHMC: Crank-Nicolson '
                        'refresh coefficient in [0,1); None/omitted = legacy '
                        'full refresh (bitwise-identical at alpha=0)')
    p.add_argument('--learn_mass', type=int, default=0,
                   help='learn a global scalar mass for the leapfrog (x += eps p/m)')
    p.add_argument('--mass_init', type=float, default=1.0)
    p.add_argument('--momentum_mode',
                   choices=['full', 'forward_only', 'fixed'], default='full',
                   help='NHMC momentum-law ablation: full = learned fwd+bwd '
                        '(default), forward_only = learned fwd + fixed N(0,I) bwd, '
                        'fixed = both fixed to standard-normal momentum')
    p.add_argument('--stage_var_weight', type=float, default=0.0,
                   help='weight on sum_t Var_batch(stage log-ratio increment)^2; '
                        'for a fixed total work variance this is minimized by '
                        'spreading the variance evenly across stages, giving the '
                        'learned schedule a flatten-the-work signal')
    p.add_argument('--ess_weight', type=float, default=0.0)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--clamp_window', type=float, default=60.0)
    p.add_argument('--eval_interval', type=int, default=250)
    p.add_argument('--eval_samples', type=int, default=20000)
    p.add_argument('--eval_batch', type=int, default=1000)
    p.add_argument('--eval_seed', type=int, default=9173)
    p.add_argument('--eval_seeds', type=str, default='9173')
    p.add_argument('--weight_float64_eval', type=int, default=1)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--replay_capacity', type=int, default=0)
    p.add_argument('--replay_add', type=int, default=8)
    p.add_argument('--replay_batch', type=int, default=0)
    p.add_argument('--replay_start', type=int, default=100)
    p.add_argument('--replay_ess_weight', type=float, default=0.0)
    p.add_argument('--replay_var_weight', type=float, default=0.0)
    p.add_argument('--replay_rank_alpha', type=float, default=0.5)
    p.add_argument('--snis_buffer_capacity', type=int, default=0,
                   help='FAB-style SNIS path buffer size (0 = off)')
    p.add_argument('--snis_buffer_batch', type=int, default=64)
    p.add_argument('--snis_buffer_start', type=int, default=200)
    p.add_argument('--snis_buffer_weight', type=float, default=0.0,
                   help='weight on the SNIS-reweighted target-side work anchor '
                        'sum_i cbar_i * W_i over buffer paths')
    p.add_argument('--snis_logw_clip', type=float, default=5.0,
                   help='clip on the per-draw log importance correction before '
                        'self-normalization (FAB-style stabilization)')
    p.add_argument('--rt_buffer', type=int, default=0, help='exact round-trip MH replay buffer (1 = on)')
    p.add_argument('--rt_start', type=int, default=300, help='epoch at which chains start and the term switches on')
    p.add_argument('--rt_chains', type=int, default=64)
    p.add_argument('--rt_every', type=int, default=2, help='advance every chain by one RT step every N epochs')
    p.add_argument('--rt_burn', type=int, default=20)
    p.add_argument('--rt_size', type=int, default=20000, help='FIFO capacity')
    p.add_argument('--rt_batch', type=int, default=160, help='buffer states per step')
    p.add_argument('--rt_weight', type=float, default=1.0)
    p.add_argument('--ckpt_minutes', type=float, default=30.0, help='exact training state every N minutes (last.pt)')
    p.add_argument('--save_dir', type=str,
                   default=os.path.join(ROOT, 'results', 'neural_nhmc_dw', 'lj13'))
    args = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    target = LJClusterTarget(
        n_particles=args.n_particles,
        temperature=args.temperature,
        osc_scale=args.osc_scale,
        lj_scale=args.lj_scale,
    )
    dim = target.dim
    kernel = LJ13NHMCKernel(target, n_leapfrog=args.n_leapfrog, hidden=args.hidden,
                            log_eps_init=args.log_eps_init,
                            anneal_power=args.anneal_power,
                            anneal_floor=args.anneal_floor,
                            anneal_base=bool(args.anneal_base),
                            learn_base=bool(args.learn_base),
                            base_scale=args.base_scale,
                            num_steps=args.num_steps,
                            stagewise_eps=bool(args.stagewise_eps),
                            learn_schedule=bool(args.learn_schedule),
                            architecture=args.architecture,
                            equiv_layers=args.equiv_layers,
                            equiv_rbf=args.equiv_rbf,
                            equiv_time_freq=args.equiv_time_freq,
                            equiv_force_scale=args.equiv_force_scale,
                            equiv_r_max=args.equiv_r_max,
                            checkpoint_equiv=bool(args.checkpoint_equiv),
                            schedule_type=args.schedule_type,
                            base_type=args.base_type,
                            structured_noise_scale=args.structured_noise_scale,
                            structured_template_scale=args.structured_template_scale,
                            structured_mix_prob=args.structured_mix_prob,
                            condition_on_component=bool(args.condition_on_component),
                            template_files=[
                                item for item in args.template_files.split(',') if item
                            ],
                            template_coordinate_scale=args.template_coordinate_scale,
                            schedule_increment_floor=args.schedule_increment_floor,
                            momentum_refresh_alpha=args.momentum_refresh_alpha,
                            learn_mass=bool(args.learn_mass),
                            mass_init=args.mass_init,
                            momentum_mode=args.momentum_mode).to(dev)
    if args.replay_capacity > 0 and args.base_type != 'gaussian':
        raise ValueError("path-noise replay currently requires --base_type gaussian")
    if args.snis_buffer_capacity > 0 and args.base_type != 'gaussian':
        raise ValueError("the SNIS path buffer currently requires --base_type gaussian")
    n_params = sum(q.numel() for q in kernel.parameters())
    print(f"LJ-{args.n_particles}  dim={dim}  steps={args.num_steps}  n_lf={args.n_leapfrog}  "
          f"arch={args.architecture}  hidden={args.hidden}  batch={args.batch_size}  "
          f"params={n_params:,}  dev={dev}")

    opt = torch.optim.Adam(kernel.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.target_lr)
    cfg = vars(args).copy(); cfg['n_params'] = n_params
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    replay = PathNoiseReplay(
        args.replay_capacity, args.num_steps, dim, args.replay_rank_alpha
    )
    snis_buffer = SNISPathBuffer(args.snis_buffer_capacity, args.num_steps, dim)
    best_ess = 0.0
    best_selection_score = 0.0
    t0 = time.time()
    if args.rt_buffer and (args.replay_capacity > 0 or args.snis_buffer_capacity > 0):
        raise ValueError("--rt_buffer is not combined with the replay / SNIS path buffers")
    rt = dict(chains=None, buf=None, n_rt_paths=0, acc=float('nan'))
    start_ep, wall0 = 1, 0.0
    last_path = os.path.join(args.save_dir, 'last.pt')
    if os.path.isfile(last_path):
        st = torch.load(last_path, map_location=dev, weights_only=False)
        kernel.load_state_dict(st['kernel']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        best_ess, best_selection_score = st['best_ess'], st['best_selection_score']
        rt = st['rt']
        if rt['chains'] is not None:
            rt['chains'], rt['buf'] = rt['chains'].to(dev), rt['buf'].to(dev)
        torch.set_rng_state(st['rng'])
        if st.get('rng_cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st['rng_cuda'])
        start_ep, wall0 = st['ep'] + 1, st['wall']
        print(f"EXACT RESUME from {last_path} at epoch {st['ep']}", flush=True)
    t0 = time.time() - wall0
    t_ck = time.time()

    def save_last(ep_done):
        rt_cpu = dict(rt)
        if rt['chains'] is not None:
            rt_cpu['chains'], rt_cpu['buf'] = rt['chains'].cpu(), rt['buf'].cpu()
        st = dict(ep=ep_done, kernel=kernel.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  best_ess=best_ess, best_selection_score=best_selection_score, rt=rt_cpu,
                  wall=time.time() - t0, rng=torch.get_rng_state(),
                  rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(st, last_path + '.tmp'); os.replace(last_path + '.tmp', last_path)

    for ep in range(start_ep, args.epochs + 1):
        kernel.train()
        opt.zero_grad()
        losses, means, stds, replay_penalties = [], [], [], []
        stage_penalties, buffer_penalties = [], []
        valid_update = True
        for _ in range(args.grad_accum):
            capture_noise = (args.replay_capacity > 0
                             or args.snis_buffer_capacity > 0)
            capture_stages = args.stage_var_weight > 0
            fresh_output = rollout(
                kernel, target, args.batch_size, args.num_steps, dev,
                base_scale=args.base_scale, return_trace_noise=capture_noise,
                return_stage_increments=capture_stages,
            )
            fresh_output = list(fresh_output)
            stage_increments = fresh_output.pop() if capture_stages else None
            fresh_trace = fresh_output.pop() if capture_noise else None
            _, W_all = fresh_output
            W_finite_mask = torch.isfinite(W_all)
            W = W_all[W_finite_mask]
            if W.numel() < 2:
                valid_update = False
                break
            # Training is robustified only inside the loss. Evaluation always
            # uses the raw, unclipped work values.
            with torch.no_grad():
                med = W.median()
            Wc = W.clamp(med - args.clamp_window, med + args.clamp_window)
            ess_penalty = logmeanexp(-2.0 * Wc) - 2.0 * logmeanexp(-Wc)
            loss = (Wc.mean() + args.var_weight * Wc.var()
                    + args.ess_weight * ess_penalty)

            stage_penalty_value = 0.0
            if capture_stages:
                # Per-stage log-ratio variance across the batch, robustified
                # the same way as the work loss.  Sum of SQUARED stage
                # variances: for a fixed total, minimized by a flat per-stage
                # variance profile -> gives the learned schedule an explicit
                # spread-the-work training signal.
                inc = stage_increments[:, W_finite_mask]
                with torch.no_grad():
                    inc_med = inc.median(dim=1, keepdim=True).values
                inc = inc.clamp(inc_med - args.clamp_window,
                                inc_med + args.clamp_window)
                stage_var = inc.var(dim=1)
                stage_penalty = args.stage_var_weight * stage_var.square().sum()
                loss = loss + stage_penalty
                stage_penalty_value = float(stage_penalty.detach())

            replay_penalty_value = 0.0
            if (ep >= args.replay_start and args.replay_batch > 0
                    and len(replay) >= args.replay_batch):
                replay_trace, replay_indices = replay.sample(
                    args.replay_batch, dev, W.dtype
                )
                _, replay_work_all = rollout(
                    kernel, target, args.replay_batch, args.num_steps, dev,
                    base_scale=args.base_scale, trace_noise=replay_trace,
                )
                replay_finite = torch.isfinite(replay_work_all)
                replay_work = replay_work_all[replay_finite]
                if replay_work.numel() >= 2:
                    combined = torch.cat([W, replay_work], dim=0)
                    with torch.no_grad():
                        combined_median = combined.median()
                    combined = combined.clamp(
                        combined_median - args.clamp_window,
                        combined_median + args.clamp_window,
                    )
                    replay_ess_penalty = (
                        logmeanexp(-2.0 * combined)
                        - 2.0 * logmeanexp(-combined)
                    )
                    replay_variance = combined.var()
                    replay_term = (
                        args.replay_ess_weight * replay_ess_penalty
                        + args.replay_var_weight * replay_variance
                    )
                    loss = loss + replay_term
                    replay_penalty_value = float(replay_term.detach())
                replay.update(replay_indices, replay_work_all)

            if args.rt_buffer and ep >= args.rt_start:
                if rt['chains'] is None:
                    with torch.no_grad():
                        xc, _ = rollout(kernel, target, args.rt_chains, args.num_steps, dev,
                                        base_scale=args.base_scale)
                        for _ in range(args.rt_burn):
                            xc, _ = rt_step(kernel, target, xc, args.num_steps)
                    rt['chains'], rt['buf'] = xc, xc.clone()
                    rt['n_rt_paths'] += args.rt_chains * (1 + 2 * args.rt_burn)
                if ep % args.rt_every == 0:
                    rt['chains'], acc_ = rt_step(kernel, target, rt['chains'], args.num_steps)
                    rt['buf'] = torch.cat([rt['buf'], rt['chains']])[-args.rt_size:]
                    rt['acc'] = float(acc_.float().mean())
                    rt['n_rt_paths'] += 2 * args.rt_chains
                xb = rt['buf'][torch.randint(rt['buf'].size(0), (args.rt_batch,), device=rt['buf'].device)]
                Wr = reverse_record_work(kernel, target, xb, args.num_steps)
                Wr = Wr[torch.isfinite(Wr)]
                rt['n_rt_paths'] += args.rt_batch
                if Wr.numel() >= 2:
                    with torch.no_grad():
                        mr = Wr.median()
                    Wrc = Wr.clamp(mr - args.clamp_window, mr + args.clamp_window)
                    loss = loss + args.rt_weight * (-Wrc.mean() + args.var_weight * Wrc.var())

            buffer_penalty_value = 0.0
            if (args.snis_buffer_capacity > 0 and args.snis_buffer_weight > 0
                    and ep >= args.snis_buffer_start
                    and len(snis_buffer) >= args.snis_buffer_batch):
                buffer_trace, buffer_indices, buffer_logq = snis_buffer.sample(
                    args.snis_buffer_batch, dev, W.dtype
                )
                _, buffer_work_all = rollout(
                    kernel, target, args.snis_buffer_batch, args.num_steps, dev,
                    base_scale=args.base_scale, trace_noise=buffer_trace,
                )
                buffer_finite = torch.isfinite(buffer_work_all)
                buffer_work = buffer_work_all[buffer_finite]
                if buffer_work.numel() >= 2:
                    with torch.no_grad():
                        buffer_med = buffer_work.median()
                    buffer_wc = buffer_work.clamp(
                        buffer_med - args.clamp_window,
                        buffer_med + args.clamp_window,
                    )
                    # Self-normalized importance correction toward the TARGET
                    # path measure: draws came in with q_i prop-to exp(logw_old),
                    # correction log c_i = logw_new - log q_i, clipped around
                    # its median for stability, softmax-normalized, detached
                    # (FAB-style SNIS gradient).
                    with torch.no_grad():
                        log_c = (-buffer_work.detach().to(torch.float64)
                                 - buffer_logq[buffer_finite.cpu()]
                                 .to(dev, torch.float64))
                        log_c = log_c - log_c.median()
                        log_c = log_c.clamp(-args.snis_logw_clip,
                                            args.snis_logw_clip)
                        cbar = torch.softmax(log_c, dim=0).to(buffer_wc.dtype)
                    buffer_penalty = (
                        args.snis_buffer_weight * (cbar * buffer_wc).sum()
                    )
                    loss = loss + buffer_penalty
                    buffer_penalty_value = float(buffer_penalty.detach())
                snis_buffer.update(buffer_indices, buffer_work_all)

            if not torch.isfinite(loss):
                valid_update = False
                break
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach()))
            means.append(float(W.mean().detach()))
            stds.append(float(W.std().detach()))
            replay_penalties.append(replay_penalty_value)
            stage_penalties.append(stage_penalty_value)
            buffer_penalties.append(buffer_penalty_value)
            if fresh_trace is not None:
                if args.replay_capacity > 0:
                    replay.add(fresh_trace, W_all, args.replay_add)
                if args.snis_buffer_capacity > 0:
                    snis_buffer.add(fresh_trace, W_all)
        if not valid_update:
            opt.zero_grad()
            sched.step()
            continue
        nn.utils.clip_grad_norm_(kernel.parameters(), 5.0)
        opt.step(); sched.step()

        loss_value = sum(losses) / len(losses)
        mean_w = sum(means) / len(means)
        std_w = sum(stds) / len(stds)
        replay_penalty_value = sum(replay_penalties) / len(replay_penalties)
        stage_penalty_value = sum(stage_penalties) / len(stage_penalties)
        buffer_penalty_value = sum(buffer_penalties) / len(buffer_penalties)

        if ep % args.log_interval == 0:
            bs = kernel.base_scale
            bs = float(bs.detach()) if torch.is_tensor(bs) else float(bs)
            template_text = ''
            if args.base_type in ('icosahedral', 'mixture', 'templates'):
                template_text = (
                    f" structured_noise={float(kernel.structured_base.noise_scale.detach()):.3f}"
                    f" template={float(kernel.structured_base.template_scale.detach()):.3f}"
                )
            eps = kernel.epsilon.detach().flatten()
            eps_text = (f"{float(eps.item()):.5f}" if eps.numel() == 1 else
                        f"{float(eps.min()):.5f}/{float(eps.median()):.5f}/{float(eps.max()):.5f}")
            print(f"[{ep:5d}/{args.epochs}] loss={loss_value:10.2f}  "
                  f"meanW={mean_w:9.2f}  stdW={std_w:8.2f}  "
                  f"eps[min/med/max]={eps_text}  base={bs:.3f}{template_text}  "
                  f"replay={len(replay)}/{args.replay_capacity}  "
                  f"replay_loss={replay_penalty_value:.3f}  "
                  f"stage_loss={stage_penalty_value:.4f}  "
                  f"buffer={len(snis_buffer)}/{args.snis_buffer_capacity}  "
                  f"buffer_loss={buffer_penalty_value:.4f}  "
                  f"rt_buf={0 if rt['buf'] is None else rt['buf'].size(0)} rt_acc={rt['acc']:.3f} "
                  f"rt_paths={rt['n_rt_paths']}  "
                  f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)

        if ep % args.eval_interval == 0:
            kernel.eval()
            if kernel.learn_schedule:
                with torch.no_grad():
                    beta = kernel.beta_schedule(args.num_steps)
                    dbeta = torch.diff(
                        torch.cat([beta.new_tensor([kernel.anneal_floor]), beta])
                    )
                print(f"   >> learned schedule ep={ep}  "
                      f"dbeta[min/med/max]={float(dbeta.min()):.5f}/"
                      f"{float(dbeta.median()):.5f}/{float(dbeta.max()):.5f}",
                      flush=True)
            metrics = multi_pool_validation(kernel, target, args, dev)
            print(f"   >> MULTI-POOL EVAL ep={ep}  median ESS={metrics['ess']*100:6.2f}%  "
                  f"worst ESS={metrics['selection_score']*100:6.2f}%  "
                  f"worst top1={metrics['worst_top1_mass']*100:6.2f}%  "
                  f"valid={metrics['valid_fraction']:.4f}  "
                  f"work std={metrics['work_std']:.2f}  "
                  f"E med={metrics['energy_median']:8.2f}  "
                  f"min-dist med={metrics['min_distance_median']:.3f}", flush=True)
            if metrics['selection_score'] > best_selection_score:
                best_selection_score = metrics['selection_score']
                best_ess = metrics['ess']
                torch.save(kernel.state_dict(),
                           os.path.join(args.save_dir, 'kernel_best.pt'))
                with open(os.path.join(args.save_dir, 'best_validation.json'), 'w') as f:
                    json.dump({'epoch': ep, **metrics}, f, indent=2)

        if time.time() - t_ck >= args.ckpt_minutes * 60:
            save_last(ep); t_ck = time.time()

    save_last(args.epochs)
    torch.save(kernel.state_dict(), os.path.join(args.save_dir, 'kernel_final.pt'))
    wall = time.time() - t0
    with open(os.path.join(args.save_dir, 'summary.json'), 'w') as f:
        json.dump({'best_ess': best_ess,
                   'best_selection_score': best_selection_score,
                   'wall_sec': wall, 'rt_paths_total': rt['n_rt_paths'],
                   'train_paths_total': args.epochs * args.batch_size * args.grad_accum, **cfg}, f, indent=2)
    print(f"done  best_median_ess={best_ess*100:.2f}%  "
          f"best_worst_pool_ess={best_selection_score*100:.2f}%  "
          f"wall={wall:.0f}s  -> {args.save_dir}", flush=True)


if __name__ == '__main__':
    main()
