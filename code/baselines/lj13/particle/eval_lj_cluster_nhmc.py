#!/usr/bin/env python3
"""Inference-only diagnostics for a variable-size LJ NHMC checkpoint.

2026-10-12 (phi4_bench/particle): the target is rebuilt from config.json via particle_targets.make_target
(system, lj_scale -- the previous version silently used lj_scale=0.5 for every run -- and DW-4), equiv_r_max /
eps_max are passed to the kernel, and three sample sets can be written for the literature metrics:
  --save_samples N : first N proposal endpoints x with their log weights -W        -> samples.npz
  --rt_chains C    : C round-trip MH chains (init: SNIS resample of the pool, --rt_burn steps), --rt_steps kept
                     states each                                                    -> rt_chain.npz
  --mala_steps M   : M pi-invariant MALA steps appended to every saved sample set (weights unchanged: the
                     appended kernel is pi-reversible); step tuned on a separate warm-up batch, then frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import numpy as np

from lj13_nhmc import (
    LJ13NHMCKernel,
    LJClusterTarget,
    N_SPACE,
    base_log_prob_scaled,
    ess_logz,
    load_cartesian_template,
    rollout,
)
from particle_targets import make_target
from lj13_chain_diagnostics import reverse_path, forward_path, reverse_step_with_work


@torch.no_grad()
def rt_step(kernel, target, x, num_steps, depth=None):
    """Round-trip MH. depth=None: back to the base sample u (q0 cancels). depth=K: partial round trip that
    shares the stage-(T-K) configuration u; only the last K stage factors and the endpoint energy enter, the
    density of u cancels as in the full update."""
    if depth is None or depth >= num_steps:
        u, log_fx, log_rx, _ = reverse_path(kernel, target, x, num_steps)
        y, log_fy, log_ry = forward_path(kernel, target, u, num_steps)
        lq0 = base_log_prob_scaled(u, kernel.base_scale).to(torch.float64)
    else:
        u = x
        log_fx = torch.zeros(x.size(0), device=x.device, dtype=torch.float64)
        log_rx = torch.zeros_like(log_fx)
        for t in range(num_steps, num_steps - depth, -1):
            u, lf, lr, _ = reverse_step_with_work(kernel, u, t, num_steps)
            log_fx += lf.to(torch.float64)
            log_rx += lr.to(torch.float64)
        y = u
        log_fy, log_ry = torch.zeros_like(log_fx), torch.zeros_like(log_fx)
        for t in range(num_steps - depth + 1, num_steps + 1):
            y, lf, lb = kernel.forward_step(y, t, num_steps)
            log_fy += lf.to(torch.float64)
            log_ry += lb.to(torch.float64)
        lq0 = torch.zeros_like(log_fx)
    wx = lq0 + (-target.log_prob(x.to(torch.float64))) + log_fx - log_rx
    wy = lq0 + (-target.log_prob(y.to(torch.float64))) + log_fy - log_ry
    ok = torch.isfinite(wy) & torch.isfinite(y).all(1)
    log_a = torch.where(ok, (wx - wy).clamp(max=0.0), torch.full_like(wx, -math.inf))
    acc = torch.log(torch.rand_like(wx).clamp_min(1e-30)) <= log_a
    return torch.where(acc[:, None], y, x), acc, torch.where(ok[:, None], y, x), log_a


def _remove_mean(x, n):
    p = x.view(-1, n, N_SPACE)
    return (p - p.mean(1, keepdim=True)).reshape(x.shape)


@torch.no_grad()
def mala(target, x, eps, steps):
    """pi-reversible MALA on the zero-COM subspace (target forces are zero-COM). Returns x, mean acceptance."""
    n = target.n_particles
    x = x.to(torch.float64)
    lp, f = target.log_prob(x), target.force(x).to(torch.float64)
    accs = []
    for _ in range(steps):
        z = _remove_mean(torch.randn_like(x), n)
        y = x + eps * f + math.sqrt(2 * eps) * z
        lpy, fy = target.log_prob(y), target.force(y).to(torch.float64)
        lq_xy = -((y - x - eps * f) ** 2).sum(1) / (4 * eps)
        lq_yx = -((x - y - eps * fy) ** 2).sum(1) / (4 * eps)
        la = lpy + lq_yx - lp - lq_xy
        acc = (torch.log(torch.rand_like(la)) < la) & torch.isfinite(la)
        x = torch.where(acc[:, None], y, x)
        lp = torch.where(acc, lpy, lp)
        f = torch.where(acc[:, None], fy, f)
        accs.append(float(acc.double().mean()))
    return x, (sum(accs) / len(accs) if accs else float('nan'))


def tune_mala(target, x, steps=50, goal=0.57):
    eps = 1e-4
    for _ in range(steps):
        _, a = mala(target, x, eps, 1)
        eps *= math.exp(2.0 * (a - goal))
    return eps


def stationary_imh_acceptance(log_weights, n_pairs, seed):
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.softmax(log_weights, dim=0)
    current = torch.multinomial(
        probabilities, n_pairs, replacement=True, generator=generator
    )
    proposed = torch.randint(log_weights.numel(), (n_pairs,), generator=generator)
    log_alpha = torch.minimum(
        torch.zeros(n_pairs, dtype=log_weights.dtype),
        log_weights[proposed] - log_weights[current],
    )
    return float(torch.exp(log_alpha).mean())


def sorted_pair_fingerprint(x, n_particles):
    positions = x.view(-1, n_particles, N_SPACE)
    distances = torch.cdist(positions, positions)
    pair = torch.triu_indices(n_particles, n_particles, offset=1, device=x.device)
    return distances[:, pair[0], pair[1]].sort(dim=1).values


CKPT = 'kernel_best.pt'


def build_kernel(config, run_dir, device):
    target = make_target(config)
    template_files = [item for item in config.get('template_files', '').split(',') if item]
    kernel = LJ13NHMCKernel(
        target,
        n_leapfrog=config['n_leapfrog'],
        hidden=config['hidden'],
        log_eps_init=config.get('log_eps_init', -6.2),
        anneal_power=config.get('anneal_power', 1.0),
        anneal_floor=config.get('anneal_floor', 0.05),
        anneal_base=bool(config.get('anneal_base', 1)),
        learn_base=bool(config.get('learn_base', 1)),
        base_scale=config.get('base_scale', 1.2),
        num_steps=config['num_steps'],
        stagewise_eps=bool(config.get('stagewise_eps', 0)),
        learn_schedule=bool(config.get('learn_schedule', 0)),
        architecture=config.get('architecture', 'mlp'),
        equiv_layers=config.get('equiv_layers', 3),
        equiv_rbf=config.get('equiv_rbf', 12),
        equiv_time_freq=config.get('equiv_time_freq', 4),
        equiv_force_scale=config.get('equiv_force_scale', 20.0),
        equiv_r_max=config.get('equiv_r_max', 4.0),
        eps_max=config.get('eps_max', 1e-2),
        drift=config.get('drift', 'exact'),
        law_force_input=bool(config.get('law_force_input', 1)),
        drift_hidden=config.get('drift_hidden', 64),
        drift_layers=config.get('drift_layers', 3),
        drift_arch=config.get('drift_arch', 'egnn'),
        drift_lmax=config.get('drift_lmax', 2),
        drift_corr=config.get('drift_corr', 3),
        drift_pair=bool(config.get('drift_pair', 0)),
        drift_clamp=config.get('drift_clamp', 20.0),
        softcore_alpha=config.get('softcore_alpha', 0.0),
        softcore_power=config.get('softcore_power', 1.0),
        softcore_learn=config.get('softcore_learn', 0),
        soft_sigma=bool(config.get('soft_sigma', 0)),
        checkpoint_equiv=bool(config.get('checkpoint_equiv', 1)),
        schedule_type=config.get('schedule_type', 'power'),
        base_type=config.get('base_type', 'gaussian'),
        structured_noise_scale=config.get('structured_noise_scale', 0.16),
        structured_template_scale=config.get('structured_template_scale', 1.0),
        structured_mix_prob=config.get('structured_mix_prob', 0.5),
        condition_on_component=bool(config.get('condition_on_component', 0)),
        template_files=template_files,
        template_coordinate_scale=config.get('template_coordinate_scale', 1.0),
        momentum_mode=config.get('momentum_mode', 'full'),
    ).to(device)
    state = torch.load(run_dir / CKPT, map_location=device)
    kernel.load_state_dict(state)
    return target, kernel.eval(), template_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=271828)
    parser.add_argument('--imh_pairs', type=int, default=500000)
    parser.add_argument('--ckpt', default='kernel_best.pt')
    parser.add_argument('--save_samples', type=int, default=0)
    parser.add_argument('--rt_chains', type=int, default=0)
    parser.add_argument('--rt_burn', type=int, default=50)
    parser.add_argument('--rt_steps', type=int, default=100)
    parser.add_argument('--rt_local', type=int, default=0,
                        help='partial round trips per RT iteration (after the full one); 0 = off')
    parser.add_argument('--rt_local_depth', type=int, default=8, help='stages reversed by a partial round trip')
    parser.add_argument('--rt_init', choices=['snis', 'raw'], default='snis',
                        help='chain starts: SNIS resample of the pool (default) or raw forward samples (tests RT alone)')
    parser.add_argument('--mala_steps', type=int, default=0)
    parser.add_argument('--tag', default='', help='suffix for output files')
    args = parser.parse_args()
    global CKPT
    CKPT = args.ckpt

    config = json.loads((args.run_dir / 'config.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    target, kernel, template_files = build_kernel(config, args.run_dir, device)

    template_fingerprints = None
    if template_files:
        templates = torch.stack([
            load_cartesian_template(
                path,
                target.n_particles,
                config.get('template_coordinate_scale', 1.0),
            ).reshape(-1)
            for path in template_files
        ]).to(device)
        template_fingerprints = sorted_pair_fingerprint(
            templates, target.n_particles
        )

    sample_parts, work_parts, basin_parts = [], [], []
    remaining = args.samples
    with torch.no_grad():
        while remaining:
            batch = min(args.batch, remaining)
            samples, work = rollout(
                kernel,
                target,
                batch,
                config['num_steps'],
                device,
                weight_dtype=torch.float64,
            )
            sample_parts.append(samples.cpu().to(torch.float32))
            work_parts.append(work.cpu().to(torch.float64))
            if template_fingerprints is not None:
                fingerprint = sorted_pair_fingerprint(samples, target.n_particles)
                distance = torch.sqrt(
                    (fingerprint[:, None] - template_fingerprints[None]).square().mean(2)
                )
                basin_parts.append(distance.argmin(1).cpu())
            remaining -= batch

    samples = torch.cat(sample_parts)
    work_all = torch.cat(work_parts)
    finite = torch.isfinite(work_all) & torch.isfinite(samples).all(1)
    samples = samples[finite]
    work = work_all[finite]
    log_weights = -work
    weights = torch.softmax(log_weights, dim=0)
    ess, log_z = ess_logz(work)
    energy = -target.log_prob(samples.to(device, dtype=torch.float64)).cpu()
    position = samples.view(-1, target.n_particles, N_SPACE)
    distance = torch.cdist(position, position)
    distance.masked_fill_(
        torch.eye(target.n_particles, dtype=torch.bool)[None], float('inf')
    )
    min_distance = distance.amin((1, 2))
    top_count = max(1, math.ceil(0.01 * work.numel()))

    summary = {
        'run_dir': str(args.run_dir),
        'n_particles': target.n_particles,
        'temperature': target.temperature,
        'osc_scale': target.osc_scale,
        'n_paths_requested': args.samples,
        'n_paths_valid': int(work.numel()),
        'valid_fraction': float(finite.double().mean()),
        'normalized_ess': float(ess),
        'log_z_hat': float(log_z),
        'work_std': float(work.std(unbiased=False)),
        'top1_weight_mass': float(torch.topk(weights, top_count).values.sum()),
        'stationary_path_imh_acceptance': stationary_imh_acceptance(
            log_weights, args.imh_pairs, args.seed + 1
        ),
        'raw_energy_mean': float(energy.mean()),
        'snis_energy_mean': float((weights * energy).sum()),
        'raw_min_pair_distance_mean': float(min_distance.mean()),
        'snis_min_pair_distance_mean': float((weights * min_distance).sum()),
        'force_evaluations_per_path': int(kernel.force_calls_per_path(config['num_steps'])),
        'drift': config.get('drift', 'exact'),
        'law_force_input': int(config.get('law_force_input', 1)),
    }
    if basin_parts:
        basin = torch.cat(basin_parts)[finite]
        counts = torch.bincount(basin, minlength=len(template_files)).double()
        weighted = torch.zeros(len(template_files), dtype=torch.float64)
        weighted.scatter_add_(0, basin, weights)
        summary['pair_fingerprint_templates'] = template_files
        summary['raw_template_basin_mass'] = (counts / counts.sum()).tolist()
        summary['snis_template_basin_mass'] = weighted.tolist()

    summary['system'] = config.get('system', 'lj')
    summary['lj_scale'] = getattr(target, 'lj_scale', None)
    summary['ckpt'] = args.ckpt
    fpp = summary['force_evaluations_per_path']
    mala_eps = None
    if args.mala_steps > 0 and (args.save_samples > 0 or args.rt_chains > 0):
        idx0 = torch.multinomial(weights, min(1024, work.numel()), replacement=True)
        mala_eps = tune_mala(target, samples[idx0].to(device))
        summary['mala_eps'] = mala_eps
    if args.save_samples > 0:
        n_s = min(args.save_samples, work.numel())
        xs = samples[:n_s].to(device)
        out = dict(x=xs.cpu().numpy(), logw=log_weights[:n_s].numpy())
        if mala_eps is not None:
            xm = torch.cat([mala(target, xs[i:i + 4096], mala_eps, args.mala_steps)[0]
                            for i in range(0, n_s, 4096)])
            out['x_mala'] = xm.cpu().numpy()
            summary['mala_force_evals_per_sample'] = args.mala_steps
        np.savez_compressed(args.run_dir / f'samples{args.tag}.npz', **out)
    if args.rt_chains > 0:
        gen = torch.Generator().manual_seed(args.seed + 7)
        if args.rt_init == 'raw':
            idx = torch.randperm(samples.shape[0], generator=gen)[:args.rt_chains]
        else:
            idx = torch.multinomial(weights, args.rt_chains, replacement=True, generator=gen)
        xc = samples[idx].to(device)
        accs, loc_accs, keep, prev, prop, loga = [], [], [], [], [], []
        for it in range(args.rt_burn + args.rt_steps):
            x0 = xc
            xc, a, y, la = rt_step(kernel, target, xc, config['num_steps'])
            accs.append(float(a.double().mean()))
            for _ in range(args.rt_local):                  # composition of pi-invariant kernels
                xc, al, _, _ = rt_step(kernel, target, xc, config['num_steps'], depth=args.rt_local_depth)
                loc_accs.append(float(al.double().mean()))
            if it >= args.rt_burn:
                keep.append(xc.float().cpu())
                prev.append(x0.float().cpu()); prop.append(y.float().cpu()); loga.append(la.cpu())
        chain = torch.stack(keep, 1)                         # [C, steps, dim]
        flat = lambda v: torch.stack(v, 1).reshape(-1, chain.shape[-1]).numpy()
        # x_prev/x_prop/log_alpha: per-step current state, proposal and log acceptance probability, for the
        # waste-recycling estimator (rt_recycle.py)
        out = dict(x=chain.reshape(-1, chain.shape[-1]).numpy(), chain_shape=np.array(chain.shape[:2]),
                   x_prev=flat(prev), x_prop=flat(prop), log_alpha=torch.stack(loga, 1).reshape(-1).double().numpy())
        if mala_eps is not None:
            xm = torch.cat([mala(target, chain.reshape(-1, chain.shape[-1])[i:i + 4096].to(device), mala_eps,
                                 args.mala_steps)[0] for i in range(0, chain.shape[0] * chain.shape[1], 4096)])
            out['x_mala'] = xm.cpu().numpy()
        np.savez_compressed(args.run_dir / f'rt_chain{args.tag}.npz', **out)
        summary['rt_acceptance'] = sum(accs[args.rt_burn:]) / max(len(accs[args.rt_burn:]), 1)
        summary['rt_acceptance_burn'] = sum(accs[:args.rt_burn]) / max(args.rt_burn, 1)
        summary['rt_force_evals_per_state'] = 2 * fpp
        if args.rt_local > 0:
            summary['rt_local_acceptance'] = sum(loc_accs) / max(len(loc_accs), 1)
            summary['rt_local'] = args.rt_local
            summary['rt_local_depth'] = args.rt_local_depth
            summary['rt_force_evals_per_iteration'] = 2 * fpp * (1 + args.rt_local * args.rt_local_depth
                                                                 / config['num_steps'])

    output = args.run_dir / f'cluster_evaluation{args.tag}.json'
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
