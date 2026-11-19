#!/usr/bin/env python3
"""DW round-trip chain ensemble, for Table 8.

Table 8 is the only place the round-trip transition is reported against
path-IMH on the same target, and it is reported in three currencies: the
integrated autocorrelation of the retained endpoint, that time multiplied by
the paths a transition costs, and that multiplied again by the force
evaluations a path costs. Our DW evaluation produced none of it -- it measured
independent path pools, never a chain.

Two conventions here differ from the LJ13 table and are easy to carry over by
mistake: the per-chain autocorrelation times are reduced by their median
rather than their mean, and the chain lengths differ between the two methods
(2000 transitions with 400 burned for path-IMH, 1000 with 200 for the round
trip) because a round-trip transition costs two paths.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from chain_stats import cost_normalized_tau, write_summary  # noqa: E402
from eval_block_dw_nhmc import (exact_positive_probability,  # noqa: E402
                                mode_indices, mode_probabilities)
from neural_nhmc_dw import ManyWellTarget, base_log_prob, rollout  # noqa: E402
from train_block_dw_nhmc import build_kernel, seed_all  # noqa: E402


def _sign_flip_layer(kernel, s, num_steps, x_flipped):
    """Undo one stage's Bernoulli sign flip, returning the pre-flip position
    and the flip's contribution to the forward and backward log densities.

    The DW runs wrap the kernel in a per-well sign flip applied *before* the
    base step, and the flip action is part of the path ratio. Going backwards
    the mask has to be drawn from the backward flip probability and then scored
    under both, mirroring the forward direction where it is drawn from the
    forward one and scored under both. Dropping this layer entirely was what
    made the first attempt fail outright, which is the benign failure; scoring
    it under one probability twice would have run and been wrong.
    """
    probability_forward = torch.sigmoid(kernel.flip_logits[s])
    probability_backward = torch.sigmoid(
        kernel.flip_logits[min(s + 1, num_steps)])
    # Kept in the path's dtype on purpose. A float32 mask multiplied by a
    # zero-dimensional float64 probability stays float32 -- dimensioned
    # tensors win that promotion -- which would quietly compute the flip's
    # contribution to a float64 work in single precision.
    flip = (torch.rand(x_flipped.size(0), kernel.n_wells,
                       device=x_flipped.device,
                       dtype=x_flipped.dtype) < probability_backward
            ).to(x_flipped.dtype)
    sign = 1.0 - 2.0 * flip
    x_prev = x_flipped.clone()
    x_prev[:, 0::2] = x_flipped[:, 0::2] * sign

    log_rho_f = (flip * torch.log(probability_forward + 1e-8)
                 + (1 - flip) * torch.log(1 - probability_forward + 1e-8)).sum(1)
    log_rho_b = (flip * torch.log(probability_backward + 1e-8)
                 + (1 - flip) * torch.log(1 - probability_backward + 1e-8)).sum(1)
    return x_prev, log_rho_f, log_rho_b


def reverse_rollout(kernel, target, x, num_steps):
    """Run a path backwards from a target-side point down to the base.

    The work is accumulated exactly as the forward rollout would have for the
    path that ends at this point,

        W = log q_base(u) + sum_t [log f_t - log r_t] - log gamma(x),

    with stage s undoing forward stage s: its backward law is read at
    (s+1)/num_steps and its forward law at s/num_steps, matching what
    forward_step does in the other direction.

    A sign-flip wrapper, if present, is peeled off after the base stage has
    been undone, since the forward direction applies it before.
    """
    from torch.distributions import Normal

    flip_kernel = kernel if hasattr(kernel, 'flip_logits') else None
    if flip_kernel is not None:
        kernel = kernel.base

    batch = x.size(0)
    endpoint = x
    increments = torch.zeros(batch, device=x.device, dtype=x.dtype)

    for s in range(num_steps, 0, -1):
        t_back = torch.full((batch,), (s + 1) / num_steps, device=x.device,
                            dtype=x.dtype)
        t_stage = torch.full((batch,), s / num_steps, device=x.device,
                             dtype=x.dtype)

        mu_b = kernel._momentum_mean(x, t_back, reverse=True)
        sig_b = torch.exp(kernel._log_sigma(x, t_back, s + 1, reverse=True))
        dist_b = Normal(mu_b, sig_b)
        momentum = dist_b.sample()
        log_bwd = dist_b.log_prob(momentum).sum(1)

        x_prev, p_out = kernel._leapfrog(
            x, momentum, kernel.epsilon_at(s), t_stage,
            kernel.beta(s, num_steps), s)
        # The forward leapfrog maps (x_prev, p0) to (x, -p_new) and the
        # backward law scores -p_new, so the draw here plays that role and the
        # same map run from it returns (x_prev, -p0).
        p0 = -p_out

        mu_f = kernel._momentum_mean(x_prev, t_stage, reverse=False)
        sig_f = torch.exp(kernel._log_sigma(x_prev, t_stage, s, reverse=False))
        log_fwd = Normal(mu_f, sig_f).log_prob(p0).sum(1)

        if flip_kernel is not None:
            x_prev, log_rho_f, log_rho_b = _sign_flip_layer(
                flip_kernel, s, num_steps, x_prev)
            log_fwd = log_fwd + log_rho_f
            log_bwd = log_bwd + log_rho_b

        increments = increments + (log_fwd - log_bwd)
        x = x_prev

    return x, base_log_prob(x) + increments - target.log_prob(endpoint)


def forward_rollout_from(kernel, target, x0, num_steps):
    """Forward path from a base point handed in rather than drawn: the shared
    bridge that makes the two legs' densities cancel to a work difference."""
    x = x0
    log_p = base_log_prob(x0)
    for step in range(1, num_steps + 1):
        x, log_fwd, log_bwd = kernel.forward_step(x, step, num_steps)
        log_p = log_p + log_fwd - log_bwd
    return x, log_p - target.log_prob(x)


def run_chain(method, kernel, target, n_chains, n_transitions, num_steps,
              device, seed=0):
    """Return the retained endpoints, shaped (transitions, chains, dim)."""
    torch.manual_seed(seed)

    with torch.no_grad():
        x, work = rollout(kernel, target, n_chains, num_steps, device)
    endpoints = np.empty((n_transitions, n_chains, target.dim),
                         dtype=np.float64)
    accept_trace = np.zeros((n_transitions, n_chains), dtype=bool)

    for step in range(n_transitions):
        with torch.no_grad():
            if method == 'path_imh':
                y, work_proposal = rollout(kernel, target, n_chains,
                                           num_steps, device)
                work_current = work
            else:
                u, work_current = reverse_rollout(kernel, target, x, num_steps)
                y, work_proposal = forward_rollout_from(kernel, target, u,
                                                        num_steps)

        log_alpha = work_current - work_proposal
        log_alpha = torch.where(torch.isfinite(log_alpha), log_alpha,
                                torch.full_like(log_alpha, -float('inf')))
        accept = torch.rand_like(log_alpha).log() < log_alpha
        x = torch.where(accept.unsqueeze(1), y, x)
        if method == 'path_imh':
            work = torch.where(accept, work_proposal, work)

        endpoints[step] = x.double().cpu().numpy()
        accept_trace[step] = accept.cpu().numpy()

        if (step + 1) % 200 == 0:
            print(f'  transition {step + 1}/{n_transitions} '
                  f'accept {accept_trace[:step + 1].mean():.4f}', flush=True)

    return endpoints, accept_trace


def mode_l1(endpoints, n_wells):
    """L1 distance between the chain's mode masses and the exact ones."""
    flat = torch.from_numpy(endpoints.reshape(-1, endpoints.shape[-1]))
    exact = mode_probabilities(n_wells, exact_positive_probability())
    counts = torch.bincount(mode_indices(flat), minlength=exact.numel())
    observed = counts.double() / counts.sum()
    return float((observed - exact).abs().sum()), int((counts > 0).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--checkpoint', default='kernel_best.pt')
    parser.add_argument('--method', choices=('path_imh', 'round_trip'),
                        default='path_imh')
    parser.add_argument('--n_chains', type=int, default=32)
    parser.add_argument('--n_transitions', type=int, default=None,
                        help='defaults to 2000 for path-IMH, 1000 round trip')
    parser.add_argument('--burn_fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--out_prefix', default=None)
    args = parser.parse_args()

    if args.n_transitions is None:
        args.n_transitions = 2000 if args.method == 'path_imh' else 1000
    if args.out_prefix is None:
        args.out_prefix = f'dw_{args.method}'

    run_dir = pathlib.Path(args.run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    namespace = argparse.Namespace(**config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(args.seed, device)

    target = ManyWellTarget(namespace.dim)
    _, kernel = build_kernel(target, namespace)
    state = torch.load(run_dir / args.checkpoint, map_location=device)
    kernel.load_state_dict(state)
    kernel = kernel.to(device).eval()

    num_steps = int(config['num_steps'])
    n_leapfrog = int(config.get('n_leapfrog', 1))
    paths = 1 if args.method == 'path_imh' else 2
    forces_per_path = num_steps * (n_leapfrog + 1)

    endpoints, accept = run_chain(args.method, kernel, target, args.n_chains,
                                  args.n_transitions, num_steps, device,
                                  seed=args.seed)

    burn = int(round(args.n_transitions * args.burn_fraction))
    endpoints, accept = endpoints[burn:], accept[burn:]

    # Table 8 reports the autocorrelation of the retained endpoint's first
    # coordinate.
    tau = cost_normalized_tau(endpoints[:, :, 0], args.n_chains, paths,
                              paths * forces_per_path)
    l1, observed_modes = mode_l1(endpoints, target.dim // 2)

    summary = {
        'method': args.method,
        'run_dir': str(run_dir),
        'n_chains': args.n_chains,
        'n_transitions': args.n_transitions,
        'burn_in': burn,
        'post_burn_transitions': int(endpoints.shape[0]),
        'acceptance_rate': float(accept.mean()),
        'mode_l1': l1,
        'observed_modes': observed_modes,
        'possible_modes': 2 ** (target.dim // 2),
        'force_evals_per_path': forces_per_path,
        **tau,
    }
    np.savez_compressed(run_dir / f'{args.out_prefix}_trace.npz',
                        endpoints=endpoints.astype(np.float32), accept=accept)
    write_summary(run_dir / f'{args.out_prefix}_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items()
                      if not isinstance(v, list)}, indent=2))


if __name__ == '__main__':
    main()
