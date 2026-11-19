#!/usr/bin/env python3
"""Revision comparison A on the v4 many-well targets: one frozen checkpoint, several update rules.

Methods (all use the same kernel, momentum laws, maps, path length and fixed flip probability):
  snis          independent paths, self-normalised weights e^{-W}
  path_imh      independence MH on path space (1 path / transition)
  round_trip    reverse path to the base, fresh forward path (2 paths / transition)
  rt_partial:K  round trip through the stage-(T-K) configuration (2K/T paths / transition)
Every method gets the same budget of complete-path equivalents per run, and the run is repeated with
independent seeds, so both within-run (tau_int) and between-run effective counts are reported.

Chains start from the sampler itself (`--init proposal`, default): forward-path endpoints, and for path-IMH
the path record is the chain's own forward path (a valid extended-space state); burn-in is discarded and a
burn-in check is written per run. `--init proposal_worst` keeps only forward endpoints in the least probable
mode (all double wells negative; own records), for relaxation studies. Starts from exact target draws
(`exact`, `worst_mode`: exact draws folded into the worst mode, IMH record = a reverse path) are legacy and
need `--allow-exact-init`.

Observables (exact means and variances from one-dimensional quadrature):
  sign1     sign of the first double-well coordinate              (mode label)
  frac_pos  fraction of positive double-well coordinates          (mode label, all wells)
  abs_even  mean |x| over double-well coordinates                 (within-mode)
  odd_sq    mean x^2 over Gaussian coordinates                    (within-mode)
  energy    -log gamma(x)
plus the L1 error of the 2^n mode probabilities and the per-transition mode change rate (accepted flow).
Cost per effective sample = cost per transition x tau_int, in paths, force evaluations and seconds.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dw_v2'))
sys.path.insert(0, str(HERE.parents[1]))   # code/ (chainstats, chain_ckpt)

import chain_ckpt as ck  # noqa: E402
import chainstats as cs  # noqa: E402
from dw_chains import _sign_flip_layer, reverse_rollout, forward_rollout_from  # noqa: E402
from eval_fixed_flip_ablation import set_fixed_flip_probability  # noqa: E402
from neural_nhmc_dw import ManyWellTarget, base_log_prob, rollout  # noqa: E402
from train_block_dw_nhmc import build_kernel, seed_all  # noqa: E402


# ----------------------------------------------------------------------------- exact reference
def exact_moments(n_wells):
    g = torch.linspace(-8.0, 8.0, 800001, dtype=torch.float64)
    dens = torch.exp(-g ** 4 + 6 * g ** 2 + 0.5 * g)
    z = torch.trapz(dens, g)
    E = lambda f: float(torch.trapz(dens * f, g) / z)
    p_pos = E((g > 0).double())
    m_abs, m_abs2 = E(g.abs()), E(g ** 2)
    u = g ** 4 - 6 * g ** 2 - 0.5 * g
    m_u, m_u2 = E(u), E(u ** 2)
    n = n_wells
    return {
        'sign1': (2 * p_pos - 1, 1 - (2 * p_pos - 1) ** 2),
        'frac_pos': (p_pos, p_pos * (1 - p_pos) / n),
        'abs_even': (m_abs, (m_abs2 - m_abs ** 2) / n),
        'odd_sq': (1.0, 2.0 / n),
        'energy': (n * (m_u + 0.5), n * ((m_u2 - m_u ** 2) + 0.5)),
    }, p_pos


def mode_index(x):
    pos = (x[:, 0::2] > 0).long()
    return (pos * (2 ** torch.arange(pos.shape[1], device=x.device))).sum(1)


def exact_mode_probs(n_wells, p_pos):
    k = torch.arange(2 ** n_wells)
    npos = torch.tensor([bin(int(i)).count('1') for i in k], dtype=torch.float64)
    return p_pos ** npos * (1 - p_pos) ** (n_wells - npos)


def observables(x, target):
    x = x.double()
    ev, od = x[:, 0::2], x[:, 1::2]
    return {
        'sign1': torch.sign(ev[:, 0]),
        'frac_pos': (ev > 0).double().mean(1),
        'abs_even': ev.abs().mean(1),
        'odd_sq': (od ** 2).mean(1),
        'energy': -target.log_prob(x),
    }


# ----------------------------------------------------------------------------- partial round trip
@torch.no_grad()
def partial_round_trip(kernel, target, x, num_steps, depth):
    """Reverse the last `depth` stages from x to u (stage T-depth), forward again from u.
    Returns (y, w_x, w_y) with w = sum_{t > T-depth}(log f_t - log r_t) - log gamma(endpoint);
    the density of u cancels in w_x - w_y."""
    from torch.distributions import Normal
    flip_kernel = kernel if hasattr(kernel, 'flip_logits') else None
    base = kernel.base if flip_kernel is not None else kernel
    B = x.size(0)
    cur, inc = x, torch.zeros(B, device=x.device, dtype=x.dtype)
    for s in range(num_steps, num_steps - depth, -1):
        t_back = torch.full((B,), (s + 1) / num_steps, device=x.device, dtype=x.dtype)
        t_stage = torch.full((B,), s / num_steps, device=x.device, dtype=x.dtype)
        mu_b = base._momentum_mean(cur, t_back, reverse=True)
        sig_b = torch.exp(base._log_sigma(cur, t_back, s + 1, reverse=True))
        db = Normal(mu_b, sig_b)
        m = db.sample()
        lb = db.log_prob(m).sum(1)
        x_prev, p_out = base._leapfrog(cur, m, base.epsilon_at(s), t_stage, base.beta(s, num_steps), s)
        mu_f = base._momentum_mean(x_prev, t_stage, reverse=False)
        sig_f = torch.exp(base._log_sigma(x_prev, t_stage, s, reverse=False))
        lf = Normal(mu_f, sig_f).log_prob(-p_out).sum(1)
        if flip_kernel is not None:
            x_prev, lrf, lrb = _sign_flip_layer(flip_kernel, s, num_steps, x_prev)
            lf, lb = lf + lrf, lb + lrb
        inc = inc + (lf - lb)
        cur = x_prev
    w_x = inc - target.log_prob(x)
    y, inc_y = cur, torch.zeros_like(inc)
    for step in range(num_steps - depth + 1, num_steps + 1):
        y, lf, lb = kernel.forward_step(y, step, num_steps)
        inc_y = inc_y + lf - lb
    return y, w_x, inc_y - target.log_prob(y)


# ----------------------------------------------------------------------------- one chain run
INIT_SEED_OFFSET = 100_000_000      # start-state stream != chain stream


@torch.no_grad()
def proposal_start(kernel, target, n_chains, num_steps, seed, dev, worst=False, batch=4096, max_paths=4_000_000):
    """Forward-path endpoints and their own works (the IMH path records); non-finite paths are dropped and
    redrawn. worst=True keeps only endpoints with every double-well coordinate negative; more paths are drawn
    (up to max_paths) until n_chains are found, and only if that fails the found subset is resampled with
    replacement."""
    # DW8 worst mode: ~6.5 % of forward endpoints under q (0.15^4 under pi), so ~4k paths for 256 chains.
    torch.manual_seed(seed + INIT_SEED_OFFSET)
    bs = batch if worst else n_chains
    xs, ws, n, drawn, dropped = [], [], 0, 0, 0
    while n < n_chains and drawn < max_paths:
        x, w = rollout(kernel, target, bs, num_steps, dev)
        drawn += bs
        fin = torch.isfinite(w) & torch.isfinite(x).all(1)          # a non-finite record would freeze a chain
        dropped += int((~fin).sum())
        keep = fin & (x[:, 0::2] < 0).all(1) if worst else fin
        xs.append(x[keep])
        ws.append(w[keep])
        n += int(keep.sum())
    x, w = torch.cat(xs), torch.cat(ws)
    info = dict(n_paths_drawn=drawn, n_candidates=int(x.shape[0]), n_dropped_nonfinite=dropped, resampled=False)
    if x.shape[0] == 0:
        raise SystemExit(f'proposal start: no usable forward endpoint (worst mode={worst}, finite W) in {drawn} paths')
    if x.shape[0] >= n_chains:
        x, w = x[:n_chains], w[:n_chains]
    else:
        idx = torch.randint(x.shape[0], (n_chains,), device=x.device)
        x, w = x[idx], w[idx]
        info['resampled'] = True
    return x, w, info


TRACE_KEYS = ('sign1', 'frac_pos', 'abs_even', 'odd_sq', 'energy')


@torch.no_grad()
def run_chain(method, depth, kernel, target, n_chains, n_trans, num_steps, init, seed, dev, ckpt=None, tag=None):
    """ckpt / tag: chain_ckpt.Checkpointer and run tag; a saved state of this tag is resumed (the start is then
    not redrawn)."""
    resume = ckpt.load(tag) if ckpt is not None else None
    work, init_info = None, {}
    if resume is not None:
        x = resume['x'].to(dev)
        work = None if resume['work'] is None else resume['work'].to(dev)
        init_info = resume['init_info']
    elif init in ('proposal', 'proposal_worst'):
        x, work, init_info = proposal_start(kernel, target, n_chains, num_steps, seed, dev,
                                            worst=(init == 'proposal_worst'))
        torch.manual_seed(seed)
    else:                                        # legacy exact-target starts (--allow-exact-init)
        torch.manual_seed(seed)
        x = target.sample(n_chains, seed=seed, device=dev, dtype=torch.float32)
        if init == 'worst_mode':                 # every chain in the least probable mode (all wells negative)
            x[:, 0::2] = -x[:, 0::2].abs()
        if method == 'path_imh':
            _, work = reverse_rollout(kernel, target, x, num_steps)   # stationary path record given x
    if method != 'path_imh':
        work = None
    traces = {k: np.empty((n_trans, n_chains)) for k in TRACE_KEYS}
    modes = np.empty((n_trans, n_chains), dtype=np.int64)
    acc = np.zeros(n_trans)
    acc_trace = np.zeros((n_trans, n_chains), dtype=bool)
    start, elapsed = 0, 0.0
    if resume is not None:
        start, elapsed = resume['step'], resume['elapsed']
        for k in TRACE_KEYS:
            traces[k][:start] = resume['traces'][k]
        modes[:start] = resume['modes']
        acc[:start] = resume['acc']
        acc_trace[:start] = resume['acc_trace']
        ck.set_rng_state(resume['rng'])
    t0 = time.time() - elapsed
    for it in range(start, n_trans):
        if method == 'path_imh':
            y, w_y = rollout(kernel, target, n_chains, num_steps, dev)
            w_x = work
        elif method == 'round_trip':
            u, w_x = reverse_rollout(kernel, target, x, num_steps)
            y, w_y = forward_rollout_from(kernel, target, u, num_steps)
        else:
            y, w_x, w_y = partial_round_trip(kernel, target, x, num_steps, depth)
        la = (w_x - w_y).double()
        la = torch.where(torch.isfinite(la), la, torch.full_like(la, -math.inf))
        a = torch.log(torch.rand_like(la)) < la
        x = torch.where(a[:, None], y, x)
        if method == 'path_imh':
            work = torch.where(a, w_y, work)
        acc[it] = float(a.double().mean())
        acc_trace[it] = a.cpu().numpy()
        for k, v in observables(x, target).items():
            traces[k][it] = v.cpu().numpy()
        modes[it] = mode_index(x).cpu().numpy()
        if ckpt is not None:
            n = it + 1
            ckpt.step(tag, n, lambda: dict(step=n, x=ck.to_cpu(x), work=ck.to_cpu(work),
                                           traces={k: traces[k][:n] for k in TRACE_KEYS}, modes=modes[:n],
                                           acc=acc[:n], acc_trace=acc_trace[:n], elapsed=time.time() - t0,
                                           init_info=init_info), force=(n == n_trans))
    return traces, modes, acc, time.time() - t0, acc_trace, init_info


def summarise_chain(traces, modes, acc, burn, exact, p_pos, n_wells, cost, acc_trace=None, relax=False):
    out = dict(acceptance=float(acc[burn:].mean()))
    if acc_trace is not None:
        out['holding'] = cs.holding_times(acc_trace[burn:])
    if relax:                                    # full traces from transition 1, before burn-in removal
        out['relaxation'] = {k: cs.relaxation_curve(traces[k]) for k in ('sign1', 'frac_pos', 'energy')}
        ex = exact_mode_probs(n_wells, p_pos).numpy()
        l1 = [float(np.abs(np.bincount(modes[t], minlength=ex.size) / modes.shape[1] - ex).sum())
              for t in range(modes.shape[0])]
        out['relaxation']['mode_l1_per_transition'] = cs.relaxation_curve(np.asarray(l1)[:, None])
    for k, tr in traces.items():
        m, v = exact[k]
        out[k] = cs.observable_summary(tr[burn:], exact_mean=m, exact_var=v, cost_per_transition=cost)
    md = modes[burn:]
    ex = exact_mode_probs(n_wells, p_pos).numpy()
    obs = np.bincount(md.reshape(-1), minlength=ex.size) / md.size
    out['mode_l1'] = float(np.abs(obs - ex).sum())
    out['mode_change_rate'] = float((md[1:] != md[:-1]).mean())
    out['modes_observed'] = int((obs > 0).sum())
    out['burnin_check'] = cs.burnin_check(traces, burn, early_keys=('frac_pos', 'energy'))
    return out


SNIS_BATCH = 8192


@torch.no_grad()
def run_snis(kernel, target, n_paths, num_steps, seed, dev, batch=None, ckpt=None, tag=None):
    batch = SNIS_BATCH if batch is None else batch
    resume = ckpt.load(tag) if ckpt is not None else None
    if resume is None:
        torch.manual_seed(seed)
        xs, ws, start, elapsed = [], [], 0, 0.0
    else:                                   # batches done so far, concatenated (the final torch.cat is the same)
        xs, ws, start, elapsed = [resume['x']], [resume['w']], resume['step'], resume['elapsed']
        ck.set_rng_state(resume['rng'])
    t0 = time.time() - elapsed
    for i in range(start, n_paths, batch):
        b = min(batch, n_paths - i)
        x, w = rollout(kernel, target, b, num_steps, dev)
        xs.append(x.double().cpu())
        ws.append(w.double().cpu())
        if ckpt is not None:
            n = i + b
            ckpt.step(tag, n, lambda: dict(step=n, x=torch.cat(xs), w=torch.cat(ws), elapsed=time.time() - t0),
                      force=(n == n_paths))
    return torch.cat(xs), torch.cat(ws), time.time() - t0


def summarise_snis(x, w, target, exact, p_pos, n_wells):
    lw = (-w).numpy()
    out = dict(kish_ess=cs.kish_ess(lw), n_paths=int(x.shape[0]))
    out['kish_ess_frac'] = out['kish_ess'] / out['n_paths']
    for k, v in observables(x, target).items():
        m = cs.snis_mean(v.numpy(), lw)
        out[k] = dict(mean=m, exact_mean=exact[k][0], error=m - exact[k][0])
    ex = exact_mode_probs(n_wells, p_pos).numpy()
    wn = np.exp(lw - lw[np.isfinite(lw)].max())
    wn[~np.isfinite(wn)] = 0
    est = np.bincount(mode_index(x).numpy(), weights=wn, minlength=ex.size) / wn.sum()
    out['mode_l1'] = float(np.abs(est - ex).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--checkpoint', default='kernel_best.pt')
    ap.add_argument('--p_flip', default='0.05', help="fixed flip probability, or 'learned'")
    ap.add_argument('--methods', nargs='+', default=['snis', 'path_imh', 'round_trip'])
    ap.add_argument('--n_chains', type=int, default=256)
    ap.add_argument('--path_budget', type=int, default=256 * 2000, help='complete-path equivalents per run')
    ap.add_argument('--n_runs', type=int, default=3)
    ap.add_argument('--init', choices=['proposal', 'proposal_worst', 'exact', 'worst_mode'], default='proposal',
                    help='proposal: forward-path endpoints, own records (default); proposal_worst: the same in the '
                         'worst mode; exact / worst_mode: LEGACY exact-target starts, need --allow-exact-init')
    ap.add_argument('--allow-exact-init', action='store_true',
                    help='permit the legacy exact-target starts (not allowed for NHMC chain results)')
    ap.add_argument('--burn_fraction', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    ck.add_args(ap)
    a = ap.parse_args()
    if a.init in ('exact', 'worst_mode'):
        if not a.allow_exact_init:
            raise SystemExit(f'--init {a.init} starts the chains from exact target draws; NHMC chain results must '
                             'start from the sampler. Use --init proposal / proposal_worst (or --allow-exact-init).')
        print(f'WARNING: --init {a.init}: chains start from exact target draws (legacy)', file=sys.stderr, flush=True)
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
    if a.p_flip != 'learned' and hasattr(kernel, 'flip_logits'):
        set_fixed_flip_probability(kernel, float(a.p_flip))
    T, nlf = int(cfg['num_steps']), int(cfg.get('n_leapfrog', 1))
    force_per_path = T * (nlf + 1)
    n_wells = ns.dim // 2
    exact, p_pos = exact_moments(n_wells)
    ckpt = ck.Checkpointer(a.out, ck.args_hash(vars(a), files=[rd / 'config.json', rd / a.checkpoint],
                                               code=ck.code_files(globals())), a.ckpt_min)
    result = dict(args=ck.public_args(a), config=cfg, force_evals_per_path=force_per_path, exact={k: v for k, v in exact.items()},
                  methods={})
    for mi, meth in enumerate(a.methods):
        name, depth = (meth.split(':')[0], int(meth.split(':')[1])) if ':' in meth else (meth, None)
        runs = []
        for r in range(a.n_runs):
            seed = a.seed + 1000 * r + 17 * len(meth)
            tag = ck.Checkpointer.tag(meth, r)
            ckpt.exit_if_term()
            done = ckpt.load_done(tag)
            if done is not None:
                s = done
            elif name == 'snis':
                x, w, sec = run_snis(kernel, target, a.path_budget, T, seed, dev, ckpt=ckpt, tag=tag)
                s = summarise_snis(x, w, target, exact, p_pos, n_wells)
                s.update(seconds=sec)
            else:
                ppt = 1.0 if name == 'path_imh' else (2.0 if name == 'round_trip' else 2.0 * depth / T)
                n_trans = max(int(a.path_budget / (a.n_chains * ppt)), 20)
                tr, md, acc, sec, acct, init_info = run_chain(name, depth, kernel, target, a.n_chains, n_trans, T,
                                                              a.init, seed, dev, ckpt=ckpt, tag=tag)
                burn = int(a.burn_fraction * n_trans)
                cost = dict(paths=ppt, force_evals=ppt * force_per_path, seconds=sec / n_trans)
                s = summarise_chain(tr, md, acc, burn, exact, p_pos, n_wells, cost, acc_trace=acct,
                                    relax=(a.init != 'exact'))
                s.update(seconds=sec, n_transitions=n_trans, burn=burn, paths_per_transition=ppt, init=init_info)
                print(f'[{meth} run {r}] ' + cs.burnin_line(s['burnin_check']), flush=True)
            if done is None:
                ckpt.done(tag, s)
            runs.append(s)
            print(f'[{meth} run {r}] ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                     for k, v in s.items() if not isinstance(v, dict)}), flush=True)
        agg = {}
        for k in exact:
            means = [rr[k]['mean'] for rr in runs]
            secs = np.mean([rr['seconds'] for rr in runs])
            agg[k] = cs.between_runs(means, exact_var=exact[k][1],
                                     cost_per_run=dict(paths=a.path_budget, seconds=secs))
        agg['mode_l1_mean'] = float(np.mean([rr['mode_l1'] for rr in runs]))
        result['methods'][meth] = dict(runs=runs, between_runs=agg)
        # with methods still to come this is a progress copy (a task guard [ -f out ] must not see a partial json)
        ck.atomic_json(result, a.out if mi == len(a.methods) - 1 else a.out + '.partial', indent=1, default=float)
    if pathlib.Path(a.out + '.partial').exists():
        pathlib.Path(a.out + '.partial').unlink()
    ckpt.finish()
    print('wrote', a.out)


if __name__ == '__main__':
    main()
