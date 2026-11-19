#!/usr/bin/env python3
"""Revision comparison A on LJ13 (official potential): one frozen checkpoint, several update rules.

Methods (same kernel, momentum laws, maps and path length):
  snis          independent paths with self-normalised weights
  imh           independence MH on path space (1 path / transition)
  rt            full round trip through the base sample (2 paths / transition)
  rt_partial:K  round trip through the stage-(T-K) configuration (2K/T paths / transition)
Chains start from the sampler itself (--init proposal, default): forward-path endpoints, and for IMH the path
record is the chain's own forward path (a valid extended-space state). A burn-in fraction is discarded and a
burn-in check is written per run. --init raw (legacy relaxation runs) uses forward endpoints too, but with a
fresh reverse-path record and the chain seed for the start (its first IMH proposal replays the start path).
--init val (Klein validation split, i.e. reference states) is legacy and needs --allow-reference-init.
Observables per configuration: energy E(x), radius of gyration Rg, mean pair distance, minimum pair
distance. Reference means/variances come from the test split. Every method gets the same budget of
complete-path equivalents per run and is repeated with independent seeds. Final chain states (or
SNIS-weighted pools) are saved for the literature W2 metrics (lit_metrics.py).
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

import chain_ckpt as ck  # noqa: E402
import chainstats as cs  # noqa: E402
import eval_lj_cluster_nhmc as ev  # noqa: E402
from lj13_chain_diagnostics import forward_path  # noqa: E402

KEYS = ('energy', 'rg', 'mean_pair', 'min_pair')


def observables(x, target, n):
    p = x.double().view(-1, n, 3)
    p = p - p.mean(1, keepdim=True)
    d = torch.cdist(p, p)
    iu = torch.triu_indices(n, n, 1, device=x.device)
    r = d[:, iu[0], iu[1]]
    return {'energy': -target.log_prob(x.double()) * target.temperature,
            'rg': p.pow(2).sum(-1).mean(1).sqrt(), 'mean_pair': r.mean(1), 'min_pair': r.min(1).values}


@torch.no_grad()
def fresh_path(kernel, target, B, T, dev):
    x0, _, _, _, _ = kernel.sample_base(B, dev, torch.float32)
    y, lf, lr = forward_path(kernel, target, x0, T)
    lq0 = ev.base_log_prob_scaled(x0, kernel.base_scale).to(torch.float64)
    return y, lq0 + (-target.log_prob(y.to(torch.float64))) + lf - lr


@torch.no_grad()
def record_work(kernel, target, x, T):
    u, lf, lr, _ = ev.reverse_path(kernel, target, x, T)
    lq0 = ev.base_log_prob_scaled(u, kernel.base_scale).to(torch.float64)
    return lq0 + (-target.log_prob(x.to(torch.float64))) + lf - lr


INIT_SEED_OFFSET = 100_000_000      # start-state stream != chain stream
SNIS_BATCH = 512


@torch.no_grad()
def run_snis(kernel, target, n_paths, T, seed, dev, ckpt=None, tag=None):
    resume = ckpt.load(tag) if ckpt is not None else None
    if resume is None:
        torch.manual_seed(seed)
        xs, ws, start, elapsed = [], [], 0, 0.0
    else:                                   # batches done so far, concatenated (the final torch.cat is the same)
        xs, ws, start, elapsed = [resume['x']], [resume['w']], resume['step'], resume['elapsed']
        ck.set_rng_state(resume['rng'])
    t0 = time.time() - elapsed
    for i in range(start, n_paths, SNIS_BATCH):
        b = min(SNIS_BATCH, n_paths - i)
        y, w = fresh_path(kernel, target, b, T, dev)
        xs.append(y.cpu()); ws.append(w.cpu())
        if ckpt is not None:
            m = i + b
            ckpt.step(tag, m, lambda: dict(step=m, x=torch.cat(xs), w=torch.cat(ws), elapsed=time.time() - t0),
                      force=(m == n_paths))
    return torch.cat(xs), torch.cat(ws), t0


@torch.no_grad()
def run_chain(method, depth, kernel, target, x0, n_trans, T, n, seed, dev, W0=None, ckpt=None, tag=None, resume=None,
              extra=None):
    """W0: IMH path record of the start state (proposal starts: the work of the forward path that produced x0);
    None draws a fresh reverse-path record from x0 (legacy val / raw starts).
    ckpt / tag: chain_ckpt.Checkpointer and run tag. resume: a saved state (x0 and W0 are then ignored).
    extra: stored in the checkpoint as is (start info)."""
    if resume is None:
        torch.manual_seed(seed)
        x = x0.clone()
        W = None
        if method == 'imh':
            W = record_work(kernel, target, x, T) if W0 is None else W0.clone()
        start, elapsed = 0, 0.0
    else:
        x = resume['x'].to(dev)
        W = None if resume['W'] is None else resume['W'].to(dev)
        start, elapsed = resume['step'], resume['elapsed']
    B = x.shape[0]
    tr = {k: np.empty((n_trans, B)) for k in KEYS}
    acc = np.zeros(n_trans)
    acc_trace = np.zeros((n_trans, B), dtype=bool)
    states = np.empty((n_trans, B, x.shape[1]), dtype=np.float32)
    if resume is not None:
        for k in KEYS:
            tr[k][:start] = resume['tr'][k]
        acc[:start] = resume['acc']
        acc_trace[:start] = resume['acc_trace']
        states[:start] = resume['states']
        ck.set_rng_state(resume['rng'])
    t0 = time.time() - elapsed
    for it in range(start, n_trans):
        if method == 'imh':
            y, Wy = fresh_path(kernel, target, B, T, dev)
            la = torch.nan_to_num(W - Wy, nan=-math.inf)
            a = torch.log(torch.rand_like(la)) < la
            x = torch.where(a[:, None], y, x)
            W = torch.where(a, Wy, W)
        else:
            x, a, _, _ = ev.rt_step(kernel, target, x, T, depth=depth if method == 'rt_partial' else None)
        acc[it] = float(a.double().mean())
        acc_trace[it] = a.cpu().numpy()
        states[it] = x.float().cpu().numpy()
        for k, v in observables(x, target, n).items():
            tr[k][it] = v.cpu().numpy()
        if ckpt is not None:
            m = it + 1
            ckpt.step(tag, m, lambda: dict(step=m, x=ck.to_cpu(x), W=ck.to_cpu(W), tr={k: tr[k][:m] for k in KEYS},
                                           acc=acc[:m], acc_trace=acc_trace[:m], states=states[:m],
                                           elapsed=time.time() - t0, extra=extra), force=(m == n_trans))
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    return tr, acc, time.time() - t0, states, acc_trace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--ckpt', default='kernel_best.pt')
    ap.add_argument('--refs', required=True)
    ap.add_argument('--methods', nargs='+', default=['snis', 'imh', 'rt'])
    ap.add_argument('--n-chains', type=int, default=256)
    ap.add_argument('--path-budget', type=int, default=256 * 600)
    ap.add_argument('--n-runs', type=int, default=3)
    ap.add_argument('--burn-fraction', type=float, default=0.2)
    ap.add_argument('--init', choices=['proposal', 'raw', 'val'], default='proposal',
                    help='proposal: forward-path endpoints with their own path records (default); raw: legacy '
                         'forward-endpoint start with reverse-path record; val: LEGACY reference validation split, '
                         'needs --allow-reference-init')
    ap.add_argument('--allow-reference-init', action='store_true',
                    help='permit --init val (chains start from reference states; not allowed for NHMC results)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--out', required=True)
    ck.add_args(ap)
    a = ap.parse_args()
    if a.init == 'val':
        if not a.allow_reference_init:
            raise SystemExit('--init val starts the chains from the reference validation split; NHMC chain results '
                             'must start from the sampler. Use --init proposal (or --allow-reference-init).')
        print('WARNING: --init val: chains start from reference states (legacy)', file=sys.stderr, flush=True)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    rd = pathlib.Path(a.run_dir)
    cfg = json.load(open(rd / 'config.json'))
    ev.CKPT = a.ckpt
    target, kernel, _ = ev.build_kernel(cfg, rd, dev)
    kernel.eval()
    T, n = cfg['num_steps'], cfg['n_particles']
    fe_path = T * (cfg['n_leapfrog'] + 1)
    val = (torch.tensor(np.load(os.path.join(a.refs, 'val_split_LJ13-1000.npy')), dtype=torch.float32, device=dev)
           if a.init == 'val' else None)
    test = torch.tensor(np.load(os.path.join(a.refs, 'test_split_LJ13-1000.npy')), dtype=torch.float32, device=dev)
    ref_obs = observables(test, target, n)
    ref = {k: (float(v.mean()), float(v.var())) for k, v in ref_obs.items()}
    ckpt = ck.Checkpointer(a.out, ck.args_hash(vars(a), files=[rd / 'config.json', rd / a.ckpt],
                                               code=ck.code_files(globals())), a.ckpt_min)
    result = dict(args=ck.public_args(a), force_evals_per_path=fe_path, reference=ref, methods={})
    outdir = pathlib.Path(a.out).with_suffix('')
    outdir.mkdir(parents=True, exist_ok=True)
    for mi, meth in enumerate(a.methods):
        name, depth = (meth.split(':')[0], int(meth.split(':')[1])) if ':' in meth else (meth, None)
        runs = []
        for r in range(a.n_runs):
            seed = a.seed + 1000 * r + 31 * len(meth)
            tag = meth.replace(':', '') + f'_run{r}'
            ctag = ck.Checkpointer.tag(meth, r)
            ckpt.exit_if_term()
            done = ckpt.load_done(ctag)
            if done is not None:
                s = done
            elif name == 'snis':
                X, Wt, t0 = run_snis(kernel, target, a.path_budget, T, seed, dev, ckpt=ckpt, tag=ctag)
                lw = (-Wt).numpy()
                s = dict(n_paths=a.path_budget, kish_ess=cs.kish_ess(lw), seconds=time.time() - t0)
                s['kish_ess_frac'] = s['kish_ess'] / a.path_budget
                for k, v in observables(X, target, n).items():
                    s[k] = dict(mean=cs.snis_mean(v.numpy(), lw), exact_mean=ref[k][0])
                np.savez_compressed(outdir / f'{tag}.npz', x=X.numpy(), logw=lw)
            else:
                ppt = 1.0 if name == 'imh' else (2.0 if name == 'rt' else 2.0 * depth / T)
                n_trans = max(int(a.path_budget / (a.n_chains * ppt)), 20)
                g = torch.Generator().manual_seed(seed)
                W0, init_info = None, {}
                resume = ckpt.load(ctag)
                if resume is not None:                  # start already drawn; the chain state is in resume
                    x0, init_info = None, resume['extra']
                elif a.init == 'proposal':
                    torch.manual_seed(seed + INIT_SEED_OFFSET)
                    xs, ws, nd, n_ok = [], [], 0, 0
                    for _ in range(64):               # drop non-finite paths (a NaN record would freeze a chain)
                        y0, w0 = fresh_path(kernel, target, a.n_chains, T, dev)
                        fin = torch.isfinite(w0) & torch.isfinite(y0).all(1)
                        nd += int((~fin).sum())
                        xs.append(y0[fin])
                        ws.append(w0[fin])
                        n_ok += int(fin.sum())
                        if n_ok >= a.n_chains:
                            break
                    if n_ok < a.n_chains:
                        raise SystemExit('proposal start: not enough finite forward paths')
                    x0, W0 = torch.cat(xs)[:a.n_chains].float(), torch.cat(ws)[:a.n_chains]
                    init_info = dict(n_dropped_nonfinite=nd)
                elif a.init == 'raw':
                    torch.manual_seed(seed)
                    x0 = fresh_path(kernel, target, a.n_chains, T, dev)[0].float()
                else:
                    x0 = val[torch.randperm(val.shape[0], generator=g)[:a.n_chains].to(dev)]
                tr, acc, sec, xf, acct = run_chain(name, depth, kernel, target, x0, n_trans, T, n, seed, dev, W0=W0,
                                                   ckpt=ckpt, tag=ctag, resume=resume, extra=init_info)
                burn = int(a.burn_fraction * n_trans)
                cost = dict(paths=ppt, force_evals=ppt * fe_path, seconds=sec / n_trans)
                s = dict(acceptance=float(acc[burn:].mean()), seconds=sec, n_transitions=n_trans, burn=burn,
                         paths_per_transition=ppt, holding=cs.holding_times(acct[burn:]), init=init_info,
                         burnin_check=cs.burnin_check(tr, burn, early_keys=('energy', 'rg')))
                print(f'[{meth} run {r}] ' + cs.burnin_line(s['burnin_check']), flush=True)
                if a.init != 'val':
                    s['relaxation'] = {k: cs.relaxation_curve(tr[k]) for k in KEYS}
                for k in KEYS:
                    s[k] = cs.observable_summary(tr[k][burn:], exact_mean=ref[k][0], exact_var=ref[k][1],
                                                 cost_per_transition=cost)
                np.savez_compressed(outdir / f'{tag}_traces.npz', **{k: tr[k] for k in KEYS}, acc=acc)
                post = xf[burn:]                                  # (T', C, D) chain states after burn-in
                np.savez_compressed(outdir / f'{tag}_chain.npz', x=post.transpose(1, 0, 2).reshape(-1, post.shape[-1]),
                                    chain_shape=np.array([post.shape[1], post.shape[0]]))
            if done is None:
                ckpt.done(ctag, s)
            runs.append(s)
            print(f'[{meth} run {r}] ' + json.dumps(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if not isinstance(v, dict)}
                | {k: round(s[k].get('tau_int', float('nan')), 2) for k in KEYS}
                | {k + '_mean': round(s[k]['mean'], 4) for k in KEYS}), flush=True)
        agg = {k: cs.between_runs([rr[k]['mean'] for rr in runs], exact_var=ref[k][1],
                                  cost_per_run=dict(paths=a.path_budget,
                                                    seconds=float(np.mean([rr['seconds'] for rr in runs]))))
               for k in KEYS}
        result['methods'][meth] = dict(runs=runs, between_runs=agg)
        # with methods still to come this is a progress copy (a task guard [ -f out ] must not see a partial json)
        ck.atomic_json(result, a.out if mi == len(a.methods) - 1 else a.out + '.partial', indent=1, default=float)
    if os.path.exists(a.out + '.partial'):
        os.remove(a.out + '.partial')
    ckpt.finish()
    print('wrote', a.out)


if __name__ == '__main__':
    main()
