#!/usr/bin/env python3
"""Timing driver for the phi4 NHMC samplers (timing table of Appendix B).

Loads the frozen model with exp_chains.PhiStepper and runs, for each method, exactly the per-transition code of the
exp_chains.py step loop (propose, log-ratio, accept, observables of x and y, waste-recycling record, observables of
the new state, numpy copies). n_chains chains are one batched tensor, as in exp_chains.py; one 'update' = one
transition of all chains. Warm-up updates are not timed. Also times forward-path generation (PhiStepper.start, the
path that SNIS and the path-IMH proposals use) at batch n_chains and at batch --fwd-batch.
Timing only: no chain statistics are written.
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

import exp_chains as LC  # noqa: E402


def one_update(st, name, s, x, W):
    y, Wx, Wy = st.propose(name, s, x, W)
    la = torch.nan_to_num((Wx - Wy).double(), nan=-math.inf, posinf=math.inf, neginf=-math.inf)
    ac = torch.log(torch.rand_like(la)) < la
    al = torch.exp(torch.clamp(la, max=0.0))
    fx, fy = st.obs(x), st.obs(y)
    rb = torch.stack([al * fy_.double() + (1 - al) * fx_.double() for fx_, fy_ in zip(fx, fy)], 1).cpu().numpy()
    x = torch.where(ac.view(-1, *([1] * (x.dim() - 1))), y, x)
    if name == 'imh':
        W = torch.where(ac, Wy, W)
    M_, S_, A_ = st.obs(x)
    _ = (M_.cpu().numpy(), S_.cpu().numpy(), A_.cpu().numpy(), ac.cpu().numpy(), rb)
    return x, W, ac


def stats(t):
    t = np.asarray(t)
    return dict(n=int(t.size), median=float(np.median(t)), mean=float(t.mean()), sd=float(t.std(ddof=1)) if t.size > 1 else 0.0,
                q10=float(np.quantile(t, .1)), q90=float(np.quantile(t, .9)), total=float(t.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--kappa', type=float, default=0.27)
    ap.add_argument('--s', type=int, nargs='+', required=True,
                    help='pRT stages s (1 <= s <= T-1); s = 3T/4 (d = T/4) and s = T-4 (d = 4)')
    ap.add_argument('--n-chains', type=int, default=4)
    ap.add_argument('--warm', type=int, default=20)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--fwd-batch', type=int, default=1024)
    ap.add_argument('--fwd-reps', type=int, default=5)
    ap.add_argument('--fwd-small-reps', type=int, default=50)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261118)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    st = LC.PhiStepper(a.run_dir, a.ckpt, a.kappa, dev)
    T, L, nlf = st.T, st.L, st.k.base.n_leapfrog
    res = dict(args=vars(a), T=T, L=L, n_leapfrog=nlf, model_id=st.id, torch=torch.__version__,
               torch_threads=torch.get_num_threads(), host=platform.node(),
               affinity=sorted(os.sched_getaffinity(0)), methods={},
               definition='one update = one transition of all n_chains chains (batched), longchain.py step-loop code')
    with torch.no_grad():
        for m in ['imh', 'rt'] + [f'rtp{s_}' for s_ in dict.fromkeys(a.s) if 1 <= s_ <= T - 1]:
            name, s = (m, 0) if m in ('imh', 'rt') else ('rtp', int(m[3:]))
            torch.manual_seed(a.seed)
            x, W = st.start(a.n_chains)
            ts, nacc = [], 0
            for i in range(a.warm + a.n):
                t0 = time.perf_counter()
                x, W, ac = one_update(st, name, s, x, W)
                dt = time.perf_counter() - t0
                if i >= a.warm:
                    ts.append(dt)
                    nacc += int(ac.sum())
            key = m
            res['methods'][key] = dict(s=s, depth=T - s, sec_per_update=stats(ts), accepted=nacc,
                                       grad_S_evals_per_chain=(nlf + 1) * (T if name == 'imh' else 2 * (T - s)))
            print(key, json.dumps(res['methods'][key]['sec_per_update']), flush=True)
        # forward paths (SNIS / path-IMH proposal): PhiStepper.start = base draw + T forward stages + action
        torch.manual_seed(a.seed + 1)
        for _ in range(3):
            st.start(a.n_chains)
        ts = []
        for _ in range(a.fwd_small_reps):
            t0 = time.perf_counter(); st.start(a.n_chains); ts.append(time.perf_counter() - t0)
        res['fwd_batch_small'] = dict(batch=a.n_chains, sec_per_call=stats(ts),
                                      sec_per_path_median=float(np.median(ts)) / a.n_chains)
        st.start(a.fwd_batch)
        ts = []
        for _ in range(a.fwd_reps):
            t0 = time.perf_counter(); st.start(a.fwd_batch); ts.append(time.perf_counter() - t0)
        res['fwd_batch_large'] = dict(batch=a.fwd_batch, sec_per_call=stats(ts),
                                      sec_per_path_median=float(np.median(ts)) / a.fwd_batch)
        print('fwd', json.dumps(res['fwd_batch_small']), json.dumps(res['fwd_batch_large']), flush=True)
    tmp = a.out + '.tmp'
    json.dump(res, open(tmp, 'w'), indent=1)
    os.replace(tmp, a.out)


if __name__ == '__main__':
    main()
