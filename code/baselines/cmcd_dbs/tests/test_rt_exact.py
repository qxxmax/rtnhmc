#!/usr/bin/env python3
"""Exactness of the path-space chains of rt_cmcd_phi4.py on L = 2 phi^4 (kappa = .30), vs exact quadrature.

Models: CMCD+Z2 and DBS+Z2 of the torch port (T = 8), randomly perturbed then briefly trained with the port's own
loss (test_weight_identity.make), i.e. deliberately imperfect samplers with usable acceptance.
Chains (1024 parallel, exact start from an HMC+Z2 bank; burn-in 20 %): imh, rt (s = 0), partial RT s = 4 and
s = 7 (= T - 1); additionally rt from a proposal start (forward endpoints, burn-in 50 %).
Statistics: |M|, M^2, M^4, S/V, sign(M); SE from the spread of per-chain time means (chains independent);
z = (mean - exact) / SE. Also tau-corrected z from chainstats (pooled ACF).
NEGATIVE CONTROL: reverse draws with eta_{k+1} instead of eta_k (W keeps the correct densities) -> wrong augmented
target -> must show a significant bias.
Also a non-Z2 CMCD model: imh and rt with the exact global flip (--z2flip; IMH refreshes its path record for
flipped chains), IMH from a proposal start; plus the negative control 'flip but keep the IMH record' (reported).
Pass: max |z| < 4.5 over all exact chains and observables, and the eta_(k+1) negative controls |z| > 6.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import rt_cmcd_phi4 as R  # noqa: E402
import l2_exact as X  # noqa: E402
import test_weight_identity as WI  # noqa: E402

torch.set_num_threads(int(os.environ.get('NTHREADS', 4)))
KAPPA, T, B = WI.KAPPA, WI.T, 1024
OBS = ('absM', 'M2', 'M4', 'action', 'sign')


def stats(tr, ref, burn):
    out = {}
    for k in OBS:
        x = tr[k][burn:]
        pc = x.mean(0)
        mean, se = float(pc.mean()), float(pc.std(ddof=1) / math.sqrt(pc.size))
        tau = R.cs.tau_int(x)[0]
        se_tau = math.sqrt(x.var() * tau / x.size)
        out[k] = dict(mean=mean, se=se, exact=ref[k], z=(mean - ref[k]) / se if se > 0 else float('nan'),
                      tau=tau, z_tau=(mean - ref[k]) / se_tau if se_tau > 0 else float('nan'))
    out['max_abs_z'] = max(abs(out[k]['z']) for k in OBS)
    return out


def line(name, st, acc):
    return (f'{name:34s} acc={acc:.3f} max|z|={st["max_abs_z"]:.2f} | ' +
            ' '.join(f'{k}:{st[k]["mean"]:.4f}({st[k]["se"]:.4f}) z={st[k]["z"]:+.2f}' for k in OBS))


def main():
    t_all = time.time()
    ref = X.exact_4d(KAPPA)
    res = dict(kappa=KAPPA, T=T, n_chains=B, exact=ref, models={})
    print('exact:', {k: round(v, 5) for k, v in ref.items()})
    ok = True
    worst = 0.0
    for alg, seed in (('cmcd', 7), ('dbs', 8)):
        m = WI.make(alg, True, seed)
        mres = {}
        bank = R.hmc_bank(m, B, 400, 0.4, 6, seed=11, device='cpu', dtype=torch.float64, flip=True)
        # sanity of the bank itself: a further 400 HMC+Z2 trajectories as a chain
        for name, meth, s, n_tr in (('imh', 'imh', None, 600), ('rt', 'rt', None, 300), ('rt_partial:4', 'rtp', 4, 600),
                                    ('rt_partial:7', 'rtp', 7, 1200)):
            t0 = time.time()
            tr, acc, sec, _ = R.run_chain(m, meth, s, bank, n_tr, seed=100 + (s or 0), with_m4=True)
            st = stats(tr, ref, n_tr // 5)
            st['acceptance'] = float(acc.mean()); st['seconds'] = time.time() - t0
            mres[name] = st
            worst = max(worst, st['max_abs_z'])
            print(line(f'{alg}+Z2 {name} exact-start', st, st['acceptance']) + f' [{time.time() - t0:.0f}s]', flush=True)
        # proposal start (relaxation must also converge to the exact law)
        torch.manual_seed(500)
        c = m.coeffs()
        x0 = m.forward_from(m.prior_sample(B, c, 'cpu', torch.float64), 0, c)[0]
        tr, acc, sec, _ = R.run_chain(m, 'rt', None, x0, 400, seed=501, with_m4=True)
        st = stats(tr, ref, 200)
        st['acceptance'] = float(acc.mean())
        mres['rt_proposal_start'] = st
        worst = max(worst, st['max_abs_z'])
        print(line(f'{alg}+Z2 rt proposal-start', st, st['acceptance']), flush=True)
        # negative control
        tr, acc, sec, _ = R.run_chain(m, 'rt', None, bank, 300, seed=777, bug='eta_next', with_m4=True)
        st = stats(tr, ref, 60)
        st['acceptance'] = float(acc.mean())
        mres['NEGATIVE_rt_eta_next'] = st
        print(line(f'{alg}+Z2 NEGATIVE rt eta_(k+1)', st, st['acceptance']), flush=True)
        ok &= st['max_abs_z'] > 6
        res['models'][f'{alg}+Z2'] = mres
    # non-Z2 model: z2flip (IMH needs the record refresh), IMH from a proposal start, and the negative control
    m = WI.make('cmcd', False, 7)
    mres = {}
    bank = R.hmc_bank(m, B, 400, 0.4, 6, seed=12, device='cpu', dtype=torch.float64, flip=True)
    torch.manual_seed(600)
    c = m.coeffs()
    xprop = m.forward_from(m.prior_sample(B, c, 'cpu', torch.float64), 0, c)[0]
    for name, meth, x0, n_tr, kw, burn in (('imh+z2flip', 'imh', bank, 600, dict(z2flip=True), 120),
                                           ('rt+z2flip', 'rt', bank, 300, dict(z2flip=True), 60),
                                           ('imh proposal-start', 'imh', xprop, 800, {}, 400),
                                           ('NEGATIVE imh+z2flip keep W', 'imh', bank, 600,
                                            dict(z2flip=True, bug='imh_flip_keepW'), 120)):
        tr, acc, sec, _ = R.run_chain(m, meth, None, x0, n_tr, seed=900 + n_tr, with_m4=True, **kw)
        st = stats(tr, ref, burn)
        st['acceptance'] = float(acc.mean())
        mres[name] = st
        print(line(f'cmcd(no Z2) {name}', st, st['acceptance']), flush=True)
        if name.startswith('NEGATIVE'):
            res['negative_imh_flip_keepW_max_abs_z'] = st['max_abs_z']
        else:
            worst = max(worst, st['max_abs_z'])
    res['models']['cmcd(noZ2)'] = mres
    ok &= worst < 4.5
    res['max_abs_z_exact_chains'] = worst
    res['PASS'] = bool(ok)
    res['seconds'] = time.time() - t_all
    json.dump(res, open(os.path.join(HERE, 'test_rt_exact.json'), 'w'), indent=1, default=float)
    print(f'max |z| over exact chains {worst:.2f};', 'PASS' if ok else 'FAIL', f'[{time.time() - t_all:.0f}s]')


if __name__ == '__main__':
    main()
