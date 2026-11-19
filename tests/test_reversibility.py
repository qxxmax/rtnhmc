#!/usr/bin/env python3
"""Exactness of the three path updates on the L = 2 phi^4 lattice (kappa = 0.30, lambda = 0.022), against quadrature.

The round-trip and path-IMH chains leave the target invariant for any kernel parameters, so a deliberately poor model
(a randomly initialised nhmc_resid2 kernel with odd nets and the Z2 wrapper, T = 8 stages, as in the paper but small)
must reproduce the exact moments within statistical errors. Exact values: l2_exact.exact_4d (4-D trapezoid quadrature).

  1. identity: updates.py reproduces the transition of exp_chains.py (PhiStepper.propose + accept) bit for bit;
  2. SNIS: log Z from forward-path works W (E_Q[e^{-W}] = Z) agrees with the exact log Z;
  3. chains: path-IMH, RT and pRT (s = 4) from forward-path starts; <|M|>, <M^2>, <S>/V agree with the exact values
     (SE from the spread of the per-chain time means of independent chains; pass: max |z| < 4.5);
  4. negative control: the same proposals accepted without the Metropolis test must be biased (|z| > 6 for some
     observable), so the test can fail.

Run on a CPU node: python tests/test_reversibility.py  (exit code 0 = PASS). Also collected by pytest.
"""
import json
import math
import os
import pathlib
import sys
import tempfile

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))
sys.path.insert(0, str(HERE))
import exp_chains as LC  # noqa: E402
import kernels  # noqa: E402
import paths  # noqa: E402
import updates  # noqa: E402
from phi4 import phi4_action  # noqa: E402
import l2_exact as X  # noqa: E402

KAPPA, LAM, L, T = 0.30, 0.022, 2, 8
CFG = dict(kernel='nhmc_resid2', embed_dim=32, channels=[8, 8], n_leapfrog=3, lam=LAM, num_steps=T, L=L,
           odd_nets=True, use_z2=True)
C, N_TRANS = int(os.environ.get('TEST_CHAINS', 2048)), int(os.environ.get('TEST_STEPS', 800))
Z_PASS, Z_NEG = 4.5, 6.0
torch.set_num_threads(int(os.environ.get('NTHREADS', 4)))


def make_model(seed=0):
    """Random kernel saved as a run directory (config.json + kernel_final.pt), loaded through kernels.load_model."""
    torch.manual_seed(seed)
    k = kernels.create_cond_kernel(CFG['kernel'], embed_dim=CFG['embed_dim'], channels=tuple(CFG['channels']),
                                   n_leapfrog=CFG['n_leapfrog'], lam=LAM, num_steps=T, use_z2=True, odd_nets=True)
    with torch.no_grad():                       # move away from the zero-initialised residual force and default steps
        for p in k.parameters():
            p.add_(0.05 * torch.randn_like(p))
    d = pathlib.Path(tempfile.mkdtemp())
    (d / 'config.json').write_text(json.dumps(CFG))
    torch.save(k.state_dict(), d / 'kernel_final.pt')
    km, _ = kernels.load_model(str(d), 'kernel_final.pt', torch.device('cpu'))
    return km, d


def observables(x):
    M = x.mean(dim=(1, 2, 3))
    return dict(absM=M.abs(), M2=M ** 2, action=phi4_action(x.squeeze(1), lam=LAM, kappa=KAPPA) / x[0].numel())


def zstats(tr, ref, burn):
    out = {}
    for key, v in tr.items():
        pc = np.asarray(v[burn:]).mean(0)                      # per-chain time means
        m, se = float(pc.mean()), float(pc.std(ddof=1) / math.sqrt(pc.size))
        out[key] = dict(mean=m, se=se, exact=ref[key], z=(m - ref[key]) / se)
    return out


def chain(k, method, s, n, seed, mh=True):
    """C independent chains from forward-path starts; returns traces (n, C) and the mean acceptance."""
    torch.manual_seed(seed)
    x, W = updates.start_states(k, C, L, L, T, KAPPA, LAM, 'cpu')
    tr = {key: np.empty((n, C)) for key in ('absM', 'M2', 'action')}
    acc = np.empty((n, C), bool)
    with torch.no_grad():
        for it in range(n):
            if not mh:                                         # negative control: accept every round-trip proposal
                u, _ = paths.rev_partial(k, x, T, s, KAPPA, LAM)
                x, _ = paths.fwd_partial(k, u, T, s, KAPPA, LAM)
                a = torch.ones(C, dtype=torch.bool)
            elif method == 'imh':
                x, W, a, _ = updates.path_imh_step(k, x, W, T, KAPPA, LAM)
            else:
                x, a, _, _, _ = updates.truncated_round_trip_step(k, x, T, s, KAPPA, LAM)
            for key, v in observables(x).items():
                tr[key][it] = v.numpy()
            acc[it] = a.numpy()
    return tr, float(acc.mean())


def test_identity_with_exp_chains():
    """updates.py transitions == exp_chains.PhiStepper transitions (same seed, same states, bitwise)."""
    _, d = make_model()
    st = LC.PhiStepper(str(d), 'kernel_final.pt', KAPPA, torch.device('cpu'))
    k = st.k
    for name, s in (('imh', 0), ('rt', 0), ('rtp', 4)):
        torch.manual_seed(11)
        x0, W0 = st.start(64)
        torch.manual_seed(12)
        x, W = x0.clone(), W0.clone()
        for _ in range(5):                                     # exp_chains main loop
            y, Wx, Wy = st.propose(name, s, x, W)
            la = torch.nan_to_num((Wx - Wy).double(), nan=-math.inf, posinf=math.inf, neginf=-math.inf)
            ac = torch.log(torch.rand_like(la)) < la
            x = torch.where(ac.view(-1, *([1] * (x.dim() - 1))), y, x)
            if name == 'imh':
                W = torch.where(ac, Wy, W)
        torch.manual_seed(12)
        x2, W2 = x0.clone(), W0.clone()
        for _ in range(5):
            if name == 'imh':
                x2, W2, _, _ = updates.path_imh_step(k, x2, W2, T, KAPPA, LAM)
            else:
                x2, _, _, _, _ = updates.truncated_round_trip_step(k, x2, T, s, KAPPA, LAM)
        assert torch.equal(x, x2), f'{name}: updates.py differs from exp_chains.py'
        if name == 'imh':
            assert torch.equal(W, W2)


def test_snis_logz():
    k, _ = make_model()
    ref = X.exact_4d(KAPPA)
    torch.manual_seed(3)
    with torch.no_grad():
        W = torch.cat([updates.start_states(k, 16384, L, L, T, KAPPA, LAM, 'cpu')[1] for _ in range(8)]).double()
    lw = -W
    logZ = float(torch.logsumexp(lw, 0) - math.log(lw.numel()))
    wr = torch.exp(lw - lw.max())
    se = float(wr.std() / wr.mean() / math.sqrt(lw.numel()))
    z = (logZ - ref['logZ']) / se
    print(f'SNIS log Z {logZ:.4f} +- {se:.4f}, exact {ref["logZ"]:.4f}, z {z:+.2f}', flush=True)
    assert abs(z) < Z_PASS


def test_chains_exact():
    k, _ = make_model()
    ref = X.exact_4d(KAPPA)
    worst = 0.0
    for name, method, s, seed in (('path-IMH', 'imh', 0, 21), ('RT', 'rt', 0, 22), ('pRT s=4', 'rtp', 4, 23)):
        tr, a = chain(k, method, s, N_TRANS, seed)
        st = zstats(tr, ref, N_TRANS // 4)
        zmax = max(abs(v['z']) for v in st.values())
        worst = max(worst, zmax)
        print(f'{name:9s} acc {a:.3f} max|z| {zmax:.2f} | ' + ' '.join(
            f'{key} {v["mean"]:.4f}({v["se"]:.4f}) exact {v["exact"]:.4f}' for key, v in st.items()), flush=True)
    assert worst < Z_PASS, f'max |z| = {worst:.2f}'


def test_negative_control():
    k, _ = make_model()
    ref = X.exact_4d(KAPPA)
    tr, _ = chain(k, 'rt', 0, N_TRANS, 31, mh=False)
    st = zstats(tr, ref, N_TRANS // 4)
    zmax = max(abs(v['z']) for v in st.values())
    print(f'no-MH control max|z| {zmax:.2f} (must exceed {Z_NEG})', flush=True)
    assert zmax > Z_NEG


if __name__ == '__main__':
    ok = True
    for f in (test_identity_with_exp_chains, test_snis_logz, test_chains_exact, test_negative_control):
        try:
            f()
            print('PASS', f.__name__, flush=True)
        except AssertionError as e:
            ok = False
            print('FAIL', f.__name__, e, flush=True)
    sys.exit(0 if ok else 1)
