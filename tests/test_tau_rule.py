#!/usr/bin/env python3
"""Checks of the autocorrelation-time rule (code/tau.py) on synthetic chains with known answers.

  * AR(1) x_t = phi x_{t-1} + sqrt(1 - phi^2) e_t has g = 1 + 2 sum_t rho(t) = (1 + phi) / (1 - phi), tau_int = g / 2;
    sokal_single and batch_means_tau recover g, tau_int_chains recovers g / 2 (median over chains).
  * holding_runs: lengths of maximal runs of rejections on a hand-made accept sequence.
  * chain_criteria: a short AR(1) chain fails n_post >= 1000 g, a long one passes; a frozen sign fails.
  * jk_delete_one of the mean reproduces the textbook standard error s / sqrt(n).
Run: python tests/test_tau_rule.py (exit code 0 = PASS), or pytest. Seconds on one CPU core.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'code'))
from tau import (batch_means_tau, chain_criteria, holding_runs, jk_delete_one, sokal_single,  # noqa: E402
                 tau_int_chains)


def ar1(n, phi, chains=1, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, chains))
    x = np.empty((n, chains))
    x[0] = e[0]
    c = math.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + c * e[t]
    return x


def test_sokal_and_batch_means():
    for phi in (0.5, 0.9):
        g = (1 + phi) / (1 - phi)
        x = ar1(400_000, phi, seed=1)[:, 0]
        gs, win = sokal_single(x)
        gb = batch_means_tau(x)
        print(f'phi {phi}: exact g {g:.3f}  Sokal {gs:.3f} (window {win})  batch means {gb:.3f}')
        assert abs(gs / g - 1) < 0.05
        assert abs(gb / g - 1) < 0.15
        assert win >= 5 * gs


def test_tau_int_chains():
    phi = 0.8
    g = (1 + phi) / (1 - phi)
    x = ar1(100_000, phi, chains=4, seed=2)
    med, err, per = tau_int_chains(x)
    print(f'tau_int median {med:.3f} +- {err:.3f} (exact {g / 2:.3f}); per chain {np.round(per, 3)}')
    assert abs(med / (g / 2) - 1) < 0.1


def test_holding_runs():
    acc = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0], bool)
    assert holding_runs(acc).tolist() == [2, 3, 1]
    assert holding_runs(np.ones(5, bool)).tolist() == [0]


def test_chain_criteria():
    phi = 0.9
    short = ar1(5_000, phi, chains=2, seed=3)
    long_ = ar1(60_000, phi, chains=2, seed=4)
    for x, want in ((short, False), (long_, True)):
        n = x.shape[0]
        sign = np.where(np.arange(n)[:, None] % 2 == 0, 1, -1).repeat(x.shape[1], 1).astype(np.int8)
        acc = np.ones((n, x.shape[1]), bool)
        ok, per = chain_criteria(np.abs(x), sign, acc, n, 0.1, 1000.0, 20.0, 50.0)
        print(f'n {n}: ok {ok}; ' + '; '.join(f'g {d["tau_max"]:.1f} fail {d["failing"]}' for d in per))
        assert ok == want
    frozen = np.ones_like(long_, dtype=np.int8)
    ok, per = chain_criteria(np.abs(long_), frozen, np.ones(long_.shape, bool), long_.shape[0], 0.1, 1000.0, 20.0, 50.0)
    assert not ok and all('sign_frozen' in d['failing'] for d in per)


def test_jackknife_mean():
    x = np.random.default_rng(5).standard_normal(200)
    full, err, _ = jk_delete_one(x, np.mean)
    assert abs(full - x.mean()) < 1e-12
    assert abs(err - x.std(ddof=1) / math.sqrt(x.size)) < 1e-10


if __name__ == '__main__':
    ok = True
    for f in (test_sokal_and_batch_means, test_tau_int_chains, test_holding_runs, test_chain_criteria,
              test_jackknife_mean):
        try:
            f()
            print('PASS', f.__name__, flush=True)
        except AssertionError as e:
            ok = False
            print('FAIL', f.__name__, e, flush=True)
    sys.exit(0 if ok else 1)
