#!/usr/bin/env python3
"""Kernel log-densities of the torch port vs a literal numpy transcription of the official JAX formulas.

The numpy side re-implements, without importing the port: targets/phi4.Phi4Distr.log_prob, distrax
MultivariateNormalDiag (prior), init_langevin.langevin_fn (betas cumsum/sum, clamped gather b[int(step)]),
init_underdamped.delta_t_fn / friction_fn with scheduler.get_cosine_schedule, PISGRADNet.__call__ (flax gelu =
tanh approximation, use_target_score False), cmcd/dbs forward_model / backward_model, the Z2PISGRADNet wrapper of
udb_bench.enable_z2, od_integrators.integrate_EM (fwd_mean, bwd_mean, log_prob_kernel) and the terminal/running
costs of overdamped_rnd.per_sample_rnd. Scores are taken by central finite differences of the transcribed log
densities (independent of the analytic force used by the port).

Inputs: the JAX-trained udb_bench parameters (params.msgpack, main_L8 runs; CMCD and DBS), random paths drawn by
the port. Checks (float64): per-step log F_k, log B_k; log w of the sampler == log P - log Q; action formula.
Also runs a random-parameter Z2 model (odd nets) through the same comparison.
"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import cmcd_phi4 as C  # noqa: E402

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
PDIR = os.path.join(os.path.dirname(HERE), 'jax_params/release_20261007/results/main_L8')


# ------------------------------------------------------------------ numpy transcription of the official code
def np_softplus(x):
    return np.logaddexp(0.0, x)


def np_gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def np_dense(q, x):
    return x @ q['kernel'] + q['bias']


def np_pisgrad(q, x, t):
    q = q['params']
    num_hid = q['timestep_phase'].shape[1]
    coeff = np.linspace(0.1, 100, num_hid)[None]
    ff = np.concatenate([np.sin(coeff * t + q['timestep_phase']), np.cos(coeff * t + q['timestep_phase'])], -1)[0]
    tc = q['time_coder_state']
    t1 = np_dense(tc['layers_2'], np_gelu(np_dense(tc['layers_0'], ff)))
    h = np.concatenate([x, t1], -1)
    st = q['state_time_net']
    h = np_gelu(np_dense(st['layers_0']['layers_0'], h))
    h = np_gelu(np_dense(st['layers_1']['layers_0'], h))
    return np.clip(np_dense(st['layers_2'], h), -1e4, 1e4)


def np_net(q, x, t, z2):
    if not z2:
        return np_pisgrad(q, x, t)
    return 0.5 * (np_pisgrad(q, x, t) - np_pisgrad(q, -x, t))


def np_logprob_phi4(x, kappa, lambd=0.022):
    L = int(round(math.sqrt(x.size)))
    x = x.reshape((-1, L, L))
    kinetic = (-2 * kappa) * x * (np.roll(x, 1, axis=-1) + np.roll(x, 1, axis=-2))
    mass = (1 - 2 * lambd) * x ** 2
    inter = lambd * x ** 4
    return -np.sum(kinetic + mass + inter)


def np_prior_logprob(x, P):
    mu, s = P['prior_mean'], np_softplus(P['prior_std'])
    return float(np.sum(-0.5 * ((x - mu) / s) ** 2 - np.log(s) - 0.5 * math.log(2 * math.pi)))


def fd_grad(f, x, h=1e-5):
    g = np.empty_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def np_log_prob_kernel(x, mean, scale):
    return float(np.sum(-0.5 * ((x - mean) / scale) ** 2 - np.log(scale) - 0.5 * math.log(2 * math.pi)))


def np_cosine(step, T, mn=0.01, s=0.008, pw=2):
    t = (T - step) / T
    off = 1 + s
    return (1. - mn) * np.cos(0.5 * np.pi * (off - t) / off) ** pw + mn


def np_step_logs(P, alg, z2, path, kappa, T):
    """integrate_EM evaluated along a given path: returns per-step (fwd_log_prob, bwd_log_prob), terminal cost."""
    b = np_softplus(P['betas']); b = np.cumsum(b) / np.sum(b)
    get_beta = lambda step: b[min(int(step), T - 1)]            # jnp gather clamps out-of-bound indices
    lg = lambda x: np_logprob_phi4(x, kappa)
    lp = lambda x: np_prior_logprob(x, P)

    def langevin(step, x):
        beta = get_beta(step)
        return beta * fd_grad(lg, x) + (1 - beta) * fd_grad(lp, x)

    fwd = lambda step, x: np_net(P['fwd_params'], x, step, z2)
    if alg == 'cmcd':
        bwd = lambda step, x: -np_net(P['fwd_params'], x, step, z2)
    else:
        bwd = lambda step, x: np_net(P['bwd_params'], x, step, z2)
    LF, LB = [], []
    for k in range(T):
        step = float(k)
        x, xn = path[k], path[k + 1]
        dt = np_softplus(P['dt'])[0] * np_cosine(step, T)
        eta = dt * (1. / np_softplus(P['friction']))
        scale = np.sqrt(2 * eta)
        fwd_mean = eta * langevin(step, x) + eta * fwd(step, x)
        bwd_mean = eta * (langevin(step + 1, xn) + bwd(step + 1, xn))
        d = xn - x
        LF.append(np_log_prob_kernel(d, fwd_mean, scale))
        LB.append(np_log_prob_kernel(-d, bwd_mean, scale))
    terminal = lp(path[0]) - lg(path[-1])
    return np.array(LF), np.array(LB), terminal


def torch_params_as_flax(m):
    """flax-style dict from a torch model (for the random-parameter Z2 check)."""
    def net(n):
        t = lambda l: dict(kernel=l.weight.detach().numpy().T.copy(), bias=l.bias.detach().numpy().copy())
        return dict(params=dict(timestep_phase=n.timestep_phase.detach().numpy().copy(),
                                time_coder_state=dict(layers_0=t(n.tc0), layers_2=t(n.tc2)),
                                state_time_net=dict(layers_0=dict(layers_0=t(n.st0)), layers_1=dict(layers_0=t(n.st1)),
                                                    layers_2=t(n.out))))
    P = dict(betas=m.betas.detach().numpy(), prior_mean=m.prior_mean.detach().numpy(),
             prior_std=m.prior_std.detach().numpy(), dt=m.dt.detach().numpy(), friction=m.friction.detach().numpy(),
             fwd_params=net(m.fwd))
    if m.bwd is not None:
        P['bwd_params'] = net(m.bwd)
    return P


def compare(m, P, alg, z2, kappa, n_paths=3, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        lw, xT, path = m.sample_paths(n_paths, 'cpu', torch.float64, keep_path=True)
        LFt, LBt = m.per_step_logs(path)
        lq, lp = m.path_logs(path)
    errs = dict(step_F=0.0, step_B=0.0, logw=0.0, logw_vs_paths=0.0)
    for i in range(n_paths):
        pth = path[:, i].numpy()
        LF, LB, term = np_step_logs(P, alg, z2, pth, kappa, m.T)
        errs['step_F'] = max(errs['step_F'], float(np.abs(LF - LFt[:, i].numpy()).max()))
        errs['step_B'] = max(errs['step_B'], float(np.abs(LB - LBt[:, i].numpy()).max()))
        logw_np = -(term - np.sum(LB - LF))                       # -(running + terminal) of per_sample_rnd
        errs['logw'] = max(errs['logw'], abs(logw_np - float(lw[i])))
        errs['logw_vs_paths'] = max(errs['logw_vs_paths'], abs(float(lp[i] - lq[i]) - float(lw[i])))
    errs['scale_logF'] = float(LFt.abs().mean())
    return errs


def main():
    res = {}
    ok = True
    # action formula: official targets/phi4.py vs repo action used by the port
    x = np.random.default_rng(0).normal(size=(5, 64)) * 1.5
    m0 = C.UDBPhi4('cmcd', 8, 0.27, z2=False)
    da = max(abs(np_logprob_phi4(x[i], 0.27) - float(m0.log_gamma(torch.as_tensor(x[i:i + 1]))[0])) for i in range(5))
    res['action_max_abs_diff'] = da
    ok &= da < 1e-10
    for run, alg, kappa in (('cmcd_k0.27_s0', 'cmcd', 0.27), ('dbs_k0.30_s0', 'dbs', 0.30), ('dbs_k0.27_s0', 'dbs', 0.27)):
        P = C.load_msgpack(os.path.join(PDIR, run, 'params.msgpack'))
        m = C.UDBPhi4(alg, 8, kappa, z2=False)
        m.load_flax(P)
        m = m.double()
        Pn = {k: (np.asarray(v, np.float64) if not isinstance(v, dict) else v) for k, v in P['params'].items()}
        for k in ('fwd_params', 'bwd_params'):
            if k in Pn:
                Pn[k] = __import__('copy').deepcopy(Pn[k])
                def cast(d):
                    return {kk: (cast(vv) if isinstance(vv, dict) else np.asarray(vv, np.float64)) for kk, vv in d.items()}
                Pn[k] = cast(Pn[k])
        e = compare(m, Pn, alg, False, kappa)
        res[run] = e
        ok &= e['step_F'] < 1e-6 and e['step_B'] < 1e-6 and e['logw'] < 1e-6 and e['logw_vs_paths'] < 1e-9
        print(run, json.dumps(e), flush=True)
    # random-parameter Z2 models (odd nets, prior mean 0), larger network outputs
    for alg in ('cmcd', 'dbs'):
        torch.manual_seed(5)
        m = C.UDBPhi4(alg, 4, 0.3, T=6, z2=True).double()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.3 * torch.randn_like(p))
            for net in (m.fwd, m.bwd):
                if net is not None:
                    net.out.weight.normal_(0, 0.3)
        P = torch_params_as_flax(m)
        e = compare(m, P, alg, True, 0.3, n_paths=4)
        res[f'random_z2_{alg}'] = e
        ok &= e['step_F'] < 1e-6 and e['step_B'] < 1e-6 and e['logw'] < 1e-6
        print(f'random_z2_{alg}', json.dumps(e), flush=True)
    res['PASS'] = bool(ok)
    json.dump(res, open(os.path.join(HERE, 'test_jax_formulas.json'), 'w'), indent=1)
    print('PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
