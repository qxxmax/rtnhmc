#!/usr/bin/env python3
"""PyTorch port of the OFFICIAL overdamped CMCD / DBS samplers (Underdamped Diffusion Bridges repo, Blessing et al.,
github.com/DenisBless/UnderdampedDiffusionBridges @ 6dbce6c) for 2-D lattice phi^4, as run by our wrapper
nhmc-phi4-bench/udb_bench.py for the '+Z2' rows of results_sync/table_udb_z2.csv.

Ported files (read line by line): algorithms/overdamped/{cmcd,dbs,od_integrators,overdamped_rnd}.py,
algorithms/common/{init_diffusion_model,utils,scheduler,learning_rate_scheduler}.py,
algorithms/common/models/pisgrad_net.py, targets/phi4.py, configs/{base_conf,algorithm/cmcd,algorithm/dbs,
algorithm/model/pisgrad_net,algorithm/dt_schedule/cosine,target/phi4}.yaml, udb_bench.py (Z2 + final evaluation).

Run config of the table rows (job_udb.sh + runs_z2_L8.txt: 'cmcd/dbs <kappa> 0 80 10000 0.1 102400 1 64'):
  target phi4 dim 64 (L = 8), lambda .022, T = num_steps = 80, iters 10000, batch 500, loss 'elbo', Adam lr 5e-4
  (warmup 'const' with warmup_iters 60000 > iters -> constant lr), optax.chain(zero_nans, clip(1.0) elementwise,
  adam), dt = .1 (learned scalar, softplus-parametrised) x cosine schedule (min .01, s .008, pow 2),
  per-dim learned friction (init 1), learned betas (T raw values, b = cumsum(softplus)/sum, b[0] and b[-1]
  stop-gradient), learned diagonal Gaussian prior (init N(0, I); with Z2 the mean is pinned to 0), PISGRADNet
  num_layers 2, num_hid 128, use_target_score False, last layer kernel 1e-8; z2 = True: every net replaced by
  0.5 (f(x) - f(-x)); final SNIS evaluation N = 102400 in chunks of 4096 (float32), no z2mix.

Kernels (per step k = 0..T-1, eta_k = dt_k / friction (per dim), scale_k = sqrt(2 eta_k)):
  drift_k(x) = b_k grad log gamma(x) + (1 - b_k) grad log pi_0(x),  b_k = b[min(k, T-1)]  (JAX gather clamps)
  F_k(x_{k+1} | x_k) = N(x_k + eta_k (drift_k(x_k) + f(x_k, k)), 2 eta_k)
  B_k(x_k | x_{k+1}) = N(x_{k+1} + eta_k (drift_{k+1}(x_{k+1}) + g(x_{k+1}, k+1)), 2 eta_k)
     CMCD: g = -f (same net);  DBS: g = separate backward net
  log w = log gamma(x_T) - log pi_0(x_0) + sum_k [log B_k - log F_k],  gamma = exp(-S),  E_Q[w] = Z.
  loss = -mean(log w) with reparametrised gradients through the whole path (stop_grad=False).

Deviations from the official code (all numerical-equivalent or bookkeeping; none changes the model/loss):
  * PyTorch instead of JAX/flax/optax; float32 by default. RNG streams differ (seed-for-seed reproduction is
    impossible; reproduce within seed spread).
  * Network outputs and drifts at (x_{k+1}, k+1) are computed once and reused by B_k and F_{k+1} (JAX computes
    them twice with identical inputs). The first state-net layer is applied as W_x x + (W_t emb_k + b) and the
    time embedding is evaluated once per step for the whole batch (it does not depend on x).
  * With Z2 the prior mean is a zero buffer instead of a parameter that is reset to 0 after every Adam step
    (identical forward pass; optax clip/zero_nans/adam are elementwise, so the other parameters are unaffected).
  * mass_std (unused by overdamped samplers) and time_coder_grad (unused with use_target_score False; flax creates
    no parameters for it) are omitted.
  * The in-training logging evaluations (n_evals 20 x 2000 paths) run inside torch.random.fork_rng and do not
    perturb the training stream; they only log ELBO / ESS.
  * Checkpointing every --ckpt-min minutes (model, Adam state, CPU and CUDA RNG states, timers, history).
  * Constant buffers (Fourier coefficients, dt schedule) are derived from float64 masters (exact in float64 evals).
Not ported because inactive in the run config: target_score_max_norm clipping (None in configs/target/phi4.yaml),
per_step_dt, learn_{betas,dt,friction,prior}=False branches, use_path_gradient, the log_var loss, the linear warmup /
cosine LR branch (warmup 'const', warmup_iters 60000 > iters), DBS drift 'zero'/'target', use_target_score=True
(time_coder_grad), underdamped variants, mass_std.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fn

sys.dont_write_bytecode = True
# code/ directory of this repository (library modules phi4.py, chainstats.py, diag.py); reference data from the
# extracted Zenodo record ($NHMC_ROOT/results_sync)
CLONE = os.environ.get('NHMC_CODE', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, CLONE)
from phi4 import phi4_action, phi4_force  # noqa: E402  (same convention as the NHMC action)

LOG2PI = math.log(2 * math.pi)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.environ.get('NHMC_ROOT', '.'), 'results_sync')


def softplus(x):
    return Fn.softplus(x)


def inverse_softplus(x):
    return x if x > 20.0 else math.log(math.expm1(x))


def cosine_schedule(step, total_steps, mn=0.01, s=0.008, pw=2):
    """algorithms/common/scheduler.get_cosine_schedule"""
    t = (total_steps - step) / total_steps
    off = 1 + s
    return (1.0 - mn) * math.cos(0.5 * math.pi * (off - t) / off) ** pw + mn


def lecun_normal_(w, fan_in):
    """flax default Dense kernel init: variance_scaling(1, 'fan_in', 'truncated_normal') (truncation at +-2 sd)."""
    std = math.sqrt(1.0 / fan_in) / .87962566103423978
    with torch.no_grad():
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2 * std, b=2 * std)


def gelu(x):
    return Fn.gelu(x, approximate='tanh')         # flax.linen.gelu default approximate=True


class PISGRADNet(nn.Module):
    """algorithms/common/models/pisgrad_net.PISGRADNet with use_target_score=False (the phi4 runs)."""

    def __init__(self, dim, num_layers=2, num_hid=128, outer_clip=1e4):
        super().__init__()
        assert num_layers == 2
        self.dim, self.num_hid, self.outer_clip = dim, num_hid, outer_clip
        self.timestep_phase = nn.Parameter(torch.zeros(1, num_hid))
        self.register_buffer('timestep_coeff', torch.linspace(0.1, 100, num_hid, dtype=torch.float64)[None].to(torch.get_default_dtype()),
                             persistent=False)
        self.tc0 = nn.Linear(2 * num_hid, num_hid)          # time_coder_state.layers_0
        self.tc2 = nn.Linear(num_hid, num_hid)              # time_coder_state.layers_2
        self.st0 = nn.Linear(dim + num_hid, num_hid)        # state_time_net.layers_0.layers_0 (input [x, t_emb])
        self.st1 = nn.Linear(num_hid, num_hid)              # state_time_net.layers_1.layers_0
        self.out = nn.Linear(num_hid, dim)                  # state_time_net.layers_2: kernel 1e-8, bias 0
        for l in (self.tc0, self.tc2, self.st0, self.st1):
            lecun_normal_(l.weight, l.in_features)
            nn.init.zeros_(l.bias)
        nn.init.constant_(self.out.weight, 1e-8)
        nn.init.zeros_(self.out.bias)

    def time_rows(self, steps):
        """t_net1 for a vector of (float) step indices -> (n, num_hid); flax: fourier features then
        Dense-gelu-Dense; additionally pre-applies the time part of the first state layer."""
        t = steps.to(self.timestep_coeff.dtype)[:, None]
        arg = self.timestep_coeff * t + self.timestep_phase
        ff = torch.cat([torch.sin(arg), torch.cos(arg)], -1)
        temb = self.tc2(gelu(self.tc0(ff)))
        return temb @ self.st0.weight[:, self.dim:].T + self.st0.bias     # (n, num_hid)

    def apply_rows(self, x, trow):
        h = gelu(x @ self.st0.weight[:, :self.dim].T + trow)
        h = gelu(self.st1(h))
        return torch.clamp(self.out(h), -self.outer_clip, self.outer_clip)


class UDBPhi4(nn.Module):
    """Overdamped CMCD ('cmcd') or DBS ('dbs', drift 'anneal') sampler on L x L phi^4."""

    def __init__(self, alg, L, kappa, lam=0.022, T=80, dt=0.1, z2=True, init_std=1.0, friction=1.0, num_hid=128):
        super().__init__()
        assert alg in ('cmcd', 'dbs')
        self.alg, self.L, self.dim, self.kappa, self.lam, self.T, self.z2 = alg, L, L * L, kappa, lam, T, z2
        d = self.dim
        self.betas = nn.Parameter(torch.ones(T))
        if z2:
            self.register_buffer('prior_mean', torch.zeros(d))
        else:
            self.prior_mean = nn.Parameter(torch.zeros(d))
        self.prior_std = nn.Parameter(torch.full((d,), inverse_softplus(init_std)))
        self.dt = nn.Parameter(torch.full((1,), inverse_softplus(dt)))
        self.friction = nn.Parameter(torch.full((d,), inverse_softplus(friction)))
        self.fwd = PISGRADNet(d, num_hid=num_hid)
        self.bwd = PISGRADNet(d, num_hid=num_hid) if alg == 'dbs' else None
        self.register_buffer('sched', torch.tensor([cosine_schedule(k, T) for k in range(T)], dtype=torch.float64).to(
            torch.get_default_dtype()), persistent=False)
        self._f64 = dict(timestep_coeff=torch.linspace(0.1, 100, num_hid, dtype=torch.float64)[None],
                         sched=torch.tensor([cosine_schedule(k, T) for k in range(T)], dtype=torch.float64))

    def _apply(self, fn, *args, **kw):
        """Non-persistent constant buffers are re-derived from float64 masters on every .to()/.double()/.cuda(),
        so a float64 evaluation uses float64 Fourier coefficients and schedule (as JAX with x64)."""
        out = super()._apply(fn, *args, **kw)
        dt, dev = self.sched.dtype, self.sched.device
        self.sched = self._f64['sched'].to(device=dev, dtype=dt)
        for net in (self.fwd, self.bwd):
            if net is not None:
                net.timestep_coeff = self._f64['timestep_coeff'].to(device=dev, dtype=net.timestep_coeff.dtype)
        return out

    # ------------------------------------------------------------------ schedule pieces (per path evaluation)
    def coeffs(self):
        """Everything that depends on parameters only: betas b_j (j = 0..T, clamped), eta_k, scale_k, prior
        sigma, time rows of the nets for steps 0..T."""
        b = softplus(self.betas)
        b = torch.cumsum(b, 0) / b.sum()
        b = torch.cat([b[:1].detach(), b[1:-1], b[-1:].detach()])
        bj = torch.cat([b, b[-1:]])                                   # index T -> T-1 (JAX gather clamps)
        dt = softplus(self.dt) * self.sched                           # (T,)
        eta = dt[:, None] / softplus(self.friction)[None, :]          # (T, d)
        steps = torch.arange(self.T + 1, device=b.device, dtype=b.dtype)
        c = dict(b=bj, eta=eta, scale=torch.sqrt(2 * eta), sig=softplus(self.prior_std), mu=self.prior_mean,
                 frow=self.fwd.time_rows(steps))
        if self.bwd is not None:
            c['brow'] = self.bwd.time_rows(steps)
        return c

    # ------------------------------------------------------------------ densities
    def log_gamma(self, x):
        return -phi4_action(x.view(-1, self.L, self.L), lam=self.lam, kappa=self.kappa)

    def score_gamma(self, x):
        return phi4_force(x.view(-1, self.L, self.L), lam=self.lam, kappa=self.kappa).reshape(x.shape)

    def prior_logpdf(self, x, c):
        z = (x - c['mu']) / c['sig']
        return (-0.5 * z * z - torch.log(c['sig']) - 0.5 * LOG2PI).sum(-1)

    def prior_sample(self, n, c, device, dtype):
        return c['mu'] + c['sig'] * torch.randn(n, self.dim, device=device, dtype=dtype)

    def _net(self, net, rows, x, j):
        if not self.z2:
            return net.apply_rows(x, rows[j])
        o = net.apply_rows(torch.cat([x, -x], 0), rows[j])          # Z2PISGRADNet: 0.5 (f(x) - f(-x))
        n = x.shape[0]
        return 0.5 * (o[:n] - o[n:])

    def point(self, x, j, c, need_f=True, need_b=True):
        """drift_j(x), forward control f(x, j), backward control g(x, j) at point x with step index j."""
        beta = c['b'][j]
        drift = beta * self.score_gamma(x) + (1 - beta) * (-(x - c['mu']) / c['sig'] ** 2)
        f = g = None
        if self.alg == 'cmcd':
            if need_f or need_b:
                f = self._net(self.fwd, c['frow'], x, j)
                g = -f
        else:
            if need_f:
                f = self._net(self.fwd, c['frow'], x, j)
            if need_b:
                g = self._net(self.bwd, c['brow'], x, j)
        return drift, f, g

    @staticmethod
    def lognormal(v, mean, scale):
        """numpyro Independent(Normal(mean, scale), 1).log_prob(v)"""
        z = (v - mean) / scale
        return (-0.5 * z * z - torch.log(scale) - 0.5 * LOG2PI).sum(-1)

    # ------------------------------------------------------------------ forward sampling (training / SNIS)
    def sample_paths(self, n, device, dtype, keep_path=False):
        """official per_sample_rnd + integrate_EM, batched. Returns log w (B,), x_T (B, d), path (T+1, B, d)|None.
        Differentiable (reparametrised) when grad is enabled."""
        c = self.coeffs()
        x = self.prior_sample(n, c, device, dtype)
        logw = -self.prior_logpdf(x, c)
        drift, f, _ = self.point(x, 0, c, need_b=False)
        path = [x] if keep_path else None
        for k in range(self.T):
            e, sc = c['eta'][k], c['scale'][k]
            fm = e * drift + e * f
            xn = x + fm + sc * torch.randn_like(x)
            drift_n, f_n, g_n = self.point(xn, k + 1, c, need_f=(k + 1 < self.T))
            bm = e * (drift_n + g_n)
            dx = xn - x
            logw = logw + self.lognormal(-dx, bm, sc) - self.lognormal(dx, fm, sc)
            x, drift, f = xn, drift_n, f_n
            if keep_path:
                path.append(x)
        logw = logw + self.log_gamma(x)
        return logw, x, (torch.stack(path, 0) if keep_path else None)

    # ------------------------------------------------------------------ path-space densities (z2mix, RT, tests)
    def path_logs(self, path, c=None):
        """log Q(path) = log pi_0(x_0) + sum log F_k, log P(path) = log gamma(x_T) + sum log B_k for a path
        (T+1, B, d); same formula as udb_bench.make_path_logs."""
        c = self.coeffs() if c is None else c
        terms = [self.point(path[j], j, c) for j in range(self.T + 1)]
        lf = lb = 0.0
        for k in range(self.T):
            e, sc = c['eta'][k], c['scale'][k]
            d0, f0, _ = terms[k]
            d1, _, g1 = terms[k + 1]
            dx = path[k + 1] - path[k]
            lf = lf + self.lognormal(dx, e * d0 + e * f0, sc)
            lb = lb + self.lognormal(-dx, e * (d1 + g1), sc)
        return self.prior_logpdf(path[0], c) + lf, self.log_gamma(path[-1]) + lb

    def per_step_logs(self, path, c=None):
        """(T, B) tensors log F_k and log B_k (for kernel tests)."""
        c = self.coeffs() if c is None else c
        terms = [self.point(path[j], j, c) for j in range(self.T + 1)]
        LF, LB = [], []
        for k in range(self.T):
            e, sc = c['eta'][k], c['scale'][k]
            d0, f0, _ = terms[k]
            d1, _, g1 = terms[k + 1]
            dx = path[k + 1] - path[k]
            LF.append(self.lognormal(dx, e * d0 + e * f0, sc))
            LB.append(self.lognormal(-dx, e * (d1 + g1), sc))
        return torch.stack(LF), torch.stack(LB)

    # ------------------------------------------------------------------ partial paths (round trip)
    def forward_from(self, u, s, c=None):
        """Forward stages s..T-1 from x_s = u. Returns x_T and W_s = -log gamma(x_T) + sum_{k>=s} [log F_k - log B_k]."""
        c = self.coeffs() if c is None else c
        x = u
        drift, f, _ = self.point(x, s, c, need_b=False)
        W = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        for k in range(s, self.T):
            e, sc = c['eta'][k], c['scale'][k]
            fm = e * drift + e * f
            xn = x + fm + sc * torch.randn_like(x)
            drift_n, f_n, g_n = self.point(xn, k + 1, c, need_f=(k + 1 < self.T))
            dx = xn - x
            W = W + self.lognormal(dx, fm, sc) - self.lognormal(-dx, e * (drift_n + g_n), sc)
            x, drift, f = xn, drift_n, f_n
        return x, W - self.log_gamma(x)

    def reverse_to(self, xT, s, c=None, bug=None):
        """Backward chain x_{T-1}, ..., x_s ~ B from x_T. Returns u = x_s and the same W_s as forward_from.
        bug='eta_next' is the NEGATIVE CONTROL of the tests: reverse draws use eta_{k+1} instead of eta_k
        (densities in W stay correct), so the augmented target is wrong."""
        c = self.coeffs() if c is None else c
        y = xT
        drift_y, _, g_y = self.point(y, self.T, c, need_f=False)
        W = -self.log_gamma(xT)
        for k in range(self.T - 1, s - 1, -1):
            e, sc = c['eta'][k], c['scale'][k]
            bm = e * (drift_y + g_y)
            if bug == 'eta_next':
                kk = min(k + 1, self.T - 1)
                e2, sc2 = c['eta'][kk], c['scale'][kk]
                x = y + e2 * (drift_y + g_y) + sc2 * torch.randn_like(y)
            else:
                x = y + bm + sc * torch.randn_like(y)
            drift_x, f_x, g_x = self.point(x, k, c, need_b=(k > s))
            dx = y - x
            W = W + self.lognormal(dx, e * drift_x + e * f_x, sc) - self.lognormal(-dx, bm, sc)
            y, drift_y, g_y = x, drift_x, g_x
        return y, W

    # ------------------------------------------------------------------ JAX parameter import (eval checks)
    def load_flax(self, params):
        """Load params dict from a udb_bench params.msgpack (decoded with load_msgpack)."""
        p = params['params']
        with torch.no_grad():
            self.betas.copy_(torch.as_tensor(p['betas']))
            self.prior_std.copy_(torch.as_tensor(p['prior_std']))
            self.dt.copy_(torch.as_tensor(p['dt']))
            self.friction.copy_(torch.as_tensor(p['friction']))
            if self.z2:
                if np.abs(p['prior_mean']).max() != 0:
                    raise SystemExit('params have a nonzero prior mean (non-Z2 run): pass --no-z2')
            else:
                self.prior_mean.copy_(torch.as_tensor(p['prior_mean']))
            for name, net in (('fwd_params', self.fwd), ('bwd_params', self.bwd)):
                if net is None:
                    continue
                q = p[name]['params']
                net.timestep_phase.copy_(torch.as_tensor(q['timestep_phase']))
                for mod, a, b in ((net.tc0, 'time_coder_state', 'layers_0'), (net.tc2, 'time_coder_state', 'layers_2'),
                                  (net.out, 'state_time_net', 'layers_2')):
                    mod.weight.copy_(torch.as_tensor(q[a][b]['kernel']).T)
                    mod.bias.copy_(torch.as_tensor(q[a][b]['bias']))
                for mod, b in ((net.st0, 'layers_0'), (net.st1, 'layers_1')):
                    mod.weight.copy_(torch.as_tensor(q['state_time_net'][b]['layers_0']['kernel']).T)
                    mod.bias.copy_(torch.as_tensor(q['state_time_net'][b]['layers_0']['bias']))


def load_msgpack(path):
    """flax.serialization.msgpack_restore without flax (ext code 1 = ndarray (shape, dtype, buffer))."""
    import msgpack

    def hook(code, data):
        if code == 1:
            shape, dtype, buf = msgpack.unpackb(data)
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return msgpack.ExtType(code, data)
    with open(path, 'rb') as f:
        return msgpack.unpackb(f.read(), ext_hook=hook, raw=False, strict_map_key=False)


# ====================================================================== training
def adam_update_(params, opt):
    """optax.chain(zero_nans(), clip(1.0), adam(lr)) applied to .grad (elementwise), then the Adam step."""
    for p in params:
        if p.grad is not None:
            p.grad.nan_to_num_(nan=0.0, posinf=float('inf'), neginf=float('-inf'))
            p.grad.clamp_(-1.0, 1.0)
    opt.step()


@torch.no_grad()
def quick_eval(m, n, device, dtype, chunk=4096):
    lws, Ms = [], []
    for i in range(0, n, chunk):
        lw, x, _ = m.sample_paths(min(chunk, n - i), device, dtype)
        lws.append(lw.double().cpu()); Ms.append(x.mean(-1).double().cpu())
    return summarize(torch.cat(lws).numpy(), torch.cat(Ms).numpy(), m.dim)


def summarize(logw, M, V, ref=None):
    """udb_bench final-evaluation statistics (SNIS)."""
    finite = np.isfinite(logw) & np.isfinite(M)
    M = np.where(finite, M, 0.0)
    lw = np.where(finite, logw, -np.inf)
    mx = lw.max()
    w = np.exp(lw - mx)
    logZ = float(mx + np.log(w.mean()))
    wn = w / w.sum()
    n = logw.size
    absm, m2 = float(np.sum(wn * np.abs(M))), float(np.sum(wn * M ** 2))
    out = dict(n=int(n), n_nonfinite=int((~finite).sum()), logZ=logZ, ess_frac=float(1.0 / np.sum(wn ** 2) / n),
               abs_mag_rw=absm, m2_rw=m2, chi_rw=float(V * (m2 - absm ** 2)), abs_mag_raw=float(np.abs(M).mean()),
               frac_pos_raw=float((M > 0).mean()), frac_pos_rw=float(np.sum(wn * (M > 0))),
               elbo=float(np.mean(logw[finite])), var_logw=float(np.var(logw[finite])))
    if ref is not None:
        out.update(dlogZ=abs(logZ - ref['logZ_mean']), logZ_ref=ref['logZ_mean'], absM_ref=ref['abs_mag'],
                   chi_ref=ref['chi'])
    return out


def rng_state(device):
    st = dict(cpu=torch.get_rng_state())
    if device.type == 'cuda':
        st['cuda'] = torch.cuda.get_rng_state(device)
    return st


def set_rng_state(st, device):
    torch.set_rng_state(st['cpu'].cpu())
    if device.type == 'cuda' and 'cuda' in st:
        torch.cuda.set_rng_state(st['cuda'].cpu(), device)


def save_ckpt(path, m, opt, it, train_sec, hist, device, args):
    tmp = path + '.tmp'
    torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), it=it, train_sec=train_sec, hist=hist,
                    rng=rng_state(device), args=args), tmp)
    os.replace(tmp, path)


def build(a):
    return UDBPhi4(a.alg, a.L, a.kappa, lam=a.lam, T=a.T, dt=a.dt, z2=a.z2, num_hid=a.num_hid)


def train(a):
    device = torch.device(a.device)
    dtype = torch.float64 if a.dtype == 'float64' else torch.float32
    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, 'ckpt.pt')
    torch.manual_seed(a.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    m = build(a).to(device=device, dtype=dtype)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    it0, train_sec, hist = 0, 0.0, []
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, map_location='cpu', weights_only=False)
        m.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        it0, train_sec, hist = st['it'], st['train_sec'], st['hist']
        set_rng_state(st['rng'], device)
        print(f'resumed from {ck} at iteration {it0}', flush=True)
    eval_freq = max(a.iters // a.n_evals, 1) if a.n_evals > 0 else None
    t_ck = time.time()
    for it in range(it0, a.iters):
        if eval_freq and (it % eval_freq == 0 or it == a.iters - 1):
            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                torch.manual_seed(777_000 + it)
                ev = quick_eval(m, a.eval_samples, device, dtype)
            hist.append(dict(it=it, train_sec=train_sec, elbo=ev['elbo'], logZ=ev['logZ'], ess=ev['ess_frac'],
                             var_logw=ev['var_logw'], nonfinite=ev['n_nonfinite']))
            print(json.dumps(hist[-1]), flush=True)
        t0 = time.time()
        logw, _, _ = m.sample_paths(a.batch, device, dtype)
        loss = -logw.mean()
        opt.zero_grad(set_to_none=False)
        loss.backward()
        adam_update_(params, opt)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_sec += time.time() - t0
        if a.stop_at is not None and it + 1 == a.stop_at:                 # test hook: emulate a kill
            save_ckpt(ck, m, opt, it + 1, train_sec, hist, device, vars(a))
            print(f'stopped at {it + 1} (test)', flush=True)
            return None
        if time.time() - t_ck > 60 * a.ckpt_min:
            save_ckpt(ck, m, opt, it + 1, train_sec, hist, device, vars(a))
            t_ck = time.time()
            if device.type == 'cuda':
                print(f'ckpt it {it + 1} peak_mem_GB {torch.cuda.max_memory_allocated(device) / 2**30:.3f}', flush=True)
    save_ckpt(ck, m, opt, a.iters, train_sec, hist, device, vars(a))
    torch.save(dict(model=m.state_dict(), args=vars(a), train_sec=train_sec), os.path.join(a.out, 'model.pt'))
    return m


def final_eval(m, a, device, dtype, train_sec=0.0, tag=''):
    """udb_bench final frozen evaluation: N paths in chunks, SNIS stats; optional z2mix (non-Z2 models)."""
    ref = json.load(open(os.path.join(RESULTS, 'logz_ref.json')))['results'].get(f'{a.kappa:.2f}') if a.L == 8 else None
    torch.manual_seed(10_000 + a.seed)
    logw, M, lwm, Mm, dc = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        c = m.coeffs()
        p_plus = None
        if a.z2mix:
            torch.manual_seed(20_000 + a.seed)
            _, xp, _ = m.sample_paths(a.chunk, device, dtype)
            p_plus = float(np.clip((xp.mean(-1) > 0).double().mean().item(), 1e-3, 1 - 1e-3))
            torch.manual_seed(10_000 + a.seed)
        for i in range(0, a.n_final, a.chunk):
            b = min(a.chunk, a.n_final - i)
            lw, x, path = m.sample_paths(b, device, dtype, keep_path=a.z2mix)
            mm = x.mean(-1)
            logw.append(lw.double().cpu()); M.append(mm.double().cpu())
            if a.z2mix:
                lq, lp = m.path_logs(path, c)
                lqm, lpm = m.path_logs(-path, c)
                dc.append((lp - lq - lw).abs().double().cpu())
                pos = mm > 0
                lr = torch.where(pos, math.log(p_plus), math.log(1 - p_plus)).to(lp.dtype)
                l1 = torch.where(pos, math.log(1 - p_plus), math.log(p_plus)).to(lp.dtype)
                lwm.append((torch.logaddexp(lr + lp, l1 + lpm) - (torch.logaddexp(lq, lqm) - math.log(2.0))).double().cpu())
                sg = torch.where(torch.rand(b, device=device) < 0.5, 1.0, -1.0).to(mm.dtype)
                Mm.append((sg * mm).double().cpu())
    out = dict(algorithm=a.alg + ('+Z2' if a.z2 else ''), kappa=a.kappa, seed=a.seed, L=a.L, T=a.T, iters=a.iters,
               batch=a.batch, dtype=str(dtype), train_sec=train_sec, eval_sec=time.time() - t0, tag=tag)
    out.update(summarize(torch.cat(logw).numpy(), torch.cat(M).numpy(), m.dim, ref))
    extra = {}
    if a.z2mix:
        out['z2mix'] = summarize(torch.cat(lwm).numpy(), torch.cat(Mm).numpy(), m.dim, ref)
        out['z2mix'].update(pilot_p_plus=p_plus, path_recompute_max_abs_err=float(torch.cat(dc).max()))
        extra = dict(log_w_mix=torch.cat(lwm).numpy(), M_mix=torch.cat(Mm).numpy())
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, f'final_eval{tag}.npz'), log_w=torch.cat(logw).numpy(), M=torch.cat(M).numpy(), **extra)
    with open(os.path.join(a.out, f'final_eval{tag}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('FINAL', json.dumps(out), flush=True)
    return out


def parser():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--alg', choices=['cmcd', 'dbs'], default='cmcd')
    pa.add_argument('--kappa', type=float, required=True)
    pa.add_argument('--lam', type=float, default=0.022)
    pa.add_argument('--L', type=int, default=8)
    pa.add_argument('--T', type=int, default=80)
    pa.add_argument('--dt', type=float, default=0.1)
    pa.add_argument('--iters', type=int, default=10000)
    pa.add_argument('--batch', type=int, default=500)
    pa.add_argument('--lr', type=float, default=5e-4)
    pa.add_argument('--num-hid', type=int, default=128)
    pa.add_argument('--no-z2', dest='z2', action='store_false')
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--n-evals', type=int, default=20)
    pa.add_argument('--eval-samples', type=int, default=2000)
    pa.add_argument('--n-final', type=int, default=102400)
    pa.add_argument('--chunk', type=int, default=4096)
    pa.add_argument('--z2mix', action='store_true', help='also the udb_bench z2mix estimator (for non-Z2 models)')
    pa.add_argument('--eval-dtype', default=None, help='float32/float64 for the final evaluation (default: --dtype)')
    pa.add_argument('--dtype', default='float32')
    pa.add_argument('--device', default='cpu')
    pa.add_argument('--threads', type=int, default=4)
    pa.add_argument('--ckpt-min', type=float, default=10.0)
    pa.add_argument('--resume', action='store_true')
    pa.add_argument('--stop-at', type=int, default=None, help='test hook: checkpoint and exit after this iteration')
    pa.add_argument('--load-jax', default=None, help='eval only: params.msgpack from udb_bench (sets iters=0)')
    pa.add_argument('--load', default=None, help='eval only: model.pt / ckpt.pt of this script')
    pa.add_argument('--out', required=True)
    return pa


def main():
    a = parser().parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    edt = torch.float64 if (a.eval_dtype or a.dtype) == 'float64' else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.load_jax or a.load:
        train_sec = 0.0
        if a.load_jax:
            m = build(a)
            m.load_flax(load_msgpack(a.load_jax))
        else:
            st = torch.load(a.load, map_location='cpu', weights_only=False)
            ar = st['args']                                   # architecture/schedule from the training run
            for k in ('alg', 'L', 'T', 'dt', 'z2', 'num_hid', 'lam'):
                setattr(a, k, ar[k])
            if abs(ar['kappa'] - a.kappa) > 1e-12:
                print(f"note: model trained at kappa={ar['kappa']}, evaluated at kappa={a.kappa}", flush=True)
            m = build(a)
            m.load_state_dict(st['model']); train_sec = st.get('train_sec', 0.0)
        m = m.to(device=device, dtype=edt).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        final_eval(m, a, device, edt, train_sec, tag=('_jax' if a.load_jax else '') + ('_f64' if edt == torch.float64 else ''))
        return
    m = train(a)
    if m is None:
        return
    st = torch.load(os.path.join(a.out, 'model.pt'), map_location='cpu', weights_only=False)
    m = m.to(dtype=edt).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    final_eval(m, a, device, edt, st['train_sec'])


if __name__ == '__main__':
    main()
